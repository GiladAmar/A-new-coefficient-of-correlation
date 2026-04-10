#!/usr/bin/env python3
"""
Ichor – Fluid Dynamics Game  (Mac port, GPU-accelerated)

Clean-room reimplementation of the original Flash game by Soylent Software
published on fun-motion.com.

Physics: Jos Stam, "Real-Time Fluid Dynamics for Games", GDC 2003
         with density-gradient driven Gaussian noise (as documented).

Controls
--------
  Menu  :  1 = Single Player (mouse)   2 = Duel (WASD vs arrow keys)
  Single:  Move mouse to steer         ESC = quit
  Duel  :  P1 WASD  P2 arrow keys      ESC = quit
  After game: SPACE = restart          ESC = quit
"""

import sys, math, random, time
import numpy as np
import pygame
from pygame.locals import *
import moderngl

# ── Grid / display ─────────────────────────────────────────────────────────────
N    = 128          # interior grid cells per axis
S    = N + 2        # array size (+ boundary ring)
CELL = 6            # screen pixels per grid cell
W    = N * CELL     # 768
H    = N * CELL     # 768
FPS  = 60

# ── Fluid physics ──────────────────────────────────────────────────────────────
DT          = 0.13
DIFF        = 4e-6
VISC        = 8e-7
ITER        = 14
NOISE_STR   = 0.022
DENS_DECAY  = 0.9988
VEL_DECAY   = 0.9994

# ── Gameplay ───────────────────────────────────────────────────────────────────
SPEED         = 115.0   # px / s
INJECT_R      = 4       # radius in cells
INJECT_AMT    = 220.0   # density / s
INJECT_FORCE  = 50.0    # velocity impulse
LOSE_FRAC     = 0.55    # opp fraction at your position → loss
LOSE_MIN      = 0.12    # min total density before checking

# ── GLSL shaders ───────────────────────────────────────────────────────────────
VERT_SRC = """
#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FLUID_FRAG = """
#version 330 core
uniform sampler2D u_d0;
uniform sampler2D u_d1;
uniform vec2  u_p0;
uniform vec2  u_p1;
uniform int   u_winner;   // -1 = none, 0 or 1
uniform float u_flash;    // 0..1
in  vec2 v_uv;
out vec4 f_col;

vec3 tonemap(vec3 c) { return c / (c + 1.0); }

void main() {
    float d0 = max(texture(u_d0, v_uv).r, 0.0);
    float d1 = max(texture(u_d1, v_uv).r, 0.0);

    vec3 c0 = vec3(1.00, 0.18, 0.04);   // player 0 : red-orange
    vec3 c1 = vec3(0.04, 0.48, 1.00);   // player 1 : blue
    vec3 bg = vec3(0.018, 0.018, 0.065);

    vec3 col = bg;
    col += c0 * pow(d0, 0.55) * 2.9 + c0 * d0 * d0 * 5.5;
    col += c1 * pow(d1, 0.55) * 2.9 + c1 * d1 * d1 * 5.5;

    // Player dots + glow  (smoothstep requires edge0 < edge1)
    float r0 = length(v_uv - u_p0);
    float r1 = length(v_uv - u_p1);
    col  = mix(col, c0 * 5.0, 1.0 - smoothstep(0.008, 0.022, r0));
    col += c0 * 0.55 * (1.0 - smoothstep(0.022, 0.065, r0));
    col  = mix(col, c1 * 5.0, 1.0 - smoothstep(0.008, 0.022, r1));
    col += c1 * 0.55 * (1.0 - smoothstep(0.022, 0.065, r1));

    // Winner flash
    if (u_winner == 0) col = mix(col, c0 * 2.2, u_flash * 0.40);
    if (u_winner == 1) col = mix(col, c1 * 2.2, u_flash * 0.40);

    f_col = vec4(tonemap(col), 1.0);
}
"""

OVERLAY_FRAG = """
#version 330 core
uniform sampler2D u_tex;
in  vec2 v_uv;
out vec4 f_col;
void main() {
    f_col = texture(u_tex, v_uv);
}
"""

# ── Fluid solver ───────────────────────────────────────────────────────────────

class Fluid:
    """
    Jos Stam stable-fluids solver.
    Two density fields (one per player) share a single velocity field.
    """
    def __init__(self):
        z = lambda: np.zeros((S, S), dtype=np.float32)
        self.vx  = z(); self.vy  = z()
        self.vx0 = z(); self.vy0 = z()
        self.d   = [z(), z()]
        self.d0  = [z(), z()]

    # ── boundary conditions ────────────────────────────────────────────────────
    def _bnd(self, b, x):
        x[0,   1:S-1] = -x[1,    1:S-1] if b == 1 else x[1,    1:S-1]
        x[S-1, 1:S-1] = -x[S-2,  1:S-1] if b == 1 else x[S-2,  1:S-1]
        x[1:S-1,    0] = -x[1:S-1,    1] if b == 2 else x[1:S-1,    1]
        x[1:S-1, S-1] = -x[1:S-1,  S-2] if b == 2 else x[1:S-1,  S-2]
        x[0,   0]   = .5*(x[1, 0]   + x[0,   1])
        x[0,   S-1] = .5*(x[1, S-1] + x[0,   S-2])
        x[S-1, 0]   = .5*(x[S-2, 0] + x[S-1, 1])
        x[S-1, S-1] = .5*(x[S-2, S-1]+x[S-1, S-2])

    # ── Gauss-Seidel (vectorised Jacobi) ──────────────────────────────────────
    def _solve(self, b, x, x0, a, c):
        ci = 1.0 / c
        for _ in range(ITER):
            x[1:S-1, 1:S-1] = (x0[1:S-1, 1:S-1] + a * (
                x[0:S-2, 1:S-1] + x[2:S,   1:S-1] +
                x[1:S-1, 0:S-2] + x[1:S-1, 2:S  ])) * ci
            self._bnd(b, x)

    def _diffuse(self, b, x, x0, coeff):
        a = DT * coeff * N * N
        self._solve(b, x, x0, a, 1.0 + 4.0 * a)

    # ── pressure projection ────────────────────────────────────────────────────
    def _project(self, vx, vy, p, div):
        h = 1.0 / N
        div[1:S-1, 1:S-1] = -0.5 * h * (
            vx[2:S,   1:S-1] - vx[0:S-2, 1:S-1] +
            vy[1:S-1, 2:S  ] - vy[1:S-1, 0:S-2])
        p[...] = 0.0
        self._bnd(0, div); self._bnd(0, p)
        self._solve(0, p, div, 1.0, 4.0)
        vx[1:S-1, 1:S-1] -= 0.5 * (p[2:S,   1:S-1] - p[0:S-2, 1:S-1]) / h
        vy[1:S-1, 1:S-1] -= 0.5 * (p[1:S-1, 2:S  ] - p[1:S-1, 0:S-2]) / h
        self._bnd(1, vx); self._bnd(2, vy)

    # ── semi-Lagrangian advection ──────────────────────────────────────────────
    def _advect(self, b, d, d0, vx, vy):
        dt0 = DT * N
        i = np.arange(1, S-1, dtype=np.float32)
        j = np.arange(1, S-1, dtype=np.float32)
        I, J = np.meshgrid(i, j, indexing='ij')
        x = np.clip(I - dt0 * vx[1:S-1, 1:S-1], 0.5, N + 0.5)
        y = np.clip(J - dt0 * vy[1:S-1, 1:S-1], 0.5, N + 0.5)
        i0 = x.astype(np.int32);  i1 = np.minimum(i0 + 1, S-1)
        j0 = y.astype(np.int32);  j1 = np.minimum(j0 + 1, S-1)
        s1 = x - i0;  s0 = 1.0 - s1
        t1 = y - j0;  t0 = 1.0 - t1
        d[1:S-1, 1:S-1] = (s0*(t0*d0[i0,j0] + t1*d0[i0,j1]) +
                            s1*(t0*d0[i1,j0] + t1*d0[i1,j1]))
        self._bnd(b, d)

    # ── public helpers ─────────────────────────────────────────────────────────
    def _weight_patch(self, cx, cy):
        """Return (i0,i1,j0,j1,w) for a radial injection patch."""
        r = INJECT_R
        i0, i1 = max(1, cx-r), min(S-1, cx+r+1)
        j0, j1 = max(1, cy-r), min(S-1, cy+r+1)
        if i0 >= i1 or j0 >= j1:
            return None
        I, J = np.mgrid[i0:i1, j0:j1]
        dist = np.sqrt((I-cx)**2 + (J-cy)**2).astype(np.float32)
        w = np.maximum(0.0, 1.0 - dist / r) ** 2
        return i0, i1, j0, j1, w

    def add_dens(self, player, cx, cy, amt):
        patch = self._weight_patch(cx, cy)
        if patch is None: return
        i0, i1, j0, j1, w = patch
        self.d[player][i0:i1, j0:j1] += amt * w

    def add_vel(self, cx, cy, dvx, dvy, strength):
        patch = self._weight_patch(cx, cy)
        if patch is None: return
        i0, i1, j0, j1, w = patch
        self.vx[i0:i1, j0:j1] += dvx * strength * w
        self.vy[i0:i1, j0:j1] += dvy * strength * w

    def sample(self, player, cx, cy, r=3):
        x0, x1 = max(0, cx-r), min(S, cx+r+1)
        y0, y1 = max(0, cy-r), min(S, cy+r+1)
        return float(np.mean(self.d[player][x0:x1, y0:y1]))

    # ── simulation step ────────────────────────────────────────────────────────
    def step(self):
        # Density-gradient Gaussian noise (as per original game description)
        total = self.d[0] + self.d[1]
        gx = np.zeros_like(total); gy = np.zeros_like(total)
        gx[1:S-1, 1:S-1] = total[2:S, 1:S-1] - total[0:S-2, 1:S-1]
        gy[1:S-1, 1:S-1] = total[1:S-1, 2:S] - total[1:S-1, 0:S-2]
        noise = np.random.standard_normal((S, S)).astype(np.float32) * NOISE_STR
        self.vx += noise * gx
        self.vy += noise * gy

        # Velocity step
        self.vx0, self.vx = self.vx, self.vx0
        self.vy0, self.vy = self.vy, self.vy0
        self._diffuse(1, self.vx, self.vx0, VISC)
        self._diffuse(2, self.vy, self.vy0, VISC)
        self._project(self.vx, self.vy, self.vx0, self.vy0)
        self.vx0, self.vx = self.vx, self.vx0
        self.vy0, self.vy = self.vy, self.vy0
        self._advect(1, self.vx, self.vx0, self.vx0, self.vy0)
        self._advect(2, self.vy, self.vy0, self.vx0, self.vy0)
        self._project(self.vx, self.vy, self.vx0, self.vy0)

        # Density step
        for p in (0, 1):
            self.d0[p], self.d[p] = self.d[p], self.d0[p]
            self._diffuse(0, self.d[p], self.d0[p], DIFF)
            self.d0[p], self.d[p] = self.d[p], self.d0[p]
            self._advect(0, self.d[p], self.d0[p], self.vx, self.vy)
            self.d[p] *= DENS_DECAY

        self.vx *= VEL_DECAY
        self.vy *= VEL_DECAY


# ── Player ─────────────────────────────────────────────────────────────────────

class Player:
    def __init__(self, pid, x, y):
        self.pid  = pid
        self.x    = float(x)
        self.y    = float(y)
        self.alive = True

    @property
    def gx(self):  # grid x
        return max(1, min(S-2, int(self.x / CELL)))

    @property
    def gy(self):  # grid y
        return max(1, min(S-2, int(self.y / CELL)))

    def move(self, dx, dy, dt, fluid):
        spd = math.sqrt(dx*dx + dy*dy)
        if spd > 1e-6:
            ndx, ndy = dx/spd, dy/spd
        else:
            ndx = ndy = 0.0

        self.x = max(CELL*2, min(W - CELL*2, self.x + dx * SPEED * dt))
        self.y = max(CELL*2, min(H - CELL*2, self.y + dy * SPEED * dt))

        cx, cy = self.gx, self.gy
        fluid.add_dens(self.pid, cx, cy, INJECT_AMT * dt)
        if spd > 1e-6:
            fluid.add_vel(cx, cy, ndx, ndy, INJECT_FORCE * dt)

    def is_engulfed(self, fluid):
        """True if this player is surrounded by the opponent's colour."""
        cx, cy = self.gx, self.gy
        opp  = 1 - self.pid
        own  = fluid.sample(self.pid, cx, cy, 4)
        other = fluid.sample(opp,     cx, cy, 4)
        total = own + other
        if total < LOSE_MIN:
            return False
        return (other / total) > LOSE_FRAC


# ── AI player ──────────────────────────────────────────────────────────────────

class AIPlayer(Player):
    """
    Simple AI: attempts to position upstream of the opponent and
    push its own fluid toward them.
    """
    def __init__(self, x, y):
        super().__init__(1, x, y)
        self.tx, self.ty = x, y
        self.rethink = 0.0

    def update(self, opponent, fluid, dt):
        self.rethink -= dt
        if self.rethink <= 0:
            self.rethink = 0.25 + random.random() * 0.35
            ox, oy = opponent.gx, opponent.gy
            ox = max(1, min(S-2, ox))
            oy = max(1, min(S-2, oy))
            vx = fluid.vx[ox, oy]
            vy = fluid.vy[ox, oy]
            spd = math.sqrt(vx*vx + vy*vy) + 1e-6
            # Go upstream of opponent (opposite of local flow)
            offset = 90 + random.random() * 70
            tx = opponent.x - (vx/spd) * offset + random.gauss(0, 25)
            ty = opponent.y - (vy/spd) * offset + random.gauss(0, 25)
            self.tx = max(CELL*3, min(W-CELL*3, tx))
            self.ty = max(CELL*3, min(H-CELL*3, ty))

        dx = self.tx - self.x
        dy = self.ty - self.y
        dist = math.sqrt(dx*dx + dy*dy) + 1e-6
        self.move(dx/dist, dy/dist, dt, fluid)


# ── GPU renderer ───────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx

        # ── fluid programme ───────────────────────────────────────────────────
        self.fluid_prog = ctx.program(
            vertex_shader=VERT_SRC,
            fragment_shader=FLUID_FRAG,
        )
        # ── overlay programme (for text) ──────────────────────────────────────
        self.overlay_prog = ctx.program(
            vertex_shader=VERT_SRC,
            fragment_shader=OVERLAY_FRAG,
        )

        verts = np.array([
            -1,-1, 0,0,  1,-1, 1,0,  1,1, 1,1,  -1,1, 0,1
        ], dtype=np.float32)
        idx = np.array([0,1,2, 0,2,3], dtype=np.int32)
        vbo = ctx.buffer(verts.tobytes())
        ibo = ctx.buffer(idx.tobytes())

        self.fluid_vao = ctx.vertex_array(
            self.fluid_prog,
            [(vbo, '2f 2f', 'in_vert', 'in_uv')],
            ibo)
        self.overlay_vao = ctx.vertex_array(
            self.overlay_prog,
            [(vbo, '2f 2f', 'in_vert', 'in_uv')],
            ibo)

        # Density textures: single-channel float32, N×N
        self.tex0 = ctx.texture((N, N), 1, dtype='f4')
        self.tex1 = ctx.texture((N, N), 1, dtype='f4')
        for t in (self.tex0, self.tex1):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False

        self.fluid_prog['u_d0'].value = 0
        self.fluid_prog['u_d1'].value = 1

        # Overlay RGBA texture (W×H)
        self.overlay_tex = ctx.texture((W, H), 4)
        self.overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.overlay_prog['u_tex'].value = 0

    def _upload_density(self, fluid):
        # fluid.d[p] shape (S,S) – extract interior (N×N), orient for GL (y-up)
        d0 = np.ascontiguousarray(
            np.flipud(fluid.d[0][1:S-1, 1:S-1].T).copy())
        d1 = np.ascontiguousarray(
            np.flipud(fluid.d[1][1:S-1, 1:S-1].T).copy())
        self.tex0.write(d0.tobytes())
        self.tex1.write(d1.tobytes())

    def render_fluid(self, fluid, players, winner, flash):
        self._upload_density(fluid)

        self.fluid_prog['u_p0'].value = (
            players[0].x / W, 1.0 - players[0].y / H)
        self.fluid_prog['u_p1'].value = (
            players[1].x / W, 1.0 - players[1].y / H)
        self.fluid_prog['u_winner'].value = winner
        self.fluid_prog['u_flash'].value  = flash

        self.tex0.use(0)
        self.tex1.use(1)
        self.fluid_vao.render()

    def render_overlay(self, surf):
        """Blit a pygame RGBA surface as a fullscreen alpha overlay."""
        data = pygame.image.tostring(surf, 'RGBA', True)
        self.overlay_tex.write(data)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.overlay_tex.use(0)
        self.overlay_vao.render()
        self.ctx.disable(moderngl.BLEND)


# ── Text helper ────────────────────────────────────────────────────────────────

def make_text_surface(lines, font_big, font_small, p0_col, p1_col):
    """
    Render a list of (text, font, colour) tuples onto a transparent surface.
    lines = [(str, is_big, rgba_tuple), ...]
    Returns a pygame Surface (W×H, RGBA).
    """
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    y = H // 3
    for text, big, col in lines:
        fnt = font_big if big else font_small
        img = fnt.render(text, True, col)
        x = (W - img.get_width()) // 2
        surf.blit(img, (x, y))
        y += img.get_height() + 12
    return surf


# ── Game state machine ─────────────────────────────────────────────────────────

STATE_MENU     = 'menu'
STATE_SINGLE   = 'single'
STATE_DUEL     = 'duel'
STATE_GAMEOVER = 'gameover'


class Game:
    P0_COL = (255, 80,  20,  230)
    P1_COL = (30,  130, 255, 230)
    WHITE  = (240, 240, 240, 230)
    GREY   = (160, 160, 160, 200)

    def __init__(self):
        self.state   = STATE_MENU
        self.fluid   = Fluid()
        self.players = self._spawn_players()
        self.winner  = -1
        self.flash   = 0.0
        self.over_timer = 0.0
        self.clock   = pygame.time.Clock()

    def _spawn_players(self):
        p0 = Player(0, W * 0.28, H * 0.50)
        p1 = Player(1, W * 0.72, H * 0.50)
        return [p0, p1]

    def reset(self, mode):
        self.fluid   = Fluid()
        self.players = self._spawn_players()
        if mode == STATE_SINGLE:
            self.players[1] = AIPlayer(W * 0.72, H * 0.50)
        self.winner    = -1
        self.flash     = 0.0
        self.over_timer = 0.0
        self.state     = mode

    # ── input handling ─────────────────────────────────────────────────────────
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == QUIT:
                return False
            if ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    return False
                if self.state == STATE_MENU:
                    if ev.key == K_1:
                        self.reset(STATE_SINGLE)
                    elif ev.key == K_2:
                        self.reset(STATE_DUEL)
                elif self.state == STATE_GAMEOVER:
                    if ev.key == K_SPACE:
                        self.state = STATE_MENU
        return True

    def _player_input(self, dt):
        """Return (dx,dy) per player from keyboard/mouse."""
        keys = pygame.key.get_pressed()

        # Player 0: WASD
        dx0 = (keys[K_d] - keys[K_a]) * 1.0
        dy0 = (keys[K_s] - keys[K_w]) * 1.0

        # Player 1: arrow keys
        dx1 = (keys[K_RIGHT] - keys[K_LEFT]) * 1.0
        dy1 = (keys[K_DOWN]  - keys[K_UP])   * 1.0

        return (dx0, dy0), (dx1, dy1)

    def _mouse_input(self):
        """Normalised direction from player 0 toward mouse cursor."""
        mx, my = pygame.mouse.get_pos()
        dx = mx - self.players[0].x
        dy = my - self.players[0].y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 5:
            return 0.0, 0.0
        return dx/dist, dy/dist

    # ── update ─────────────────────────────────────────────────────────────────
    def update(self, dt):
        if self.state == STATE_MENU:
            # Slowly evolve the demo fluid
            cx, cy = S//2, S//2
            angle = time.time() * 0.7
            self.fluid.add_vel(cx, cy,
                               math.cos(angle)*0.3, math.sin(angle)*0.3, 8.0)
            self.fluid.add_dens(0, cx-8, cy, 2.0 * dt * 60)
            self.fluid.add_dens(1, cx+8, cy, 2.0 * dt * 60)
            self.fluid.step()
            return

        if self.state == STATE_GAMEOVER:
            self.over_timer -= dt
            self.flash = abs(math.sin(self.over_timer * 3.5)) * 0.8
            self.fluid.step()
            return

        # ── playing ───────────────────────────────────────────────────────────
        p0, p1 = self.players

        if self.state == STATE_SINGLE:
            dx0, dy0 = self._mouse_input()
            p0.move(dx0, dy0, dt, self.fluid)
            p1.update(p0, self.fluid, dt)   # AI
        else:
            (dx0, dy0), (dx1, dy1) = self._player_input(dt)
            p0.move(dx0, dy0, dt, self.fluid)
            p1.move(dx1, dy1, dt, self.fluid)

        self.fluid.step()

        # Win check
        p0_lost = p0.is_engulfed(self.fluid)
        p1_lost = p1.is_engulfed(self.fluid)
        if p0_lost or p1_lost:
            self.winner = 1 if p0_lost else 0
            self.over_timer = 4.0
            self.state = STATE_GAMEOVER

    # ── render ─────────────────────────────────────────────────────────────────
    def render(self, renderer, font_big, font_small):
        renderer.ctx.clear(0.02, 0.02, 0.07)

        renderer.render_fluid(
            self.fluid, self.players, self.winner, self.flash)

        # Overlay text
        if self.state == STATE_MENU:
            surf = make_text_surface([
                ("ICHOR",                       True,  self.WHITE),
                ("1  –  Single Player  (mouse)", False, self.P0_COL),
                ("2  –  Duel   (WASD  vs  ↑↓←→)", False, self.P1_COL),
                ("",                            False, self.GREY),
                ("Engulf your opponent in your colour", False, self.GREY),
            ], font_big, font_small, self.P0_COL, self.P1_COL)
            renderer.render_overlay(surf)

        elif self.state == STATE_GAMEOVER:
            name = "PLAYER 1" if self.winner == 0 else "PLAYER 2"
            col  = self.P0_COL if self.winner == 0 else self.P1_COL
            surf = make_text_surface([
                (f"{name}  WINS",  True,  col),
                ("SPACE  –  menu", False, self.GREY),
            ], font_big, font_small, self.P0_COL, self.P1_COL)
            renderer.render_overlay(surf)

        elif self.state == STATE_SINGLE:
            # Subtle HUD: show controls reminder for first few seconds
            pass

        pygame.display.flip()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    pygame.init()

    # Request OpenGL 3.3 Core Profile (works on Mac via Metal translation)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    screen = pygame.display.set_mode((W, H), OPENGL | DOUBLEBUF)
    pygame.display.set_caption("Ichor")

    ctx      = moderngl.create_context()
    renderer = Renderer(ctx)

    font_big   = pygame.font.SysFont("helvetica", 56, bold=True)
    font_small = pygame.font.SysFont("helvetica", 28)

    game  = Game()
    clock = pygame.time.Clock()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)   # clamp to avoid spiral-of-death

        running = game.handle_events()
        game.update(dt)
        game.render(renderer, font_big, font_small)

    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
