"""
ui/easter_egg.py — Пасхалка «Eye -> Heart» (ASCII 60 FPS анімація на PySide6 + NumPy).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

# --- Параметри сітки та вікна ---
COLS = 104
ROWS = 46
CW = 7
CH = 12
WIDTH = COLS * CW
HEIGHT = ROWS * CH
ASPECT = WIDTH / HEIGHT
TARGET_FPS = 60
CYCLE_DURATION = 52.0
BG_HEX = "#07090e"
NUM_CHARS = 16  # 0..15 градієнт, 16..19 спецсимволи зірок
MAX_CORNER_R = math.hypot(ASPECT, 1.0) + 0.05

_GLYPH_PATTERNS = [
    [".....", ".....", ".....", ".....", ".....", ".....", ".....", ".....", "....."],  # ' '
    [".....", ".....", ".....", ".....", ".....", ".....", ".....", "..##.", "....."],  # '.'
    [".....", ".....", ".....", ".....", ".....", ".....", "..##.", "..#..", ".#..."],  # ','
    [".....", ".....", "..##.", ".....", ".....", ".....", "..##.", ".....", "....."],  # ':'
    [".....", ".....", "..##.", ".....", ".....", ".....", "..##.", "..#..", ".#..."],  # ';'
    [".....", "..#..", ".....", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],  # 'i'
    ["..#..", ".##..", "#.#..", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],  # '1'
    [".....", "..#..", "..#..", "#####", "..#..", "..#..", "..#..", "..#.#", "...#."],  # 't'
    ["...##", "..#..", "..#..", "#####", "..#..", "..#..", "..#..", "..#..", "..#.."],  # 'f'
    ["#....", "#....", "#....", "#....", "#....", "#....", "#....", "#....", "#####"],  # 'L'
    [".###.", "#...#", "#....", "#....", "#....", "#....", "#....", "#...#", ".###."],  # 'C'
    [".###.", "#...#", "#....", "#....", "#.###", "#...#", "#...#", "#...#", ".###."],  # 'G'
    [".###.", "#...#", "#..##", "#.#.#", "#.#.#", "##..#", "#...#", "#...#", ".###."],  # '0'
    [".###.", "#...#", "#...#", "#...#", ".###.", "#...#", "#...#", "#...#", ".###."],  # '8'
    [".#.#.", ".#.#.", "#####", ".#.#.", ".#.#.", ".#.#.", "#####", ".#.#.", ".#.#."],  # '#'
    [".###.", "#...#", "#.###", "#.#.#", "#.###", "#.##.", "#....", "#...#", ".###."],  # '@'
    [".....", "..#..", "#.#.#", ".###.", "..#..", ".###.", "#.#.#", "..#..", "....."],  # '*'
    [".....", "..#..", "..#..", "..#..", "#####", "..#..", "..#..", "..#..", "....."],  # '+'
    [".....", ".....", ".....", ".....", "#####", ".....", ".....", ".....", "....."],  # '-'
    ["..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],  # '|'
]

_CACHE_INITIALIZED = False
_BG_BGRA: Any = None
_GLYPHS: Any = None
_Z_VALS: Any = None
_Z_FLAT: Any = None


def _ensure_tables() -> Any:
    """Ліниво ініціалізує numpy-таблиці гліфів, щоб не уповільнювати запуск головної програми."""
    global _CACHE_INITIALIZED, _BG_BGRA, _GLYPHS, _Z_VALS, _Z_FLAT
    import numpy as np

    if not _CACHE_INITIALIZED:
        _BG_BGRA = np.array([14, 9, 7, 255], dtype=np.uint8)  # B, G, R, A (0xFF для Format_RGB32)
        atlas = np.zeros((len(_GLYPH_PATTERNS), CH, CW), dtype=np.uint8)
        for i, rows in enumerate(_GLYPH_PATTERNS):
            for r, line in enumerate(rows):
                for c, ch in enumerate(line):
                    if ch == "#":
                        atlas[i, r + 2, c + 1] = 1
        _GLYPHS = atlas
        _Z_VALS = np.linspace(1.45, -1.45, 20, dtype=np.float32)[:, None]
        _Z_FLAT = _Z_VALS[:, 0]
        _CACHE_INITIALIZED = True
    return np


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def blink_factor(t: float, t0: float, close_d: float = 0.095, hold_d: float = 0.025, open_d: float = 0.23) -> float:
    if t <= t0 or t >= t0 + close_d + hold_d + open_d:
        return 1.0
    if t < t0 + close_d:
        return 1.0 - smoothstep(t0, t0 + close_d, t)
    if t < t0 + close_d + hold_d:
        return 0.0
    return smoothstep(t0 + close_d + hold_d, t0 + close_d + hold_d + open_d, t)


def eval_gaze(t: float) -> tuple[float, float]:
    keys = (
        (0.0, 0.00, -0.02),
        (2.1, 0.00, 0.00),
        (2.8, -0.14, 0.03),
        (4.3, -0.14, 0.03),
        (4.9, 0.15, 0.02),
        (6.6, 0.15, 0.02),
        (7.3, -0.07, -0.04),
        (8.7, -0.07, -0.04),
        (9.6, 0.00, 0.00),
        (CYCLE_DURATION, 0.00, 0.00),
    )
    for i in range(len(keys) - 1):
        t0, x0, y0 = keys[i]
        t1, x1, y1 = keys[i + 1]
        if t <= t1:
            s = smoothstep(t0, t1, t)
            gx = x0 + (x1 - x0) * s
            gy = y0 + (y1 - y0) * s
            if 1.2 < t < 12.8:
                env = smoothstep(1.2, 2.2, t) * (1.0 - smoothstep(11.6, 12.8, t))
                gx += (0.008 * math.sin(t * 2.7) + 0.004 * math.cos(t * 5.9)) * env
                gy += (0.005 * math.cos(t * 2.1) + 0.003 * math.sin(t * 4.7)) * env
            return gx, gy
    return 0.0, 0.0


def make_stars(count: int = 135, seed: int = 42) -> Dict[str, Any]:
    np = _ensure_tables()
    rng = np.random.default_rng(seed)
    return {
        "x": rng.uniform(-1.55, 1.55, count).astype(np.float32),
        "y": rng.uniform(-1.25, 1.25, count).astype(np.float32),
        "z": rng.uniform(0.18, 1.25, count).astype(np.float32),
        "phase": rng.uniform(0.0, 2.0 * math.pi, count).astype(np.float32),
        "speed": rng.uniform(1.5, 4.2, count).astype(np.float32),
        "is_cross": rng.random(count) < 0.22,
        "size": rng.integers(2, 5, count),
    }


def render_into(
    total_t: float,
    U: Any,
    V: Any,
    stars: Dict[str, Any],
    diff_bgra: Any,
    frame_5d: Any,
) -> None:
    np = _ensure_tables()
    t = total_t % CYCLE_DURATION
    fade_in = smoothstep(0.0, 0.9, total_t)

    # --- Відкриття повік та природне кліпання ---
    if t < 0.6:
        openness = 0.0
    elif t < 1.9:
        openness = smoothstep(0.6, 1.9, t)
    elif t < 50.0:
        openness = 1.0
    elif t < 51.2:
        openness = 1.0 - smoothstep(50.0, 51.2, t)
    else:
        openness = 0.0

    openness *= blink_factor(t, 3.80, 0.095, 0.025, 0.24)
    openness *= blink_factor(t, 7.95, 0.100, 0.030, 0.25)
    openness *= blink_factor(t, 11.25, 0.090, 0.020, 0.21)

    # --- Циклічний зум у зіницю й назад ---
    if t <= 13.5:
        z_prog = 0.0
    elif t <= 25.0:
        z_prog = smoothstep(13.5, 25.0, t)
    elif t <= 43.2:
        z_prog = 1.0
    elif t <= 50.0:
        z_prog = 1.0 - smoothstep(43.2, 50.0, t)
    else:
        z_prog = 0.0

    zoom = math.exp(2.42 * z_prog)
    gaze_x, gaze_y = eval_gaze(t)

    cy_w = (2.0 * math.pi * t) / CYCLE_DURATION
    micro_x = np.float32((0.011 * math.sin(cy_w * 5.0)) / zoom)
    micro_y = np.float32((0.008 * math.cos(cy_w * 4.0)) / zoom)

    wx = U * np.float32(1.0 / zoom) - micro_x
    wy = V * np.float32(1.0 / zoom) - micro_y

    light_adapt = 0.038 * (1.0 - smoothstep(0.8, 2.6, t))
    zoom_dilate = 0.048 * (smoothstep(11.6, 20.0, t) - smoothstep(44.0, 50.2, t))
    hippus = 0.006 * math.sin(cy_w * 17.0) * (1.0 - z_prog)
    pupil_r = np.float32(0.194 + light_adapt + zoom_dilate + hippus)
    iris_r = np.float32(0.545)

    R = np.full((ROWS, COLS), 7.0, dtype=np.float32)
    G = np.full((ROWS, COLS), 9.0, dtype=np.float32)
    B = np.full((ROWS, COLS), 14.0, dtype=np.float32)
    lum = np.zeros((ROWS, COLS), dtype=np.float32)

    # --- 1. ШАР ОКА ---
    if (MAX_CORNER_R / zoom) >= float(pupil_r) - 0.01:
        ex = wx - np.float32(gaze_x)
        ey = wy - np.float32(gaze_y)
        r_dist = np.sqrt(ex * ex + ey * ey)
        in_pupil = r_dist < pupil_r

        eye_half_w = np.float32(1.38)
        ax = np.clip(np.abs(wx) / eye_half_w, 0.0, 1.0)
        arch = np.power(np.maximum(0.0, 1.0 - ax * ax), 0.72)

        lid_follow = np.float32(0.20 * gaze_y * openness)
        seam_y = -0.04 * arch
        y_top = seam_y + np.float32(0.57 * openness) * arch + lid_follow * arch
        y_bot = seam_y - np.float32(0.43 * openness) * arch + np.float32(0.45) * lid_follow * arch

        in_eye = (np.abs(wx) < eye_half_w) & (wy < y_top) & (wy > y_bot) & (openness > 0.01)

        crease_y = y_top + np.float32(0.185 + 0.055 * (1.0 - openness))
        crease_d = wy - crease_y
        crease_shadow = 1.0 - 0.54 * np.exp(-(crease_d * crease_d) / 0.0052) * arch
        lid_highlight = 0.24 * np.exp(-((wy - (y_top + 0.085)) ** 2) / 0.0055) * arch
        under_fold = 1.0 - 0.22 * np.exp(-((wy - (y_bot - 0.14)) ** 2) / 0.008) * arch
        vignette = np.clip(1.05 - 0.22 * (wx * wx * 0.55 + wy * wy * 1.1), 0.34, 1.05)
        skin_tex = 0.94 + 0.06 * np.sin(wy * 42.0 + wx * 3.0)

        skin_l = np.clip((0.68 * crease_shadow * under_fold + lid_highlight) * vignette * skin_tex, 0.12, 1.0)

        lash_w = np.float32(0.0017 if openness > 0.12 else 0.0026)
        lash_top = np.exp(-((wy - y_top) ** 2) / lash_w) * arch
        lash_bot = np.exp(-((wy - y_bot) ** 2) / 0.0009) * arch

        dy_lash = wy - y_top
        lash_len = np.float32(0.085 * (0.35 + 0.65 * openness))
        lash_zone = (dy_lash > 0.0) & (dy_lash < lash_len) & (ax < 0.96)
        lash_tufts = np.zeros((ROWS, COLS), dtype=np.float32)
        if np.any(lash_zone):
            lz_x = wx[lash_zone]
            lz_dy = dy_lash[lash_zone]
            lz_ax = ax[lash_zone]
            strands = 0.45 + 0.55 * np.sin(lz_x * 46.0 - np.sign(lz_x) * lz_dy * 24.0)
            fade_h = np.exp(-((lz_dy / (lash_len + 1e-4)) ** 2) * 2.8)
            lash_tufts[lash_zone] = strands * fade_h * (0.45 + 0.55 * lz_ax) * 0.72

        skin_l *= 1.0 - np.clip(lash_top * 0.95 + lash_bot * 0.55 + lash_tufts, 0.0, 0.95)

        R = 218.0 * skin_l
        G = 136.0 * skin_l
        B = 96.0 * skin_l
        lum = 0.22 + 0.72 * skin_l

        if np.any(in_eye):
            top_dist = np.maximum(0.0, y_top - wy)
            bot_dist = np.maximum(0.0, wy - y_bot)
            lid_shadow = 0.34 + 0.66 * np.clip(top_dist / 0.17, 0.0, 1.0)
            corner_shade = np.clip(1.0 - 0.44 * (ax * ax), 0.42, 1.0)
            corner_pink = np.clip((ax - 0.64) / 0.36, 0.0, 1.0) ** 1.6
            tear_gloss = 0.16 * np.exp(-((bot_dist - 0.022) ** 2) / 0.00045) * arch * (1.0 - z_prog)

            sclera_l = np.clip(0.94 * lid_shadow * corner_shade + tear_gloss, 0.20, 1.0)

            s_ie = sclera_l[in_eye]
            cp_ie = corner_pink[in_eye]
            R[in_eye] = np.clip(244.0 * s_ie, 0.0, 255.0)
            G[in_eye] = np.clip((240.0 - 52.0 * cp_ie) * s_ie, 0.0, 255.0)
            B[in_eye] = np.clip((238.0 - 58.0 * cp_ie) * s_ie, 0.0, 255.0)
            lum[in_eye] = 0.35 + 0.65 * s_ie

            in_iris = in_eye & (r_dist < iris_r) & (~in_pupil)
            if np.any(in_iris):
                exi = ex[in_iris]
                eyi = ey[in_iris]
                ri = r_dist[in_iris]
                ang = np.arctan2(eyi, exi)
                rn = np.clip((ri - pupil_r) / (iris_r - pupil_r), 0.0, 1.0)

                fibers = np.clip(
                    0.48
                    + 0.22 * np.sin(ang * 24.0 + rn * 3.8 + np.float32(0.18 * math.sin(cy_w * 3.0)))
                    + 0.16 * np.sin(ang * 47.0 - rn * 5.2)
                    + 0.14 * np.cos(ang * 73.0 + rn * 8.5),
                    0.0,
                    1.0,
                )
                collarette = np.exp(-((rn - 0.23) ** 2) / 0.024) * 0.38
                limbus = 1.0 - 0.86 * np.power(np.clip((rn - 0.72) / 0.28, 0.0, 1.0), 1.8)
                inner_edge = np.clip(rn / 0.09, 0.0, 1.0)

                caustic = 0.22 * np.clip((-exi * 0.65 - eyi * 0.75) / (iris_r + 1e-4), 0.0, 1.0) * (1.0 - z_prog * 0.7)

                iris_lid_sh = 1.0 - (1.0 - lid_shadow[in_iris]) * np.float32(
                    max(0.0, 1.0 - smoothstep(1.5, 4.5, zoom))
                )
                iris_val = np.clip(
                    (0.32 + 0.64 * fibers + collarette + caustic) * limbus * inner_edge * iris_lid_sh,
                    0.05,
                    1.0,
                )

                R[in_iris] = np.clip(255.0 * np.power(iris_val, 0.76), 0.0, 255.0)
                G[in_iris] = np.clip(184.0 * np.power(iris_val, 1.16) + 30.0 * collarette * limbus, 0.0, 255.0)
                B[in_iris] = np.clip(58.0 * np.power(iris_val, 1.92), 0.0, 255.0)
                lum[in_iris] = 0.20 + 0.78 * iris_val

            pupil_mask = in_eye & in_pupil
            R[pupil_mask] = 7.0
            G[pupil_mask] = 9.0
            B[pupil_mask] = 14.0
            lum[pupil_mask] = 0.0

            glint_fade = 1.0 - smoothstep(1.8, 4.0, zoom)
            if glint_fade > 0.01:
                g1 = np.exp(-(((ex - 0.15) ** 2 + (ey - 0.12) ** 2) / 0.0072))
                g2 = 0.52 * np.exp(-(((ex + 0.14) ** 2 + (ey + 0.11) ** 2) / 0.0024))
                glint = np.clip((g1 + g2) * np.float32(glint_fade), 0.0, 1.0)
                g_apply = in_eye & (glint > 0.08)
                if np.any(g_apply):
                    ga = glint[g_apply]
                    R[g_apply] = np.minimum(255.0, R[g_apply] + 255.0 * ga)
                    G[g_apply] = np.minimum(255.0, G[g_apply] + 255.0 * ga)
                    B[g_apply] = np.minimum(255.0, B[g_apply] + 255.0 * ga)
                    lum[g_apply] = np.maximum(lum[g_apply], ga)
        else:
            pupil_mask = np.zeros((ROWS, COLS), dtype=bool)
    else:
        pupil_mask = np.ones((ROWS, COLS), dtype=bool)

    star_override = np.zeros((ROWS, COLS), dtype=np.int32)

    # --- 2. КОСМОС І ЗІРКИ ---
    space_alpha = smoothstep(1.25, 2.9, zoom)
    if space_alpha > 0.01 and np.any(pupil_mask):
        if t <= 42.5:
            drift = np.float32(max(0.0, t - 15.0) * 0.055)
        else:
            drift = np.float32((42.5 - 15.0) * 0.055 - (t - 42.5) * 0.14)

        sz = (stars["z"] - drift) % 1.08 + 0.18
        sx = stars["x"] / sz
        sy = stars["y"] / sz

        cols_s = np.rint((sx / (2.0 * ASPECT) + 0.5) * COLS - 0.5).astype(np.int32)
        rows_s = np.rint((0.5 - sy * 0.5) * ROWS - 0.5).astype(np.int32)

        twinkle = 0.55 + 0.45 * np.sin(np.float32(t) * stars["speed"] + stars["phase"])
        brightness = twinkle * np.clip(1.25 - sz * 0.75, 0.25, 1.0) * np.float32(space_alpha)

        for i in range(len(sz)):
            r0 = int(rows_s[i])
            c0 = int(cols_s[i])
            if r0 < 0 or r0 >= ROWS or c0 < 0 or c0 >= COLS or not pupil_mask[r0, c0]:
                continue
            br = float(brightness[i])
            if br < 0.14:
                continue

            if stars["is_cross"][i]:
                c_val = min(255.0, 255.0 * br)
                R[r0, c0] = c_val
                G[r0, c0] = c_val
                B[r0, c0] = min(255.0, c_val * 1.04)
                lum[r0, c0] = max(lum[r0, c0], br)
                star_override[r0, c0] = 17  # '+'

                arm_x = int(stars["size"][i])
                arm_y = max(1, arm_x - 1)

                for dx in range(1, arm_x + 1):
                    fall = br * (1.0 - dx / (arm_x + 1.2))
                    if fall < 0.12:
                        break
                    for nc in (c0 - dx, c0 + dx):
                        if 0 <= nc < COLS and pupil_mask[r0, nc] and fall > lum[r0, nc]:
                            v_col = 235.0 * fall
                            R[r0, nc] = v_col
                            G[r0, nc] = v_col
                            B[r0, nc] = min(255.0, v_col * 1.06)
                            lum[r0, nc] = fall
                            star_override[r0, nc] = 18 if dx < arm_x else 1

                for dy in range(1, arm_y + 1):
                    fall = br * (1.0 - dy / (arm_y + 1.1))
                    if fall < 0.12:
                        break
                    for nr in (r0 - dy, r0 + dy):
                        if 0 <= nr < ROWS and pupil_mask[nr, c0] and fall > lum[nr, c0]:
                            v_col = 235.0 * fall
                            R[nr, c0] = v_col
                            G[nr, c0] = v_col
                            B[nr, c0] = min(255.0, v_col * 1.06)
                            lum[nr, c0] = fall
                            star_override[nr, c0] = 19 if dy < arm_y else 1
            else:
                v_col = 240.0 * br
                R[r0, c0] = v_col
                G[r0, c0] = v_col
                B[r0, c0] = min(255.0, v_col * 1.08)
                lum[r0, c0] = max(lum[r0, c0], br)
                star_override[r0, c0] = 16 if br > 0.65 else 1

    # --- 3. 3D-СЕРЦЕ ---
    heart_appear = smoothstep(20.5, 26.2, t) * (1.0 - smoothstep(41.5, 45.2, t))
    if heart_appear > 0.01:
        beat_phase = (t * 1.15) % 1.0
        pulse = (
            0.032 * math.exp(-((beat_phase - 0.15) ** 2) / 0.004)
            + 0.020 * math.exp(-((beat_phase - 0.32) ** 2) / 0.005)
        )
        h_scale = np.float32((0.04 + 0.50 * heart_appear) * (1.0 + pulse))

        sx_grid = U / h_scale
        sy_grid = (V - np.float32(0.02)) / h_scale
        rad2 = sx_grid * sx_grid + sy_grid * sy_grid

        glow = np.exp(-rad2 / 1.35) * np.float32(0.24 * heart_appear)
        glow_mask = pupil_mask & (glow > 0.04)
        if np.any(glow_mask):
            gl = glow[glow_mask]
            R[glow_mask] = np.minimum(255.0, R[glow_mask] + 215.0 * gl)
            G[glow_mask] = np.minimum(255.0, G[glow_mask] + 35.0 * gl)
            B[glow_mask] = np.minimum(255.0, B[glow_mask] + 95.0 * gl)

        cand = pupil_mask & (rad2 < 2.15)
        if np.any(cand):
            csx = sx_grid[cand]
            csy = sy_grid[cand]

            theta = (t - 20.5) * 0.90
            cos_t = np.float32(math.cos(theta))
            sin_t = np.float32(math.sin(theta))

            hx = csx[None, :] * cos_t + _Z_VALS * sin_t
            hy = csy[None, :]
            hz = -csx[None, :] * sin_t + _Z_VALS * cos_t

            hy2 = hy * hy
            a = hx * hx + np.float32(1.72) * hz * hz + hy2 - np.float32(1.0)
            f_val = a * a * a - (hx * hx + np.float32(0.1125) * hz * hz) * (hy2 * hy)

            inside = f_val <= 0.0
            any_hit = np.any(inside, axis=0)

            if np.any(any_hit):
                hit_idx = np.argmax(inside, axis=0)[any_hit]
                hsx = csx[any_hit]
                hsy = csy[any_hit]

                z_in = _Z_FLAT[hit_idx]
                z_out = _Z_FLAT[np.maximum(hit_idx - 1, 0)]

                for _ in range(4):
                    z_mid = np.float32(0.5) * (z_out + z_in)
                    mx = hsx * cos_t + z_mid * sin_t
                    my = hsy
                    mz = -hsx * sin_t + z_mid * cos_t
                    ma = mx * mx + np.float32(1.72) * mz * mz + my * my - np.float32(1.0)
                    mf = ma * ma * ma - (mx * mx + np.float32(0.1125) * mz * mz) * (my * my * my)
                    in_m = mf <= 0.0
                    z_in = np.where(in_m, z_mid, z_in)
                    z_out = np.where(in_m, z_out, z_mid)

                zh = np.float32(0.5) * (z_out + z_in)
                mx = hsx * cos_t + zh * sin_t
                my = hsy
                mz = -hsx * sin_t + zh * cos_t

                ma = mx * mx + np.float32(1.72) * mz * mz + my * my - np.float32(1.0)
                ma2 = ma * ma
                my2 = my * my
                my3 = my2 * my

                gx = np.float32(6.0) * ma2 * mx - np.float32(2.0) * mx * my3
                gy = np.float32(6.0) * ma2 * my - np.float32(3.0) * (mx * mx + np.float32(0.1125) * mz * mz) * my2
                gz = np.float32(10.32) * ma2 * mz - np.float32(0.225) * mz * my3

                nx = gx * cos_t - gz * sin_t
                ny = gy
                nz = gx * sin_t + gz * cos_t
                n_len = np.sqrt(nx * nx + ny * ny + nz * nz) + np.float32(1e-8)
                nx /= n_len
                ny /= n_len
                nz /= n_len

                diff = np.maximum(0.0, nx * np.float32(-0.22) + ny * np.float32(0.24) + nz * np.float32(0.945))
                spec = np.power(
                    np.maximum(0.0, nx * np.float32(-0.112) + ny * np.float32(0.122) + nz * np.float32(0.986)),
                    9.0,
                )

                shade = np.clip(0.14 + 0.72 * diff + 0.52 * spec, 0.08, 1.0) * np.float32(heart_appear)
                core_white = np.power(np.clip((shade - 0.52) / 0.48, 0.0, 1.0), 1.6)

                cand_rows, cand_cols = np.nonzero(cand)
                h_rows = cand_rows[any_hit]
                h_cols = cand_cols[any_hit]

                R[h_rows, h_cols] = np.clip(90.0 + 165.0 * np.power(shade, 0.65), 0.0, 255.0)
                G[h_rows, h_cols] = np.clip(14.0 + 85.0 * np.power(shade, 1.5) + 155.0 * core_white, 0.0, 255.0)
                B[h_rows, h_cols] = np.clip(40.0 + 115.0 * np.power(shade, 1.25) + 100.0 * core_white, 0.0, 255.0)
                lum[h_rows, h_cols] = np.clip(0.28 + 0.72 * shade, 0.0, 1.0)
                star_override[h_rows, h_cols] = 0

    if fade_in < 1.0:
        fi = np.float32(fade_in)
        R = 7.0 + (R - 7.0) * fi
        G = 9.0 + (G - 9.0) * fi
        B = 14.0 + (B - 14.0) * fi
        lum *= fi

    idx = np.clip((lum * (NUM_CHARS - 1)).astype(np.int32), 0, NUM_CHARS - 1)
    idx = np.where(star_override > 0, star_override, idx)

    diff_bgra[..., 0] = np.clip(B, 14.0, 255.0).astype(np.uint8) - np.uint8(14)
    diff_bgra[..., 1] = np.clip(G, 9.0, 255.0).astype(np.uint8) - np.uint8(9)
    diff_bgra[..., 2] = np.clip(R, 7.0, 255.0).astype(np.uint8) - np.uint8(7)

    m = _GLYPHS[idx].swapaxes(1, 2)
    np.multiply(m[..., None], diff_bgra[:, None, :, None, :], out=frame_5d)
    frame_5d += _BG_BGRA


class _AsciiCanvasWidget(QWidget):
    """Легкий віджет, який малює BGRA-буфер напряму через QImage.Format_RGB32."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        np = _ensure_tables()
        self.setFixedSize(WIDTH, HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        xs = (((np.arange(COLS, dtype=np.float32) + 0.5) / COLS) - 0.5) * 2.0 * ASPECT
        ys = (0.5 - ((np.arange(ROWS, dtype=np.float32) + 0.5) / ROWS)) * 2.0
        self._U, self._V = np.meshgrid(xs, ys)
        self._stars = make_stars()

        self._frame_buf = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
        self._frame_buf[:] = _BG_BGRA
        self._frame_5d = self._frame_buf.reshape(ROWS, CH, COLS, CW, 4)
        self._diff_bgra = np.zeros((ROWS, COLS, 4), dtype=np.uint8)
        self._image = QImage(
            self._frame_buf.data,
            WIDTH,
            HEIGHT,
            WIDTH * 4,
            QImage.Format.Format_RGB32,
        )
        self._start_t = time.perf_counter()
        self.render_frame(0.0)

    def render_frame(self, elapsed_s: Optional[float] = None) -> None:
        t = (time.perf_counter() - self._start_t) if elapsed_s is None else float(elapsed_s)
        render_into(t, self._U, self._V, self._stars, self._diff_bgra, self._frame_5d)
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)
        painter.end()


class BlaiteEasterEggDialog(QDialog):
    """Вікно пасхалки «Eye -> Heart», що відкривається при натисканні на нік Blaite."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Eye -> Heart")
        self.setStyleSheet(f"QDialog {{ background-color: {BG_HEX}; }}")
        self.setFixedSize(WIDTH, HEIGHT)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._canvas = _AsciiCanvasWidget(self)
        lay.addWidget(self._canvas)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(max(1, round(1000.0 / TARGET_FPS)))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def _on_tick(self) -> None:
        self._canvas.render_frame()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        try:
            from ui.dialogs import center_dialog_on_screen

            center_dialog_on_screen(self)
        except Exception:
            pass

    def done(self, result: int) -> None:
        self._timer.stop()
        super().done(result)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._timer.stop()
        super().closeEvent(event)
