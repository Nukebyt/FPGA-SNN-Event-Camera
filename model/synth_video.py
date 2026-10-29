"""Deterministic synthetic 8-bit grayscale clips for tests and bring-up (no dataset needed)."""
import numpy as np


def moving_bar(n_frames=24, size=128, bar_w=12, bg=40, fg=200, step=4):
    """Vertical bright bar sweeping left to right."""
    frames = np.full((n_frames, size, size), bg, dtype=np.uint8)
    for k in range(n_frames):
        x0 = (k * step) % (size - bar_w)
        frames[k, :, x0:x0 + bar_w] = fg
    return frames


def moving_blob(n_frames=24, size=128, radius=14, bg=30, fg=220):
    """Soft disc moving on a circle."""
    yy, xx = np.mgrid[0:size, 0:size]
    frames = np.zeros((n_frames, size, size), dtype=np.uint8)
    for k in range(n_frames):
        a = 2 * np.pi * k / n_frames
        cy = size / 2 + 0.3 * size * np.sin(a)
        cx = size / 2 + 0.3 * size * np.cos(a)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        frames[k] = np.clip(bg + (fg - bg) * np.exp(-r2 / (2 * (radius / 2) ** 2)), 0, 255).astype(np.uint8)
    return frames


def flash(n_frames=8, size=128, lo=20, hi=230, at=3):
    """Whole-frame step change: stress test for event-rate bursts."""
    frames = np.full((n_frames, size, size), lo, dtype=np.uint8)
    frames[at:] = hi
    return frames


def noise(n_frames=6, size=128, seed=1):
    """Independent uniform noise every frame: nearly every pixel fires (worst-case event rate)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n_frames, size, size), dtype=np.uint8)


def static(n_frames=6, size=128, level=100):
    return np.full((n_frames, size, size), level, dtype=np.uint8)
