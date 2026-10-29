"""Small synthetic gesture set rendered as video and pushed through the integer emulator.

6 classes: bar sweeping right / left / down / up, blob orbiting clockwise / counter-clockwise.
It exercises the whole Track B chain (video -> emulator -> bins -> SNN) without any dataset,
and stands in for the "4-6 own-recorded gesture classes" until real clips exist.
"""
import numpy as np

from model.emulator_int import EmuConfig, emulate
from model.snn_input import bin_emulated

CLASSES = ["bar_right", "bar_left", "bar_down", "bar_up", "orbit_cw", "orbit_ccw"]


def render(cls, rng, n_frames=48, size=128):
    bg = int(rng.integers(20, 90))
    fg = int(rng.integers(150, 250))
    yy, xx = np.mgrid[0:size, 0:size]
    frames = np.empty((n_frames, size, size), np.uint8)
    if cls < 4:
        w = int(rng.integers(8, 20))
        speed = rng.uniform(1.5, 3.2) * size / n_frames
        start = rng.uniform(-0.2, 0.25) * size
        for k in range(n_frames):
            pos = start + speed * k
            if cls in (1, 3):
                pos = size - pos
            c = xx if cls < 2 else yy
            frames[k] = np.where((c >= pos) & (c < pos + w), fg, bg)
    else:
        r = rng.uniform(0.2, 0.32) * size
        rad = rng.uniform(7, 12)
        phase = rng.uniform(0, 2 * np.pi)
        turns = rng.uniform(0.8, 1.5)
        sgn = 1 if cls == 4 else -1
        for k in range(n_frames):
            a = phase + sgn * 2 * np.pi * turns * k / n_frames
            cy, cx = size / 2 + r * np.sin(a), size / 2 + r * np.cos(a)
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            frames[k] = np.clip(bg + (fg - bg) * np.exp(-d2 / (2 * (rad / 2) ** 2)), 0, 255)
    return frames


def make_dataset(n_per_class=40, seed=0, n_frames=48, cfg=EmuConfig(theta=9, mode="step"), down=4):
    """Returns (bins uint8 [N, T, 2, 128/down, 128/down], labels int [N])."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for c in range(len(CLASSES)):
        for _ in range(n_per_class):
            X.append(bin_emulated(emulate(render(c, rng, n_frames), cfg), down=down))
            Y.append(c)
    return np.stack(X), np.array(Y)
