"""Windowed bit-packed event-bin datasets for SNN training/eval.

A sample is uint8 [n_bins, 2, hw, hw] (0/1), hw = 32 or 64, stored bit-packed. Training takes a random
temporal window of T bins (plus a small random spatial shift); evaluation tiles the sample with
non-overlapping windows so every bin is used once.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

def pack(sample):
    return np.packbits(sample.reshape(len(sample), -1), axis=1)


def unpack(packed, hw=32):
    return np.unpackbits(packed, axis=1)[:, :2 * hw * hw].reshape(len(packed), 2, hw, hw)


# Horizontal mirror of a DVS128 Gesture window swaps the hand/arm side and reverses the circling sense:
# right wave <-> left wave, right arm cw <-> left arm ccw, right arm ccw <-> left arm cw (0-based class ids).
# Clapping, arm roll, air drums, air guitar and "other" map to themselves.
FLIP_MAP_DVS = {1: 2, 2: 1, 3: 6, 6: 3, 4: 5, 5: 4}


class WindowDataset(Dataset):
    """Training-only augmentations (all default off): event dropout `drop` (each active cell removed with this
    probability), mirror `flip_map` (50% of windows, x axis reversed and the label remapped), `speed` (this
    probability to take 2T bins and OR-merge neighbouring pairs, i.e. a gesture performed twice as fast)."""

    def __init__(self, samples, labels, T=48, train=True, shift=None, seed=0, drop=0.0, flip_map=None, speed=0.0):
        self.drop, self.flip_map, self.speed = drop, flip_map, speed
        self.hw = samples[0].shape[-1]
        shift = self.hw // 16 if shift is None else shift      # +-2 cells at 32x32, +-4 at 64x64 (same fraction of the image)
        self.packed = [pack(s) for s in samples]
        self.labels = np.asarray(labels)
        self.T, self.train, self.shift = T, train, shift
        self.rng = np.random.default_rng(seed)
        self.index = []     # eval: (sample, start)
        if not train:
            for i, p in enumerate(self.packed):
                for s in range(0, max(len(p), 1), T):
                    self.index.append((i, s))

    def __len__(self):
        return len(self.packed) if self.train else len(self.index)

    def _window(self, packed, start, n=None):
        n = self.T if n is None else n
        w = unpack(packed[start:start + n], self.hw)
        if len(w) < n:
            w = np.concatenate([w, np.zeros((n - len(w), 2, self.hw, self.hw), np.uint8)])
        return w

    def __getitem__(self, i):
        if self.train:
            p = self.packed[i]
            label = int(self.labels[i])
            fast = self.speed > 0 and self.rng.random() < self.speed
            n = 2 * self.T if fast else self.T
            start = int(self.rng.integers(0, max(len(p) - n, 0) + 1))
            w = self._window(p, start, n)
            if fast:
                w = w.reshape(self.T, 2, 2, self.hw, self.hw).max(axis=1)
            if self.flip_map is not None and self.rng.random() < 0.5:
                w = w[..., ::-1]
                label = self.flip_map.get(label, label)
            if self.drop > 0:
                w = w * (self.rng.random(w.shape) >= self.drop)
            if self.shift:
                dy, dx = self.rng.integers(-self.shift, self.shift + 1, 2)
                w = np.roll(w, (int(dy), int(dx)), axis=(2, 3))
            return torch.from_numpy(np.ascontiguousarray(w).astype(np.float32)), label, i
        si, start = self.index[i]
        return torch.from_numpy(self._window(self.packed[si], start).astype(np.float32)), int(self.labels[si]), si


def load_cache(path):
    z = np.load(path)
    lens = z["lens"]
    hw = int(z["hw"]) if "hw" in z else 32
    off = np.concatenate([[0], np.cumsum(lens)])
    bits = z["bits"]
    samples = [unpack(bits[off[i]:off[i + 1]], hw) for i in range(len(lens))]
    return samples, z["labels"]


def save_cache(path, samples, labels):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bits=np.concatenate([pack(s) for s in samples]),
                        lens=np.array([len(s) for s in samples]), labels=np.asarray(labels), hw=samples[0].shape[-1])
