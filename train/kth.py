"""KTH action videos (hand clapping, hand waving, ...) as a real-video test for the event pipeline.

Source: https://www.csc.kth.se/cvap/actions/  (per-action zips + 00sequences.txt), 25 subjects, static camera,
160x120, 25 fps, four conditions d1 (homogeneous background), d2 (+ scale change), d3 (+ different clothes),
d4 (+ lighting changes). Publicly available for non-commercial use; cite Schuldt, Laptev, Caputo, ICPR 2004.

Each video holds about four repetitions; 00sequences.txt gives their 1-based inclusive frame ranges and the
subject split of the original paper (train 11-18, validation 19-21 23-25 01 04, test 22 02 03 05-10).
Frames are centre-cropped to 120x120, resized to 128x128 gray and resampled from 25 to 30 fps by linear blending
(the whole system is built on 30 fps frames; duplicating frames would create zero-change frames and false pauses).
"""
import re
from pathlib import Path

import cv2
import numpy as np

from model.video_io import to_gray_square

SPLITS = {"train": {11, 12, 13, 14, 15, 16, 17, 18}, "val": {19, 20, 21, 23, 24, 25, 1, 4},
          "test": {22, 2, 3, 5, 6, 7, 8, 9, 10}}
_LINE = re.compile(r"^(person(\d+)_(\w+?)_(d\d))\s+frames\s+(.*)$")


def split_of(person):
    for name, ids in SPLITS.items():
        if person in ids:
            return name
    raise ValueError("unknown KTH subject %r" % person)


def parse_sequences(path):
    """-> {video key e.g. 'person01_handclapping_d1': (person, action, cond, [(start, end), ...])} (1-based, inclusive)."""
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        key, person, action, cond, rest = m.groups()
        ranges = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", rest)]
        out[key] = (int(person), action, cond, ranges)
    return out


def resample_linear(frames, src_fps=25.0, dst_fps=30.0):
    """Blend neighbouring frames so that the output runs at dst_fps (same duration)."""
    n = len(frames)
    m = int(np.floor((n - 1) * dst_fps / src_fps)) + 1
    pos = np.arange(m) * (src_fps / dst_fps)
    i0 = np.minimum(pos.astype(int), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    a = (pos - i0)[:, None, None]
    return np.clip(np.rint(frames[i0] * (1 - a) + frames[i1] * a), 0, 255).astype(np.uint8)


# The six KTH actions in the paper's order; the index is the class label of the KTH network (train/prep_kth.py).
ACTIONS = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]


def to_gray_squash(frame_bgr, size=128):
    """Whole 160x120 frame -> size x size (x scaled 0.8, y 1.07). Keeps walkers who cross the sides of the frame."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def read_all_frames(path, size=128, fit="crop"):
    """fit: 'crop' = centre 120x120 (used by the domain-gap test), 'squash' = whole frame (used by the KTH network)."""
    to_sq = {"crop": to_gray_square, "squash": to_gray_squash}[fit]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("cannot open %s" % path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(to_sq(f, size))
    cap.release()
    return np.stack(fr)


def load_kth(root, action, split=None, conds=("d1", "d2", "d3", "d4"), seq_file=None, max_videos=None, fit="crop"):
    """Yield (frames uint8 [n, 128, 128] at 30 fps, meta dict) for every repetition of `action` in the chosen split."""
    root = Path(root)
    seqs = parse_sequences(seq_file or root / "00sequences.txt")
    videos = {}
    for f in root.rglob("*.avi"):
        m = re.match(r"(person\d+_\w+?_d\d)", f.name)
        if m:
            videos[m.group(1)] = f
    n = 0
    for key in sorted(seqs):
        person, act, cond, ranges = seqs[key]
        if act != action or cond not in conds or (split and split_of(person) != split) or key not in videos:
            continue
        frames = read_all_frames(videos[key], fit=fit)
        for k, (a, b) in enumerate(ranges):
            seg = frames[a - 1:b]
            if len(seg) < 25:
                continue
            yield resample_linear(seg), {"video": key, "person": person, "action": act, "cond": cond, "rep": k,
                                          "split": split_of(person)}
        n += 1
        if max_videos and n >= max_videos:
            return
