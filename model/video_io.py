"""Video file -> 8-bit grayscale square frames. Shared by sw/frame_feeder.py and the dataset code.

Square frames are made by a centre crop (not a squash) so motion keeps its aspect ratio, then
cv2.INTER_AREA down to size x size.
"""
import cv2
import numpy as np


def to_gray_square(frame_bgr, size=128):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    h, w = g.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    g = g[y0:y0 + s, x0:x0 + s]
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def read_video(path, size=128, target_fps=None, max_frames=None):
    """Returns (frames uint8 [N, size, size], fps_out). target_fps resamples by dropping frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_out = src_fps if not target_fps else min(target_fps, src_fps)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        # keep frame when the output clock (fps_out) has advanced past the next slot
        if i * fps_out / src_fps >= len(frames):
            frames.append(to_gray_square(f, size))
        i += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames), fps_out
