"""Webcam clips recorded with the live demo ("Record training clips" in scripts/snn_web.py) as training data.

    data/recordings/<action>/<subject>_<action>_<YYYYmmdd-HHMMSS>.avi    the camera frames, 4:3, at the rate they really came
    data/recordings/<action>/<subject>_<action>_<YYYYmmdd-HHMMSS>.json   label, subject, capture time of every frame, warnings

<action> is a KTH action name (train.kth.ACTIONS) or "none" (no action: background, standing still; for a reject class).
Deleted recordings are in data/recordings/_trash/ and are skipped.

load_recordings() gives the frames prepared exactly like the KTH network's training data (grey, whole picture squashed to 128x128,
30 fps). The 30 fps frames are blended from the two recorded frames around each 1/30 s instant, using the real capture times, the
same blending as train.kth.resample_linear does for KTH's 25 fps (for an even rate the two agree to within 1 grey level, where a
blend is an exact half and float rounding breaks the tie differently), so a camera that drops a frame or runs at 29.7 fps still
gives correct timing. A whole recording is one sample; KTH's samples are single repetitions, so trim the start and end (walking into
position after the countdown) before training if needed, e.g. with train.eval_int.trim_inactive on the bins.

Split by subject, as KTH is split by person, so that a person's clips are never in both training and test data.
"""
import json
from pathlib import Path

import numpy as np

from train.kth import ACTIONS, read_all_frames

REC_ACTIONS = ACTIONS + ["none"]


def resample_times(frames, t_ms, dst_fps=30.0):
    """frames [n, h, w] captured at times t_ms (ascending) -> frames at exactly dst_fps from the first capture to the last."""
    t = (np.asarray(t_ms, np.float64) - t_ms[0]) / 1000.0
    n = len(frames)
    if n != len(t):
        raise ValueError("%d frames but %d capture times" % (n, len(t)))
    m = int(np.floor(t[-1] * dst_fps + 1e-9)) + 1
    tq = np.arange(m) / dst_fps
    i1 = np.clip(np.searchsorted(t, tq, side="right"), 1, n - 1)     # t[i1 - 1] <= tq < t[i1]
    i0 = i1 - 1
    span = np.maximum(t[i1] - t[i0], 1e-9)
    a = np.clip((tq - t[i0]) / span, 0.0, 1.0)[:, None, None]
    return np.clip(np.rint(frames[i0] * (1 - a) + frames[i1] * a), 0, 255).astype(np.uint8)


def load_recordings(root="data/recordings", actions=ACTIONS, size=128, fit="squash", subjects=None):
    """Yield (frames uint8 [n, size, size] at 30 fps, meta) for every recording whose action is in `actions`.

    meta: the sidecar (file, action, subject, fps, frames, warnings, ...) plus "path" and "label" = index of the action in
    `actions` (pass actions=REC_ACTIONS to include "none" as the last class). subjects: optional set of subject names."""
    root = Path(root)
    for action in actions:
        d = root / action
        if not d.is_dir():
            continue
        for js in sorted(d.glob("*.json")):
            if js.name.startswith("_"):
                continue
            avi = js.with_suffix(".avi")
            if not avi.exists():
                continue
            meta = json.loads(js.read_text(encoding="utf-8"))
            if subjects is not None and meta.get("subject") not in subjects:
                continue
            frames = read_all_frames(avi, size=size, fit=fit)
            stamps = meta.get("stamps_ms")
            if not stamps or len(stamps) != len(frames):
                fps = float(meta.get("fps") or 30.0)              # sidecar without times: assume an even frame rate
                stamps = np.arange(len(frames)) * 1000.0 / fps
            meta = dict(meta, path=str(avi), label=actions.index(action))
            yield resample_times(frames, stamps), meta
