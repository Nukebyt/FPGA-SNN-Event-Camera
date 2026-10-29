"""Recording training clips from the live demo: the server's save / list / trash functions and the training loader."""
import json
import os
import struct

import cv2
import numpy as np
import pytest

from scripts import snn_web as W
from train.kth import read_all_frames, resample_linear
from train.recordings import REC_ACTIONS, load_recordings, resample_times


def _frames(n, w=320, h=240):
    """A bright disc moving left to right on grey, so every frame differs."""
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for k in range(n):
        img = np.full((h, w), 120, np.uint8)
        img[(xx - (40 + 6 * k)) ** 2 + (yy - h // 2) ** 2 < 30 ** 2] = 230
        out.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return out


def _body(frames, stamps, action="handwaving", subject="p01", **extra):
    jpgs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for f in frames]
    h = dict(action=action, subject=subject, note="test", sizes=[len(j) for j in jpgs], stamps_ms=list(stamps), missed=0,
             timing="test", camera={"label": "fake"}, user_agent="pytest")
    h.update(extra)
    hb = json.dumps(h).encode()
    return struct.pack("<I", len(hb)) + hb + b"".join(jpgs)


def test_save_list_load_and_trash(tmp_path):
    n = 60
    stamps = 1000.0 + np.arange(n) * (1000 / 29.7)                 # a camera a little slower than 30 fps
    stamps[30:] += 40.0                                               # and one late frame
    meta = W.save_recording(_body(_frames(n), stamps), root=str(tmp_path))
    avi = meta["path"]
    assert os.path.basename(os.path.dirname(avi)) == "handwaving" and os.path.basename(avi).startswith("p01_handwaving_")
    side = json.loads(open(avi[:-4] + ".json", encoding="utf-8").read())
    assert side["label"] == 4 and side["subject"] == "p01" and side["frames"] == n and side["width"] == 320
    assert side["stamps_ms"][0] == 0 and abs(side["stamps_ms"][-1] - (stamps[-1] - stamps[0])) < 0.01
    cap = cv2.VideoCapture(avi)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
    assert abs(cap.get(cv2.CAP_PROP_FPS) - side["fps"]) < 0.01      # the measured rate, not rounded to 30
    cap.release()

    # listed, playable from the demo, not mixed into another action
    recs = W.list_recordings(root=str(tmp_path))
    assert [r["path"] for r in recs] == [avi]
    clips = W.clip_list(rec_root=str(tmp_path))
    mine = [c for c in clips if c["group"] == "your recordings"]
    assert [c["path"] for c in mine] == [avi] and mine[0]["truth"] == "hand waving"

    # training loader: exact 30 fps from the capture times, label from the folder
    (frames, m), = list(load_recordings(tmp_path))
    dur = (stamps[-1] - stamps[0]) / 1000
    assert len(frames) == int(np.floor(dur * 30)) + 1 and frames.shape[1:] == (128, 128) and m["label"] == 4
    assert list(load_recordings(tmp_path, actions=["walking"])) == []
    assert list(load_recordings(tmp_path, subjects={"someone-else"})) == []

    # a second save in the same second gets its own name
    meta2 = W.save_recording(_body(_frames(n), stamps), root=str(tmp_path))
    assert meta2["path"] != avi and len(W.list_recordings(root=str(tmp_path))) == 2

    # trash: moved, not erased, and gone from the lists
    W.trash_recording(avi, root=str(tmp_path))
    assert not os.path.exists(avi) and os.path.exists(tmp_path / "_trash" / "handwaving" / os.path.basename(avi))
    assert [r["path"] for r in W.list_recordings(root=str(tmp_path))] == [meta2["path"]]
    assert len(list(load_recordings(tmp_path))) == 1
    with pytest.raises(ValueError):
        W.trash_recording(str(tmp_path / "_trash" / "handwaving" / os.path.basename(avi)), root=str(tmp_path))
    with pytest.raises(ValueError):
        W.trash_recording(os.path.abspath(__file__), root=str(tmp_path))


def test_none_class_and_warnings(tmp_path):
    n = 40
    stamps = np.arange(n) * 66.7                                       # 15 frames/s
    stamps[20:] += 500                                                 # half a second missing
    meta = W.save_recording(_body(_frames(n), stamps, action="none", missed=3), root=str(tmp_path))
    assert meta["label"] is None and len(meta["warnings"]) == 3
    (_, m), = list(load_recordings(tmp_path, actions=REC_ACTIONS))
    assert m["label"] == REC_ACTIONS.index("none")
    assert list(load_recordings(tmp_path)) == []                      # default: the six KTH actions only


@pytest.mark.parametrize("change, msg", [
    (dict(action="dancing"), "unknown action"),
    (dict(subject="../evil"), "subject"),
    (dict(subject="a_b"), "subject"),
    (dict(subject=""), "subject"),
    (dict(n=5), "at least"),
    (dict(stamps=lambda s: s[::-1]), "in order"),
    (dict(stamps=lambda s: s * 100), "duration"),
    (dict(cut=10), "add up"),
    (dict(sizes_extra=True), "same length"),
])
def test_bad_recordings_are_refused(tmp_path, change, msg):
    n = change.get("n", 30)
    stamps = np.arange(n) * 33.3
    if "stamps" in change:
        stamps = change["stamps"](stamps)
    kw = {k: change[k] for k in ("action", "subject") if k in change}
    body = _body(_frames(n), stamps, **kw)
    if "cut" in change:
        body = body[:-change["cut"]]
    if change.get("sizes_extra"):
        hl = struct.unpack("<I", body[:4])[0]
        h = json.loads(body[4:4 + hl])
        h["stamps_ms"] = h["stamps_ms"][:-1]
        hb = json.dumps(h).encode()
        body = struct.pack("<I", len(hb)) + hb + body[4 + hl:]
    with pytest.raises(ValueError, match=msg):
        W.save_recording(body, root=str(tmp_path))
    assert not any(f.endswith(".avi") for _, _, fs in os.walk(tmp_path) for f in fs)


def test_resample_times_equals_kth_blending_for_an_even_rate():
    rng = np.random.default_rng(1)
    fr = rng.integers(0, 256, (51, 8, 8)).astype(np.uint8)
    a = resample_times(fr, np.arange(51) * 40.0).astype(int)           # 25 fps, like KTH
    b = resample_linear(fr, 25.0, 30.0).astype(int)
    assert a.shape == b.shape
    d = np.abs(a - b)
    assert d.max() <= 1                                                # same blend; exact halves may round the other way
    assert np.array_equal(a[::2], b[::2])                              # even frames: no half-way blend, identical


def test_loader_matches_the_kth_preparation(tmp_path):
    n = 30
    W.save_recording(_body(_frames(n), np.arange(n) * 1000 / 30), root=str(tmp_path))
    (frames, m), = list(load_recordings(tmp_path))
    raw = read_all_frames(m["path"], fit="squash")                    # the KTH network's preparation
    assert len(frames) == n and np.abs(frames.astype(int) - raw.astype(int)).max() <= 1
