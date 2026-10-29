import cv2
import numpy as np

from train.kth import SPLITS, load_kth, parse_sequences, resample_linear, split_of

SEQ = """
This is a header.

person01_boxing_d1\t\tframes\t1-95, 96-185
person11_handclapping_d1\tframes\t1-60, 61-110, 140-200
person22_handclapping_d2\tframes\t1-70
"""


def test_parse_and_split(tmp_path):
    p = tmp_path / "00sequences.txt"
    p.write_text(SEQ)
    s = parse_sequences(p)
    assert s["person11_handclapping_d1"] == (11, "handclapping", "d1", [(1, 60), (61, 110), (140, 200)])
    assert s["person01_boxing_d1"][3] == [(1, 95), (96, 185)]
    assert split_of(11) == "train" and split_of(1) == "val" and split_of(22) == "test"
    assert len(SPLITS["train"] | SPLITS["val"] | SPLITS["test"]) == 25


def test_resample_keeps_duration_and_blends():
    f = np.stack([np.full((4, 4), v, np.uint8) for v in (0, 100, 200, 250)])
    r = resample_linear(f, 25.0, 30.0)
    assert len(r) == int(np.floor(3 * 30 / 25)) + 1 == 4
    assert r[0, 0, 0] == 0 and 0 < r[1, 0, 0] < 100          # position 25/30 lies between frames 0 and 1
    assert r[1, 0, 0] == round(100 * 25 / 30)


def test_load_kth_slices_ranges_and_filters(tmp_path):
    (tmp_path / "00sequences.txt").write_text(SEQ)
    for name in ("person11_handclapping_d1_uncomp.avi", "person22_handclapping_d2_uncomp.avi"):
        vw = cv2.VideoWriter(str(tmp_path / name), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120), True)
        for k in range(210):
            vw.write(np.full((120, 160, 3), (k * 3) % 250, np.uint8))
        vw.release()
    out = list(load_kth(tmp_path, "handclapping", split="train"))
    assert [m["rep"] for _, m in out] == [0, 1, 2] and all(m["person"] == 11 for _, m in out)
    fr0, _ = out[0]
    assert fr0.shape[1:] == (128, 128) and len(fr0) == int(np.floor(59 * 30 / 25)) + 1       # 60 source frames -> 30 fps
    assert [m["person"] for _, m in load_kth(tmp_path, "handclapping", split="test")] == [22]
    assert list(load_kth(tmp_path, "handwaving")) == []
