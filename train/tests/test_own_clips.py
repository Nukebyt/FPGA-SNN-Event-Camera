import cv2
import numpy as np
import pytest

from train.eval_int import load_own_clips, trim_inactive


def _write_clip(path, still_s=1.0, move_s=3.0, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240), True)
    yy, xx = np.mgrid[0:240, 0:320]
    for k in range(int(fps * (2 * still_s + move_s))):
        t = k / fps
        img = np.full((240, 320), 140.0)
        if still_s < t < still_s + move_s:
            cx = 160 + 50 * np.sin(2 * np.pi * 2 * t)
            img = np.where(((xx - cx) ** 2 + (yy - 120) ** 2) < 35 ** 2, 230.0, img)
        vw.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()


def test_labels_come_from_folder_prefix_and_clips_are_trimmed(tmp_path):
    for name in ("05_left_arm_cw", "00_hand_clapping"):
        (tmp_path / name).mkdir()
        _write_clip(tmp_path / name / "A_p1_01.avi")
    X, Y, names = load_own_clips(tmp_path)
    assert sorted(Y.tolist()) == [0, 5] and names == ["00_hand_clapping", "05_left_arm_cw"]
    assert Y[names.index("00_hand_clapping")] == 0 and Y[1 - names.index("00_hand_clapping")] == 5
    full = load_own_clips(tmp_path, trim=False)[0][0]
    assert len(full) == 150 and 80 <= len(X[0]) <= 100          # 5 s clip, about 3 s of motion kept


def test_folder_without_class_number_is_rejected(tmp_path):
    (tmp_path / "left_arm_cw").mkdir()
    with pytest.raises(ValueError):
        load_own_clips(tmp_path)


def test_trim_keeps_short_clips_untouched():
    b = np.zeros((20, 2, 32, 32), np.uint8)
    assert len(trim_inactive(b)) == 20
