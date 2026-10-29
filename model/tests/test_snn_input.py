import numpy as np

from model.emulator_int import EmuConfig, emulate
from model.snn_input import bin_emulated, bin_real
from model import synth_video as sv


def test_bin_real_places_events_and_bins_by_time():
    x = np.array([0, 3, 4, 127, 64])
    y = np.array([0, 3, 4, 127, 64])
    p = np.array([1, 0, 1, 1, 0])
    t = np.array([0, 1000, 33333, 70000, 99999])
    b = bin_real(x, y, p, t, bin_us=33333)
    assert b.shape == (4, 2, 32, 32)                       # bins 0,0,1,2,2 -> t=99999 is bin 3 (99999 // 33333 = 3)
    assert b[0, 1, 0, 0] == 1 and b[0, 0, 0, 0] == 1       # (3,3) shares the 4x4 cell (0,0) with (0,0), other polarity
    assert b[1, 1, 1, 1] == 1                              # (4,4) -> cell (1,1)
    assert b[2, 1, 31, 31] == 1 and b[3, 0, 16, 16] == 1
    assert b.sum() == 5


def test_bin_real_fixed_length_drops_late_events():
    b = bin_real(np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([0, 10 * 33333]), n_bins=3)
    assert b.shape[0] == 3 and b.sum() == 1


def test_bin_emulated_matches_events():
    out = emulate(sv.moving_bar(6), EmuConfig(theta=9, mode="reset"))
    b = bin_emulated(out)
    assert b.shape == (6, 2, 32, 32)
    assert b[0].sum() == 0 and b[1:].sum() > 0
    assert b.max() == 1
