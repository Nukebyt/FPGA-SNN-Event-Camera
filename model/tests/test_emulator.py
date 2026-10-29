import numpy as np

from model import synth_video as sv
from model.emulator_int import EmuConfig, IntEmulator, emulate
from model.emulator_ref import emulate_float
from model.event_format import frame_ts, pack, unpack
from model.log_lut import LUT, LN_PER_LSB, theta_from_ln


def test_lut_shape_and_monotonic():
    assert LUT[0] == 0 and LUT[255] == 255
    assert np.all(np.diff(LUT.astype(int)) >= 0)
    assert theta_from_ln(0.2) == 9


def test_event_roundtrip():
    w = pack([1, 0], [127, 3], [5, 126], [0x1FFFF, 7])
    u = unpack(w)
    assert list(u["pol"]) == [1, 0] and list(u["y"]) == [127, 3]
    assert list(u["x"]) == [5, 126] and list(u["ts"]) == [0x1FFFF, 7]


def test_first_frame_and_static_make_no_events():
    out = emulate(sv.static(), EmuConfig())
    assert all(len(e) == 0 for e in out)


def test_flash_polarity_and_full_frame_burst():
    cfg = EmuConfig(theta=9, mode="reset")
    out = emulate(sv.flash(at=3), cfg)
    assert [len(e) for e in out] == [0, 0, 0, 128 * 128, 0, 0, 0, 0]
    assert np.all(unpack(out[3])["pol"] == 1)


def test_step_mode_spreads_large_change_over_frames():
    cfg = EmuConfig(theta=9, mode="step")
    n = [len(e) for e in emulate(sv.flash(n_frames=30, at=3), cfg)]
    d = int(LUT[230]) - int(LUT[20])
    assert sum(1 for v in n if v) == d // 9          # one event per theta of contrast, per pixel
    assert n[3] == 128 * 128


def test_off_events_and_timestamps():
    cfg = EmuConfig(theta=9, mode="reset", period_ticks=333)
    fr = sv.flash(n_frames=6, at=2)[::-1].copy()    # bright then dark
    out = emulate(fr, cfg)
    ev = out[4]                                      # reversed clip: bright for frames 0-3, dark from 4
    assert len(ev) == 128 * 128 and np.all(unpack(ev)["pol"] == 0)
    assert np.all(unpack(ev)["ts"] == frame_ts(4, 333))


def test_timestamp_wraps_17_bits():
    assert frame_ts(1000, 333) == (333000 & 0x1FFFF)


def test_raster_order_and_coordinates():
    fr = np.full((3, 128, 128), 50, np.uint8)
    fr[1, 5, 100] = 250
    fr[2, 5, 100] = 250
    fr[1, 70, 3] = 250
    ev = emulate(fr, EmuConfig(mode="reset"))[1]
    u = unpack(ev)
    assert list(zip(u["y"], u["x"])) == [(5, 100), (70, 3)]


def test_int_matches_float_statistics_on_moving_bar():
    fr = sv.moving_bar(n_frames=24)
    ei = emulate(fr, EmuConfig(theta=theta_from_ln(0.2), mode="step"))
    n_int = sum(len(e) for e in ei)
    ef = emulate_float(fr, theta_ln=0.2)
    n_flt = len(ef["t"])
    assert 0.3 < n_int / n_flt < 1.5, (n_int, n_flt)   # 1 event/pixel/frame cap -> fewer, not wildly different
    pol_int = np.mean(np.concatenate([unpack(e)["pol"] for e in ei if len(e)]))
    assert abs(pol_int - np.mean(ef["pol"])) < 0.1
