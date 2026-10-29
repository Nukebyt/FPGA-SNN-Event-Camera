"""Streaming path tests: frame preparation and 30 fps resampling equal the training pipeline, the simulated board answers like the
golden model, and the C daemon (built with its fake engine in WSL) speaks the same protocol."""
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from model.snn_int import IntSNN
from sw.host import snnproto as P
from sw.host.snnboard_sim import SimBoard

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "rtl" / "weights" / "kth6"


def test_prepare_frame_equals_training_squash():
    import cv2
    from train.kth import to_gray_squash
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    assert np.array_equal(P.prepare_frame(img), to_gray_squash(img))
    assert P.prepare_frame(img).shape == (128, 128) and P.prepare_frame(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).dtype == np.uint8


@pytest.mark.parametrize("fps", [25.0, 24.0, 50.0, 29.97])
def test_resampler_matches_training_blend(fps):
    from train.kth import resample_linear
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, (60, 8, 8), dtype=np.uint8)
    rs = P.Resampler30(fps)
    out = [g for f in frames for g in rs.push(f)]
    ref = resample_linear(frames, fps, 30.0)
    n = min(len(out), len(ref))
    assert n >= len(ref) - 1
    assert np.array_equal(np.stack(out[:n]), ref[:n])


def test_result_payload_roundtrip():
    r = P.Result(seq=7, done_seq=3, pred=2, windows=1, wcount=5, spk1=11, spk2=12, events=13, step_cycles=205832, latency_cycles=222233,
                 status=1, cur=[-1, 2, -3, 4, -5, 6], score=[10, -20, 30, -40, 50, -60], win_score=[0, 1, 2, 3, 4, -5])
    assert P.parse_result(7, P.build_result_payload(r)) == r


def _pattern_frames(n):
    from model.pattern_source import pattern_frame
    return [pattern_frame(f) for f in range(n)]


@pytest.fixture(scope="module")
def sim():
    b = SimBoard(str(WEIGHTS))
    port = b.start()
    yield b, port
    b.stop()


def test_sim_board_matches_model_and_window_logic(sim):
    from model.emulator_int import EmuConfig, emulate
    from model.snn_input import bin_emulated
    board, port = sim
    c = P.Client("127.0.0.1", port)
    try:
        assert c.hello() == P.EXPECTED_BUILD
        frames = _pattern_frames(40)
        net = IntSNN.load(str(WEIGHTS))
        bins = bin_emulated(emulate(np.stack(frames), EmuConfig(theta=9, mode="step")), down=4)
        want = net.run(bins[:32])
        res = []
        for f in frames:
            c.send_frame(f)
            res.append(c.recv_result())
        assert [r.seq for r in res] == list(range(1, 41))
        assert res[31].windows == 1 and res[31].wcount == 0 and res[31].pred == int(np.argmax(want))
        assert res[31].win_score == [int(v) for v in want]
        assert res[39].windows == 1 and res[39].wcount == 8
        assert res[0].events == 0                                   # the init frame produces no events
        # a RESET restarts the window phase
        c.reset()
        c.send_frame(frames[0])
        r = c.recv_result()
        assert r.windows == 0 and r.wcount == 1 and r.done_seq == 1
    finally:
        c.close()


def test_sim_board_rejects_wrong_size(sim):
    import socket
    import struct
    _, port = sim
    s = socket.create_connection(("127.0.0.1", port), timeout=3)
    s.sendall(P.pack_msg(P.T_HELLO, 0, struct.pack("<HHHH", 640, 480, 0, 0)))
    hdr = s.recv(16)
    magic, ver, mtype, seq, length = P.HDR.unpack(hdr)
    assert mtype == P.T_ERROR
    s.close()


def _wsl_gcc_ok():
    if not shutil.which("wsl"):
        return False
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", "which gcc"], capture_output=True, text=True, timeout=60)
    return r.returncode == 0 and "gcc" in r.stdout


@pytest.mark.skipif(not _wsl_gcc_ok(), reason="needs WSL Ubuntu with gcc")
def test_c_daemon_with_fake_engine():
    src = "/mnt/" + ROOT.drive[0].lower() + str(ROOT.as_posix())[2:] + "/sw/snn_streamer.c"
    b = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", "gcc -std=gnu99 -O2 -Wall -Wextra -DSNN_SIM -o /tmp/snn_streamer_sim '%s'" % src],
                       capture_output=True, text=True, timeout=120)
    assert b.returncode == 0, b.stdout + b.stderr
    assert "warning" not in (b.stdout + b.stderr), b.stdout + b.stderr
    proc = subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", "/tmp/snn_streamer_sim --port 5191"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        c = None
        for _ in range(40):
            try:
                c = P.Client("127.0.0.1", 5191, timeout=3)
                break
            except OSError:
                time.sleep(0.25)
        assert c is not None, "daemon not reachable"
        assert c.hello() == P.EXPECTED_BUILD
        frames = [np.full((128, 128), v, np.uint8) for v in (10, 20, 30, 40)]
        got = []
        for f in frames:
            c.send_frame(f)
            got.append(c.recv_result())
        assert [r.seq for r in got] == [1, 2, 3, 4]
        assert [r.done_seq for r in got] == [1, 2, 3, 4]
        assert [r.spk1 for r in got] == [10, 20, 30, 40]            # fake engine: first pixel -> spk1, so the frame bytes arrived intact
        assert [r.pred for r in got] == [1, 2, 3, 4]
        assert got[0].cur[0] == 100 - 10 and got[0].cur[5] == 600 - 10
        assert got[0].latency_cycles == 222233
        # registers the client may not touch are ignored: IN_WR_SEQ must not move
        c.csr_write(6, 99)
        assert c.csr_read(6) == 4
        c.csr_write(9, 12)
        assert c.csr_read(9) == 12
        c.reset()
        c.send_frame(frames[0])
        r = c.recv_result()
        assert r.done_seq == 1 and r.seq == 1
        c.close()
        # a wrong-size HELLO is refused with an ERROR
        import socket
        import struct
        s = socket.create_connection(("127.0.0.1", 5191), timeout=3)
        s.sendall(P.pack_msg(P.T_HELLO, 0, struct.pack("<HHHH", 640, 480, 0, 0)))
        assert P.HDR.unpack(s.recv(16))[2] == P.T_ERROR
        s.close()
    finally:
        subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", "pkill snn_streamer_sim"], capture_output=True, timeout=30)
        proc.kill()
