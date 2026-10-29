"""Laptop side of the board streaming protocol (docs/snn_soc.md): framing, frame preparation, a small client.

Same framing as the night-vision project (16-byte header, little endian, one TCP connection) with its own magic so the two daemons
cannot be mixed up. The board runs sw/snn_streamer.c; sw/host/snnboard_sim.py is a drop-in simulated board.
"""
import socket
import struct
from dataclasses import dataclass

import cv2
import numpy as np

MAGIC = 0x454E4E53          # 'S','N','N','E'
VERSION = 1
HDR = struct.Struct("<IHHII")
T_HELLO, T_HELLO_ACK, T_FRAME_IN, T_RESULT, T_CSR_WRITE, T_CSR_READ, T_CSR_REPLY, T_ERROR, T_RESET = range(1, 10)
ERRORS = {1: "bad magic", 2: "bad version", 3: "bad size", 4: "short read", 5: "no memory", 6: "busy", 7: "FPGA timeout"}
FRAME = 128
NCLS = 6
CLASSES = ["walking", "jogging", "running", "boxing", "hand waving", "hand clapping"]
EXPECTED_BUILD = 0x534E0001
RESULT_WORDS = 28


class ProtocolError(RuntimeError):
    pass


@dataclass
class Result:
    seq: int
    done_seq: int
    pred: int               # class of the last FINISHED window (valid once windows >= 1)
    windows: int
    wcount: int             # steps done in the current window
    spk1: int
    spk2: int
    events: int
    step_cycles: int
    latency_cycles: int
    status: int
    cur: list               # class currents of this frame (summed into the window score)
    score: list             # running window score including this frame
    win_score: list         # scores of the last finished window

    @property
    def leader(self):
        """Class currently ahead in the running window (what the board would answer if the window ended now)."""
        return int(np.argmax(self.score))


def pack_msg(mtype, seq=0, payload=b""):
    return HDR.pack(MAGIC, VERSION, mtype, seq, len(payload)) + payload


def parse_result(seq, payload):
    if len(payload) != RESULT_WORDS * 4:
        raise ProtocolError("RESULT payload of %d bytes" % len(payload))
    u = struct.unpack("<%dI" % RESULT_WORDS, payload)
    s = struct.unpack("<%di" % RESULT_WORDS, payload)
    return Result(seq=seq, done_seq=u[0], pred=u[1], windows=u[2], wcount=u[3], spk1=u[4], spk2=u[5], events=u[6],
                  step_cycles=u[7], latency_cycles=u[8], status=u[9], cur=list(s[10:16]), score=list(s[16:22]), win_score=list(s[22:28]))


def build_result_payload(r):
    words = [r.done_seq, r.pred, r.windows, r.wcount, r.spk1, r.spk2, r.events, r.step_cycles, r.latency_cycles, r.status]
    ints = list(r.cur) + list(r.score) + list(r.win_score)
    return struct.pack("<10I", *words) + struct.pack("<18i", *ints)


def prepare_frame(img, fit="squash"):
    """Any image (BGR or grey) -> 128x128 uint8 luma exactly as the network was trained on (train/kth.py: grey, whole frame
    squashed with INTER_AREA). fit='crop' takes the centre square instead (for 16:9 phone video; NOT what the network saw)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if fit == "crop":
        h, w = g.shape
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        g = g[y0:y0 + s, x0:x0 + s]
    return cv2.resize(g, (FRAME, FRAME), interpolation=cv2.INTER_AREA)


class Resampler30:
    """Blend a stream of frames at src_fps into a stream at 30 fps (the rate and the 25 -> 30 blending of the training data).

    Output frame m sits at source position m * src_fps / 30 and is the blend of the two source frames around it, computed with
    the same expression as train.kth.resample_linear so that the result is identical (also on exact rounding ties)."""

    def __init__(self, src_fps, dst_fps=30.0):
        self.ratio = src_fps / dst_fps
        self.m = 0               # next output frame
        self.prev = None
        self.idx = -1

    def push(self, frame):
        """Feed one source frame, get the list of output frames that are now complete."""
        out = []
        self.idx += 1
        cur = frame.astype(np.float64)
        while True:
            pos = self.m * self.ratio
            i0 = int(pos)
            if i0 > self.idx or (i0 == self.idx - 0 and pos > self.idx):
                break
            if i0 == self.idx:                       # exactly on this source frame (a = 0)
                out.append(frame.copy())
            elif i0 == self.idx - 1:
                a = pos - i0
                out.append(np.clip(np.rint(self.prev * (1 - a) + cur * a), 0, 255).astype(np.uint8))
            else:
                break
            self.m += 1
        self.prev = cur
        return out


class Client:
    def __init__(self, host, port=5002, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seq = 0

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ProtocolError("connection closed by the board")
            buf += chunk
        return buf

    def recv(self):
        magic, ver, mtype, seq, length = HDR.unpack(self._read(HDR.size))
        if magic != MAGIC or ver != VERSION:
            raise ProtocolError("bad header magic %08x version %d" % (magic, ver))
        payload = self._read(length) if length else b""
        if mtype == T_ERROR:
            code = struct.unpack("<I", payload[:4])[0]
            raise ProtocolError("board error %d (%s)" % (code, ERRORS.get(code, "?")))
        return mtype, seq, payload

    def hello(self):
        self.sock.sendall(pack_msg(T_HELLO, 0, struct.pack("<HHHH", FRAME, FRAME, 0, 0)))
        mtype, _, payload = self.recv()
        if mtype != T_HELLO_ACK:
            raise ProtocolError("expected HELLO_ACK, got type %d" % mtype)
        status, build, w, h = struct.unpack("<IIHH", payload)
        if status != 0:
            raise ProtocolError("HELLO status %d" % status)
        if build != EXPECTED_BUILD:
            raise ProtocolError("board build id 0x%08X, this software expects 0x%08X (bitstream and software must match)" % (build, EXPECTED_BUILD))
        return build

    def reset(self):
        """Restart the engine (new clip): all state cleared, the decision window starts again at step 0."""
        self.sock.sendall(pack_msg(T_RESET, 0, b""))
        mtype, _, payload = self.recv()
        if mtype != T_HELLO_ACK:
            raise ProtocolError("expected HELLO_ACK after RESET, got type %d" % mtype)
        self.seq = 0

    def send_frame(self, frame):
        assert frame.shape == (FRAME, FRAME) and frame.dtype == np.uint8
        self.seq += 1
        self.sock.sendall(pack_msg(T_FRAME_IN, self.seq, frame.tobytes()))
        return self.seq

    def recv_result(self):
        mtype, seq, payload = self.recv()
        if mtype != T_RESULT:
            raise ProtocolError("expected RESULT, got type %d" % mtype)
        return parse_result(seq, payload)

    def csr_write(self, addr_word, value):
        self.sock.sendall(pack_msg(T_CSR_WRITE, 0, struct.pack("<II", addr_word * 4, value & 0xFFFFFFFF)))

    def csr_read(self, addr_word):
        self.sock.sendall(pack_msg(T_CSR_READ, 0, struct.pack("<I", addr_word * 4)))
        mtype, _, payload = self.recv()
        if mtype != T_CSR_REPLY:
            raise ProtocolError("expected CSR_REPLY, got %d" % mtype)
        return struct.unpack("<II", payload)[1]
