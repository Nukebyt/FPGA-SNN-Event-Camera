"""A simulated board: speaks the streaming protocol (docs/snn_soc.md) and runs the bit-accurate Python models instead of the FPGA.

    from sw.host.snnboard_sim import SimBoard
    b = SimBoard("rtl/weights/kth6"); port = b.start(); ...; b.stop()

It exists for tests and for trying the viewer without hardware. It is the golden reference of what the real board must answer:
emulator (model/emulator_int.py, theta from the THETA CSR) -> 2x32x32 bins -> IntSNN step -> window of T steps, argmax with ties to
the lowest class, all state cleared at the start of every window, exactly as rtl/snn_engine.v + snn_ctrl.v do. The cycle counters
are the constants measured in simulation (205,832 clocks per step, 222,233 per frame).
"""
import socket
import struct
import threading

import numpy as np

from model.emulator_int import EmuConfig, IntEmulator
from model.snn_input import bin_emulated
from model.snn_int import IntSNN
from sw.host import snnproto as P

STEP_CYCLES = 205832
LATENCY_CYCLES = 222233


class SimBoard:
    def __init__(self, weights="rtl/weights/kth6", window=32):
        self.net = IntSNN.load(weights)
        self.window = window
        self.srv = None
        self.thread = None
        self.theta = 9
        self._reset_state()

    # ---- model state (one engine) ----
    def _reset_state(self):
        self.emu = IntEmulator(EmuConfig(theta=self.theta, mode="step"))
        self.net.reset()
        self.score = np.zeros(self.net.cfg.classes, np.int64)
        self.wcount = 0
        self.windows = 0
        self.pred = 0
        self.win_score = np.zeros(self.net.cfg.classes, np.int64)
        self.done_seq = 0

    def process(self, frame, seq):
        words = self.emu.step(frame)
        bins = bin_emulated([words], size=128, down=4)[0]
        s1, s2, cur = self.net.step(bins)
        self.score += cur
        self.done_seq += 1
        self.wcount += 1
        res_score = self.score.copy()
        if self.wcount == self.window:
            self.pred = int(np.argmax(self.score))
            self.win_score = self.score.copy()
            self.windows += 1
            self.wcount = 0
            self.score[:] = 0
            self.net.reset()
        return P.Result(seq=seq, done_seq=self.done_seq, pred=self.pred, windows=self.windows, wcount=self.wcount,
                        spk1=int(s1.sum()), spk2=int(s2.sum()), events=int(len(words)), step_cycles=STEP_CYCLES,
                        latency_cycles=LATENCY_CYCLES, status=1, cur=[int(v) for v in cur], score=[int(v) for v in res_score],
                        win_score=[int(v) for v in self.win_score])

    # ---- server ----
    def start(self, host="127.0.0.1", port=0):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(2)
        self.srv.settimeout(0.2)
        self._stop = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self.srv.getsockname()[1]

    def stop(self):
        self._stop = True
        if self.thread:
            self.thread.join(timeout=3)
        if self.srv:
            self.srv.close()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(5.0)
            try:
                self._handle(conn)
            except (OSError, P.ProtocolError):
                pass
            finally:
                conn.close()

    @staticmethod
    def _read(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise P.ProtocolError("closed")
            buf += chunk
        return buf

    def _handle(self, conn):
        def recv():
            magic, ver, mtype, seq, length = P.HDR.unpack(self._read(conn, P.HDR.size))
            payload = self._read(conn, length) if length else b""
            return magic, ver, mtype, seq, payload

        def error(code):
            conn.sendall(P.pack_msg(P.T_ERROR, 0, struct.pack("<I", code)))

        magic, ver, mtype, seq, payload = recv()
        if magic != P.MAGIC:
            return error(1)
        if ver != P.VERSION:
            return error(2)
        if mtype != P.T_HELLO or len(payload) != 8:
            return error(3)
        w, h = struct.unpack("<HH", payload[:4])
        if (w, h) != (P.FRAME, P.FRAME):
            return error(3)
        self._reset_state()                                           # engine_start(): soft reset
        conn.sendall(P.pack_msg(P.T_HELLO_ACK, 0, struct.pack("<IIHH", 0, P.EXPECTED_BUILD, P.FRAME, P.FRAME)))
        scratch = 0
        while True:
            magic, ver, mtype, seq, payload = recv()
            if magic != P.MAGIC:
                return error(1)
            if mtype == P.T_FRAME_IN:
                if len(payload) != P.FRAME * P.FRAME:
                    return error(3)
                frame = np.frombuffer(payload, np.uint8).reshape(P.FRAME, P.FRAME)
                conn.sendall(P.pack_msg(P.T_RESULT, seq, P.build_result_payload(self.process(frame, seq))))
            elif mtype == P.T_RESET and len(payload) == 0:
                self._reset_state()
                conn.sendall(P.pack_msg(P.T_HELLO_ACK, seq, struct.pack("<IIHH", 0, P.EXPECTED_BUILD, P.FRAME, P.FRAME)))
            elif mtype == P.T_CSR_WRITE and len(payload) == 8:
                a, v = struct.unpack("<II", payload)
                if a // 4 == 9:
                    self.theta = v & 0xFF
                    self.emu.cfg = EmuConfig(theta=max(self.theta, 1), mode="step")
                elif a // 4 == 11:
                    scratch = v
            elif mtype == P.T_CSR_READ and len(payload) == 4:
                a = struct.unpack("<I", payload)[0]
                val = {0: P.EXPECTED_BUILD, 8: self.done_seq, 9: self.theta, 10: 25000000, 11: scratch, 12: self.window,
                       16: self.pred, 17: self.windows, 18: self.wcount}.get(a // 4, 0)
                conn.sendall(P.pack_msg(P.T_CSR_REPLY, seq, struct.pack("<II", a, val)))
            else:
                return error(3)
