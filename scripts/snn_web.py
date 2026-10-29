"""Live demo frontend in the browser: play clips, upload your own video, or use the webcam, and watch the FPGA classify it.

    python scripts/snn_web.py --board 192.168.137.200        # real board (opens http://127.0.0.1:8080)
    python scripts/snn_web.py                                # no hardware: the bit-accurate Python models answer
    python scripts/snn_web.py --board 192.168.137.200 --port 8090 --no-browser

In the page: choose a sample clip (KTH test people) or upload any video file or press "Webcam"; choose Board or Model as the
backend; with "check against the model" on, every board answer is compared on the fly with the bit-accurate model and the page shows
a running count of matching frames. Frames are prepared exactly like the training data (grey, whole frame squashed to 128x128 with area
averaging, 30 fps). Webcam frames travel browser -> server as JPEG (about 320x240) and are then squashed here, so a webcam is slightly
noisier than a video file that is decoded on the server; a webcam that delivers fewer than 30 frames/s also moves more per network step
than the training video did (one frame is one step whatever its arrival time).

Only one client can talk to the board at a time (the ARM daemon is single-connection), so the connection is opened when a source starts
and closed when it stops, which leaves the board free for scripts/check_board.py in between. A watchdog stops a webcam source that
receives no frame for WEBCAM_IDLE_S seconds (tab closed without Stop) and any source with no page open for NO_VIEWER_S seconds, so a
forgotten tab cannot hold the board. Details and the test record: docs/live_demo.md.

All browser tabs share one session: every tab sees the same stream, and the last control pressed in any tab wins.

"Record training clips" saves webcam clips with a label for later training (independent of the classifier, no board needed):
data/recordings/<action>/<subject>_<action>_<YYYYmmdd-HHMMSS>.avi (the browser's frames, centre-cropped to 4:3, at most 640x480,
MJPG at the frame rate the camera really delivered) plus a .json sidecar with the label, the subject and every frame's capture time
(train/recordings.py turns them into exact 30 fps clips). Deleted recordings are moved to data/recordings/_trash/, not erased.
"""
import argparse
import base64
import collections
import datetime
import itertools
import json
import os
import queue
import re
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.emulator_int import EmuConfig, IntEmulator  # noqa: E402
from model.event_format import unpack  # noqa: E402
from sw.host import snnproto as P  # noqa: E402
from sw.host.snnboard_sim import SimBoard  # noqa: E402

PAGE = os.path.join(ROOT, "sw", "web", "index.html")
UPLOADS = os.path.join(ROOT, "work", "uploads")
USER_CLIPS = os.path.join(ROOT, "data", "user_clips")
RECORDINGS = os.path.join(ROOT, "data", "recordings")
KTH = os.path.join(ROOT, "data", "kth", "videos")
WEIGHTS = os.path.join(ROOT, "rtl", "weights", "kth6")
FIELDS = ["pred", "windows", "wcount", "spk1", "spk2", "events", "cur", "score", "win_score"]
VIDEO_EXT = (".avi", ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv")
KTH_NAMES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
KTH_DEMO_PEOPLE = ("person22_", "person02_")                 # two of the test people (never seen in training)
MAX_FRAME_BYTES = 4 << 20                                    # one webcam JPEG
WEBCAM_IDLE_S = 6.0
NO_VIEWER_S = 15.0                                           # a source with no page listening is stopped after this
FAST_PUBLISH_S = 1 / 45.0                                    # unpaced runs: at most ~45 pictures/s to the page
REC_ACTIONS = KTH_NAMES + ["none"]                          # folder names; "none" = no action (for a future reject class)
REC_SUBJECT = re.compile(r"[A-Za-z0-9-]{1,32}")              # no "_": it separates subject, action and time in the file name
REC_MAX_S = 60.0
REC_MIN_FRAMES = 15
REC_MAX_SIDE = 1920


def b64jpg(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def b64png(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def events_rgb(words, size=128):
    img = np.full((size, size, 3), 28, np.uint8)
    if len(words):
        u = unpack(words)
        on = u["pol"] == 1
        img[u["y"][on], u["x"][on]] = (90, 230, 60)           # BGR: green
        img[u["y"][~on], u["x"][~on]] = (60, 60, 235)         # BGR: red
    return img


def video_info(path):
    """(source fps as used for resampling, number of 30 fps frames the clip will give; 0 if unknown)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 30.0
    if n <= 0:
        return fps, 0
    if abs(fps - 30.0) < 0.05:
        return fps, n
    return fps, int((n - 1) * 30.0 / fps) + 1               # Resampler30 emits m while m * fps/30 <= n - 1


def frames_of(path, fit="squash"):
    """Yield 128x128 uint8 frames at 30 fps from a video file (same preparation as the training data)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open " + os.path.basename(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 30.0
    rs = None if abs(fps - 30.0) < 0.05 else P.Resampler30(fps)
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            f = P.prepare_frame(img, fit)
            if rs is None:
                yield f
            else:
                for g in rs.push(f):
                    yield g
    finally:
        cap.release()


class Stopped(Exception):
    """The source this work belongs to was stopped or replaced."""


class Backend:
    """One answer per frame, from the board or from the models; optional comparison of the two."""

    def __init__(self, kind, host, verify):
        self.kind, self.host, self.verify = kind, host, verify
        self.client = None
        self.sim = SimBoard(WEIGHTS) if (kind == "sim" or verify) else None
        self.emu = IntEmulator(EmuConfig(theta=9, mode="step"))        # only for the event picture
        self.checked = self.matched = 0
        self.first_mismatch = None
        self.n = 0
        if kind == "board":
            h, _, port = host.partition(":")
            try:
                self.client = P.Client(h, int(port) if port else 5002, timeout=3.0)
            except OSError as e:
                raise ConnectionError("cannot connect to the board at %s (%s). Is it on and is sw/snn_streamer running?" % (host, e))
            self.client.sock.settimeout(8.0)
            try:
                self.client.hello()
                self.client.reset()
            except OSError as e:
                self.close()
                raise ConnectionError("the board at %s accepted the connection but did not answer HELLO (%s). The daemon serves one "
                                      "client at a time: is another program (check_board.py, snn_viewer.py, another demo server) "
                                      "connected?" % (host, e))
            except Exception:
                self.close()
                raise

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def restart(self):
        """New clip (loop): engine, model and event picture start from scratch, as for a new connection."""
        if self.client:
            self.client.reset()
        if self.sim:
            self.sim._reset_state()
        self.emu.reset()
        self.n = 0

    def infer(self, frame):
        self.n += 1
        words = self.emu.step(frame)
        ref = self.sim.process(frame, self.n) if self.sim else None
        if self.kind == "board":
            self.client.send_frame(frame)
            res = self.client.recv_result()
        else:
            res = ref
        match = None
        if self.kind == "board" and self.verify:
            bad = [k for k in FIELDS if getattr(res, k) != getattr(ref, k)]
            match = not bad
            self.checked += 1
            self.matched += int(match)
            if bad and self.first_mismatch is None:
                self.first_mismatch = "frame %d: %s" % (self.n, ", ".join(bad))
        return res, words, match


class Run:
    """One source (a clip, possibly looping, or a webcam session) and everything that belongs to it."""

    ids = itertools.count(1)

    def __init__(self, source, cfg, default_board):
        self.id = next(Run.ids)
        self.source = source                                  # "clip" or "webcam"
        self.cfg = cfg
        self.kind = "board" if cfg.get("backend") == "board" else "sim"
        self.host = (cfg.get("host") or default_board or "192.168.137.200").strip()
        self.verify = bool(cfg.get("verify", True)) and self.kind == "board"
        self.stop = threading.Event()
        self.lock = threading.Lock()                          # one frame at a time through the backend
        self.backend = None
        self.thread = None
        self.token = secrets.token_hex(8)
        self.last_frame_t = time.time()
        self.idx = 0
        self.stamps = collections.deque(maxlen=31)
        self.last_pub = 0.0
        self.new_pass()

    def new_pass(self):
        self.window_preds, self.totals, self.last_windows, self.last_res = [], np.zeros(P.NCLS, np.int64), 0, None

    def open(self):
        self.backend = Backend(self.kind, self.host, self.verify)

    def close(self):
        with self.lock:
            if self.backend:
                self.backend.close()

    def clip_answer(self):
        """The viewer's rule: largest sum of the finished windows' scores; a clip shorter than one window uses its running score."""
        if self.last_windows:
            return int(np.argmax(self.totals)), "sum of the scores of %d finished windows" % self.last_windows
        if self.last_res is not None:
            return self.last_res.leader, "clip shorter than one 32-frame window: running score of %d frames" % self.idx
        return None, "no frames"


class Hub:
    """State shared by the HTTP handlers, the clip thread and the SSE subscribers."""

    def __init__(self, default_board):
        self.lock = threading.Lock()                          # subs and status
        self.ctl = threading.Lock()                           # start / stop
        self.subs = []
        self.run = None
        self.default_board = default_board
        self.status = {"running": False, "source": None, "msg": "idle"}

    # ---- publishing ----
    def publish(self, obj):
        data = ("data: " + json.dumps(obj) + "\n\n").encode()
        with self.lock:
            for q in self.subs:
                try:
                    q.put_nowait(data)
                except queue.Full:                            # slow tab: drop its oldest message, keep the newest
                    try:
                        q.get_nowait()
                        q.put_nowait(data)
                    except (queue.Empty, queue.Full):
                        pass

    def set_status(self, **kw):
        with self.lock:
            self.status.update(kw)
            st = dict(self.status)
        self.publish({"type": "status", **st})

    def set_idle(self):
        with self.lock:
            self.status = {"running": False, "source": None, "msg": "idle", "run": None}
        self.publish({"type": "status", **self.status})

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def subscribe(self):
        q = queue.Queue(maxsize=90)
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    # ---- one frame ----
    def handle_frame(self, run, frame, name, total, truth=""):
        with run.lock:
            if run.stop.is_set() or run.backend is None or self.run is not run:
                raise Stopped()
            t1 = time.time()
            res, words, match = run.backend.infer(frame)
            rtt = time.time() - t1
            be = run.backend
            run.last_res = res
            new_window = res.windows > run.last_windows
            if new_window:
                run.last_windows = res.windows
                run.window_preds.append(res.pred)
                run.totals += np.array(res.win_score, np.int64)
            now = time.time()
            run.stamps.append(now)
            run.last_frame_t = now
            idx = run.idx
            run.idx += 1
            fps = (len(run.stamps) - 1) / max(run.stamps[-1] - run.stamps[0], 1e-6) if len(run.stamps) > 1 else 0.0
            last = total and run.idx >= total
            if run.cfg.get("fast") and not new_window and not last and now - run.last_pub < FAST_PUBLISH_S:
                return res
            run.last_pub = now
            msg = {
                "type": "frame", "run": run.id, "source": run.source, "name": name, "truth": truth, "idx": idx, "total": total,
                "backend": be.kind, "frame": b64jpg(frame), "events": b64png(events_rgb(words)),
                "score": res.score, "cur": res.cur, "win_score": res.win_score, "pred": res.pred, "windows": res.windows,
                "wcount": res.wcount, "leader": res.leader, "new_window": new_window,
                "spk1": res.spk1, "spk2": res.spk2, "nevents": res.events, "latency_ms": res.latency_cycles / 25000.0,
                "step_ms": res.step_cycles / 25000.0, "rtt_ms": rtt * 1000.0, "fps": fps, "window_preds": run.window_preds[-24:],
                "verify": be.verify, "checked": be.checked, "matched": be.matched, "match": match, "first_mismatch": be.first_mismatch,
            }
        self.publish(msg)
        return res

    # ---- sources ----
    def _stop_locked(self, why=None):
        run = self.run
        if run is None:
            return
        run.stop.set()
        if run.thread and run.thread.is_alive() and run.thread is not threading.current_thread():
            run.thread.join(timeout=12)                       # longer than one board timeout (8 s)
        run.close()
        with self.lock:
            self.run = None
        if why:
            self.publish({"type": "error", "msg": why, "run": run.id})
        self.set_idle()

    def stop(self, why=None):
        with self.ctl:
            self._stop_locked(why)

    def start_clip(self, cfg, path, truth):
        with self.ctl:
            self._stop_locked()
            run = Run("clip", cfg, self.default_board)
            self.run = run
            run.thread = threading.Thread(target=self.run_clip, args=(run, path, truth), daemon=True)
            run.thread.start()
            return run

    def start_webcam(self, cfg):
        with self.ctl:
            self._stop_locked()
            run = Run("webcam", cfg, self.default_board)
            run.open()                                        # raises if the board is not there
            self.run = run
            self.set_status(running=True, source="webcam", name="webcam", msg="webcam", backend=run.kind, host=run.host,
                            verify=run.verify, run=run.id, truth="")
            return run

    def run_clip(self, run, path, truth):
        name = os.path.basename(path)
        try:
            self.set_status(running=True, source="clip", name=name, truth=truth, msg="connecting" if run.kind == "board" else "playing",
                            backend=run.kind, host=run.host, verify=run.verify, run=run.id)
            run.open()
            if run.stop.is_set():
                return
            self.set_status(msg="playing")
            _, total = video_info(path)
            loop, fast = bool(run.cfg.get("loop")), bool(run.cfg.get("fast"))
            while not run.stop.is_set():
                t_next = time.time()
                for f in frames_of(path, "squash"):
                    if run.stop.is_set():
                        break
                    self.handle_frame(run, f, name, total, truth)
                    if not fast:
                        t_next += 1 / 30.0
                        d = t_next - time.time()
                        if d > 0:
                            time.sleep(d)
                        elif d < -0.25:                       # fell behind (slow board / laptop): do not try to catch up
                            t_next = time.time()
                if run.stop.is_set():
                    break
                if run.idx == 0:
                    raise IOError("no frames could be read from %s" % name)
                cls, how = run.clip_answer()
                be = run.backend
                self.publish({"type": "clip_done", "run": run.id, "name": name, "truth": truth, "frames": run.idx,
                              "answer": None if cls is None else P.CLASSES[cls], "how": how, "totals": [int(v) for v in run.totals],
                              "windows": [P.CLASSES[p] for p in run.window_preds[-40:]], "backend": be.kind,
                              "verify": be.verify, "checked": be.checked, "matched": be.matched})
                if not loop:
                    break
                with run.lock:
                    run.backend.restart()
                    run.new_pass()
                    run.idx = 0
                    run.stamps.clear()
        except Stopped:
            pass
        except Exception as e:                      # noqa: BLE001 - show every failure in the page
            if not run.stop.is_set():
                self.publish({"type": "error", "msg": "%s: %s" % (type(e).__name__, e), "backend": run.kind, "run": run.id})
        finally:
            run.close()
            with self.lock:                                   # not hub.ctl: stop() may hold it while it joins this thread
                mine = self.run is run
                if mine:
                    self.run = None
            if mine:
                self.set_idle()

    def webcam_frame(self, token, jpg):
        run = self.run
        if run is None or run.source != "webcam" or run.token != token:
            return 409, {"ok": False, "msg": "this webcam session is no longer active (stopped, or replaced by another source or tab)"}
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return 400, {"ok": False, "msg": "not a JPEG/PNG image"}
        try:
            self.handle_frame(run, P.prepare_frame(img, "squash"), "webcam", 0)
        except Stopped:
            return 409, {"ok": False, "msg": "webcam session stopped"}
        except Exception as e:                      # noqa: BLE001
            msg = "%s: %s" % (type(e).__name__, e)
            self.stop(why=msg)
            return 500, {"ok": False, "msg": msg}
        return 200, {"ok": True}

    def watchdog(self):
        """Release the board when nobody is using it: a webcam that stopped sending, or a (looping) clip with no page open."""
        unwatched = None
        while True:
            time.sleep(1.0)
            run = self.run
            why = None
            with self.lock:
                nsubs = len(self.subs)
            unwatched = None if (nsubs or run is None) else (unwatched or time.time())
            if run and run.source == "webcam" and time.time() - run.last_frame_t > WEBCAM_IDLE_S:
                why = "webcam stopped: no frame for %.0f s (tab closed or hidden?); the board connection was released" % WEBCAM_IDLE_S
            elif run and unwatched and time.time() - unwatched > NO_VIEWER_S:
                why = "stopped: no page has been open for %.0f s; the board connection was released" % NO_VIEWER_S
            if why:
                with self.ctl:
                    if self.run is run:
                        self._stop_locked(why)
                unwatched = None


def clip_list(rec_root=None):
    rec_root = rec_root or RECORDINGS
    dirs = [(KTH, "KTH test people"), (USER_CLIPS, "your clips")]
    if os.path.isdir(rec_root):
        dirs += [(os.path.join(rec_root, a), "your recordings") for a in REC_ACTIONS]
    dirs.append((UPLOADS, "uploaded"))
    out = []
    for d, label in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(VIDEO_EXT) or f.startswith("_"):      # "_" = a recording still being written
                continue
            if d == KTH and not f.startswith(KTH_DEMO_PEOPLE):
                continue
            truth = ""
            for k, c in enumerate(KTH_NAMES):
                if "_" + c + "_" in f:
                    truth = P.CLASSES[k]
            out.append({"path": os.path.join(d, f), "group": label, "file": f, "truth": truth})
    return out


def safe_name(name):
    base, ext = os.path.splitext(os.path.basename(name.replace("\\", "/")))
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")[:80] or "upload"
    return base + ext.lower()


# ---------- recordings (training data) ----------
_rec_lock = threading.Lock()                                  # unique file names when two saves arrive together


def save_recording(body, root=None):
    """Body from the page: uint32 LE header length, JSON header, then the JPEG frames back to back.

    Header: action (REC_ACTIONS), subject (REC_SUBJECT), note, sizes [bytes per JPEG], stamps_ms [capture time per frame],
    missed (frames the page saw presented but could not grab), timing ("frame-callback" or "timer"), camera, user_agent.
    Writes <root>/<action>/<subject>_<action>_<time>.avi + .json and returns the sidecar dict. Raises ValueError on bad input."""
    root = root or RECORDINGS
    if len(body) < 4:
        raise ValueError("empty recording")
    hl = int.from_bytes(body[:4], "little")
    if hl <= 0 or hl > (4 << 20) or 4 + hl > len(body):
        raise ValueError("bad header length")
    try:
        h = json.loads(body[4:4 + hl].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError("bad header: %s" % e)
    if not isinstance(h, dict):
        raise ValueError("bad header")
    action, subject = h.get("action"), str(h.get("subject", ""))
    if action not in REC_ACTIONS:
        raise ValueError("unknown action %r (one of %s)" % (action, ", ".join(REC_ACTIONS)))
    if not REC_SUBJECT.fullmatch(subject):
        raise ValueError("subject must be 1-32 letters, digits or '-'")
    sizes, stamps = h.get("sizes"), h.get("stamps_ms")
    if not isinstance(sizes, list) or not isinstance(stamps, list) or len(sizes) != len(stamps):
        raise ValueError("sizes and stamps_ms must be lists of the same length")
    n = len(sizes)
    if n < REC_MIN_FRAMES:
        raise ValueError("only %d frames; record at least %d" % (n, REC_MIN_FRAMES))
    try:
        sizes = [int(s) for s in sizes]
        stamps = [float(t) for t in stamps]
    except (TypeError, ValueError):
        raise ValueError("sizes / stamps_ms must be numbers")
    if min(sizes) <= 0 or sum(sizes) != len(body) - 4 - hl:
        raise ValueError("frame sizes do not add up to the body length")
    if any(not np.isfinite(t) for t in stamps) or any(b < a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("frame times must be finite and in order")
    dur = (stamps[-1] - stamps[0]) / 1000.0
    if not 0.2 < dur <= REC_MAX_S + 2.0:
        raise ValueError("duration %.1f s outside 0.2-%.0f s" % (dur, REC_MAX_S))
    frames, pos = [], 4 + hl
    for k, s in enumerate(sizes):
        img = cv2.imdecode(np.frombuffer(body, np.uint8, s, pos), cv2.IMREAD_COLOR)
        pos += s
        if img is None:
            raise ValueError("frame %d is not a JPEG/PNG image" % k)
        if frames and img.shape != frames[0].shape:
            raise ValueError("frame %d is %dx%d, the first was %dx%d" % (k, img.shape[1], img.shape[0], frames[0].shape[1], frames[0].shape[0]))
        if max(img.shape[:2]) > REC_MAX_SIDE:
            raise ValueError("frames larger than %d pixels" % REC_MAX_SIDE)
        frames.append(img)
    fps = (n - 1) / dur
    gaps = np.diff(stamps)
    warnings = []
    if fps < 25.0:
        warnings.append("the camera delivered only %.1f frames/s (the training video is 30): more light usually helps" % fps)
    if gaps.max() > 3 * np.median(gaps):
        warnings.append("a gap of %.0f ms between two frames (tab hidden or computer busy?)" % gaps.max())
    if int(h.get("missed") or 0) > 0:
        warnings.append("%d camera frames could not be grabbed by the page" % int(h["missed"]))
    hgt, wid = frames[0].shape[:2]
    d = os.path.join(root, action)
    os.makedirs(d, exist_ok=True)
    with _rec_lock:
        base = "%s_%s_%s" % (subject, action, time.strftime("%Y%m%d-%H%M%S"))
        name, k = base, 1
        while os.path.exists(os.path.join(d, name + ".avi")):
            k += 1
            name = "%s-%d" % (base, k)
        tmp = os.path.join(d, "_writing_" + name + ".avi")         # hidden from the clip list until complete
        open(tmp, "wb").close()                                      # reserve the name
    try:
        w = cv2.VideoWriter(tmp, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"MJPG"), fps, (wid, hgt))
        if not w.isOpened():
            raise IOError("OpenCV cannot write MJPG AVI files")
        for img in frames:
            w.write(img)
        w.release()
        cap = cv2.VideoCapture(tmp)
        n_back = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if n_back != n:
            raise IOError("wrote %d frames but the file holds %d" % (n, n_back))
        avi = os.path.join(d, name + ".avi")
        os.replace(tmp, avi)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    meta = {
        "file": name + ".avi", "action": action,
        "label": KTH_NAMES.index(action) if action in KTH_NAMES else None,
        "label_names": KTH_NAMES, "subject": subject, "note": str(h.get("note", ""))[:500],
        "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "frames": n, "duration_s": round(dur, 3), "fps": round(fps, 3), "width": wid, "height": hgt,
        "frame_interval_ms": {"median": round(float(np.median(gaps)), 2), "max": round(float(gaps.max()), 2)},
        "missed": int(h.get("missed") or 0), "timing": str(h.get("timing", ""))[:40],
        "stamps_ms": [round(t - stamps[0], 2) for t in stamps],
        "source": "webcam, browser JPEG q0.92, centre crop to 4:3, MJPG AVI (OpenCV/FFmpeg default quality)",
        "camera": h.get("camera") if isinstance(h.get("camera"), dict) else {},
        "user_agent": str(h.get("user_agent", ""))[:300], "warnings": warnings,
    }
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    meta["path"] = avi
    return meta


def list_recordings(root=None):
    root = root or RECORDINGS
    out = []
    for a in REC_ACTIONS:
        d = os.path.join(root, a)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f.startswith("_"):
                continue
            avi = os.path.join(d, f[:-5] + ".avi")
            if not os.path.exists(avi):
                continue
            try:
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    m = json.load(fh)
            except (OSError, ValueError):
                continue
            out.append({"path": avi, "file": m.get("file", f[:-5] + ".avi"), "action": a, "subject": m.get("subject", ""),
                        "frames": m.get("frames"), "duration_s": m.get("duration_s"), "fps": m.get("fps"),
                        "recorded_at": m.get("recorded_at", ""), "warnings": m.get("warnings", [])})
    out.sort(key=lambda r: r["recorded_at"], reverse=True)
    return out


def trash_recording(path, root=None):
    """Move a recording (.avi + .json) to <root>/_trash/<action>/; only files directly inside <root>/<action>/ are accepted."""
    root = os.path.abspath(root or RECORDINGS)
    p = os.path.abspath(str(path))
    d, f = os.path.split(p)
    if os.path.dirname(d) != root or os.path.basename(d) not in REC_ACTIONS or not f.endswith(".avi") or f.startswith("_") \
            or not os.path.isfile(p):
        raise ValueError("not a recording")
    tdir = os.path.join(root, "_trash", os.path.basename(d))
    os.makedirs(tdir, exist_ok=True)
    stem = f[:-4]
    for ext in (".avi", ".json"):
        src = os.path.join(d, stem + ext)
        if os.path.exists(src):
            dst, k = os.path.join(tdir, stem + ext), 1
            while os.path.exists(dst):
                k += 1
                dst = os.path.join(tdir, "%s-%d%s" % (stem, k, ext))
            os.rename(src, dst)                                       # same disk: all or nothing (a playing file raises)


def make_handler(hub, args):
    max_upload = args.max_upload_mb << 20

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body)
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _length(self, limit):
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                n = -1
            if n < 0 or n > limit:
                self.close_connection = True                  # the body is not read: do not reuse this connection
                self._send(413 if n > limit else 411, {"ok": False, "msg": "body missing or larger than %d MB" % (limit >> 20)})
                return None
            return n

        def _json(self):
            n = self._length(1 << 16)
            if n is None:
                return None
            try:
                v = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(v, dict):
                    raise ValueError("not an object")
                return v
            except ValueError as e:
                self._send(400, {"ok": False, "msg": "bad JSON: %s" % e})
                return None

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                try:
                    with open(PAGE, "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    return self._send(500, "sw/web/index.html is missing", "text/plain")
            if path == "/api/clips":
                return self._send(200, {"clips": clip_list(), "classes": P.CLASSES, "board": args.board or "",
                                        "default_backend": "board" if args.board else "sim", "status": hub.get_status(),
                                        "max_upload_mb": args.max_upload_mb})
            if path == "/api/recordings":
                recs = list_recordings()
                return self._send(200, {"recordings": recs, "actions": REC_ACTIONS, "root": RECORDINGS, "max_s": REC_MAX_S,
                                        "counts": dict(collections.Counter(r["action"] for r in recs))})
            if path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                q = hub.subscribe()
                try:
                    self.wfile.write(b"retry: 2000\n\n")
                    self.wfile.write(("data: " + json.dumps({"type": "status", **hub.get_status()}) + "\n\n").encode())
                    self.wfile.flush()
                    while True:
                        try:
                            self.wfile.write(q.get(timeout=10))
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except OSError:                               # tab closed (BrokenPipe, ConnectionReset, ConnectionAborted)
                    pass
                finally:
                    hub.unsubscribe(q)
                return
            self._send(404, {"ok": False})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path == "/api/start":
                cfg = self._json()
                if cfg is None:
                    return
                want = os.path.normcase(os.path.abspath(str(cfg.get("path", ""))))
                clip = next((c for c in clip_list() if os.path.normcase(os.path.abspath(c["path"])) == want), None)
                if clip is None:
                    return self._send(400, {"ok": False, "msg": "unknown clip (only the listed clips can be played)"})
                run = hub.start_clip(cfg, clip["path"], clip["truth"])
                return self._send(200, {"ok": True, "run": run.id})
            if path == "/api/stop":
                n = self._length(1 << 16)
                if n is None:
                    return
                self.rfile.read(n)
                hub.stop()
                return self._send(200, {"ok": True})
            if path == "/api/upload":
                name = safe_name(self.headers.get("X-Filename", "upload.mp4"))
                if not name.lower().endswith(VIDEO_EXT):
                    self.close_connection = True
                    return self._send(415, {"ok": False, "msg": "not a video file name (%s)" % ", ".join(VIDEO_EXT)})
                n = self._length(max_upload)
                if n is None:
                    return
                os.makedirs(UPLOADS, exist_ok=True)
                dest = os.path.join(UPLOADS, name)
                tmp = dest + ".part"
                left = n
                with open(tmp, "wb") as f:
                    while left:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                if left:
                    os.remove(tmp)
                    self.close_connection = True
                    return self._send(400, {"ok": False, "msg": "upload interrupted"})
                os.replace(tmp, dest)
                cap = cv2.VideoCapture(dest)
                ok = cap.isOpened() and cap.read()[0]
                cap.release()
                if not ok:
                    os.remove(dest)
                    return self._send(415, {"ok": False, "msg": "OpenCV cannot decode %s" % name})
                return self._send(200, {"ok": True, "path": dest})
            if path == "/api/webcam/start":
                cfg = self._json()
                if cfg is None:
                    return
                try:
                    run = hub.start_webcam(cfg)
                except Exception as e:                        # noqa: BLE001
                    hub.publish({"type": "error", "msg": "%s: %s" % (type(e).__name__, e), "backend": cfg.get("backend")})
                    return self._send(200, {"ok": False, "msg": "%s: %s" % (type(e).__name__, e)})
                return self._send(200, {"ok": True, "token": run.token, "run": run.id})
            if path == "/api/webcam/frame":
                n = self._length(MAX_FRAME_BYTES)
                if n is None:
                    return
                body = self.rfile.read(n)
                code, obj = hub.webcam_frame(self.headers.get("X-Session", ""), body)
                return self._send(code, obj)
            if path == "/api/record":
                n = self._length(max_upload)
                if n is None:
                    return
                body = self.rfile.read(n)
                if len(body) != n:
                    self.close_connection = True
                    return self._send(400, {"ok": False, "msg": "recording upload interrupted"})
                try:
                    meta = save_recording(body)
                except ValueError as e:
                    return self._send(400, {"ok": False, "msg": str(e)})
                except Exception as e:                        # noqa: BLE001
                    return self._send(500, {"ok": False, "msg": "%s: %s" % (type(e).__name__, e)})
                return self._send(200, {"ok": True, "path": meta["path"], "file": meta["file"], "frames": meta["frames"],
                                        "duration_s": meta["duration_s"], "fps": meta["fps"], "width": meta["width"],
                                        "height": meta["height"], "warnings": meta["warnings"]})
            if path == "/api/recordings/delete":
                req = self._json()
                if req is None:
                    return
                try:
                    trash_recording(req.get("path", ""))
                except ValueError as e:
                    return self._send(400, {"ok": False, "msg": str(e)})
                except OSError as e:
                    return self._send(409, {"ok": False, "msg": "could not move it (is it playing? press Stop first): %s" % e})
                return self._send(200, {"ok": True})
            self.close_connection = True
            self._send(404, {"ok": False})

    return H


def main():
    global RECORDINGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", help="HOST[:PORT] of the board (default for the page's Board backend)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to let a phone open the page (camera needs localhost or HTTPS)")
    ap.add_argument("--max-upload-mb", type=int, default=200)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--recordings", default=RECORDINGS, help="where recorded training clips go (default data/recordings)")
    a = ap.parse_args()
    RECORDINGS = os.path.abspath(a.recordings)
    hub = Hub(a.board)
    threading.Thread(target=hub.watchdog, daemon=True).start()
    srv = ThreadingHTTPServer((a.bind, a.port), make_handler(hub, a))
    srv.daemon_threads = True
    url = "http://127.0.0.1:%d/" % a.port
    print("live demo at", url, "  (board: %s)" % (a.board or "not set, the page starts with the Python models"), flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()


if __name__ == "__main__":
    main()
