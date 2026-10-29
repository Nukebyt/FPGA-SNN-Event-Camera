"""Frontend: send video clips to the board (or to the simulated board) and see what the network decides.

    python scripts/snn_viewer.py --source data/kth/videos/person22_boxing_d1_uncomp.avi --connect 192.168.137.200
    python scripts/snn_viewer.py --source clipA.mp4 clipB.mp4 webcam:0 --connect 192.168.137.200 --loop
    python scripts/snn_viewer.py --source clip.avi --sim                      # no hardware: the Python models answer
    python scripts/snn_viewer.py --source clip.avi --sim --headless --report out.json

What it does: every source frame is converted exactly like the training data (grey, whole frame squashed to 128x128, 30 fps with
the same 25 -> 30 fps blending), sent over TCP (docs/snn_soc.md), and the board answers per frame with the class currents and,
every 32 frames, the class of the finished window. The window shows the frame, the DVS events of that frame (computed here with
the same bit-accurate model, so you can see what the emulator in the FPGA sees), and the running scores of the six classes.
For a clip the final answer is the class with the largest sum of window scores (how the sample accuracy was measured).
Keys: q quit, space pause. Without a display (or with --headless) it only prints.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sw.host import snnproto as P  # noqa: E402

COLORS = [(80, 160, 255), (80, 200, 255), (60, 120, 255), (60, 220, 120), (255, 180, 80), (200, 120, 255)]   # BGR per class


def frames_30fps(spec, fit="squash"):
    """Yield (128x128 uint8 frame at 30 fps, source name) from a video file or webcam:N."""
    cap = cv2.VideoCapture(int(spec.split(":")[1]) if spec.startswith("webcam:") else spec)
    if not cap.isOpened():
        raise IOError("cannot open %s" % spec)
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


class Display:
    def __init__(self, enabled, save=None):
        self.enabled = enabled
        self.writer = None
        self.save = save
        self.paused = False

    def show(self, frame, events_img, res, title, window_preds, T=32):
        W, H = 1000, 420
        c = np.full((H, W, 3), 24, np.uint8)
        c[18:18 + 384, 10:10 + 384] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), (384, 384), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, "input 128x128", (14, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        e = cv2.resize(events_img, (192, 192), interpolation=cv2.INTER_NEAREST)
        c[18:18 + 192, 404:404 + 192] = e
        cv2.putText(c, "DVS events of this frame (model)", (404, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        # running window scores as bars
        sc = np.array(res.score, np.float64)
        span = max(np.abs(sc).max(), 1.0)
        x0, y0 = 610, 40
        cv2.putText(c, "running window score", (x0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        for k in range(P.NCLS):
            y = y0 + 34 * k
            lead = k == res.leader
            cv2.putText(c, P.CLASSES[k], (x0, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255) if lead else (160, 160, 160), 1)
            mid = x0 + 230
            w = int(110 * sc[k] / span)
            cv2.rectangle(c, (mid, y), (mid + w, y + 18), COLORS[k] if lead else (110, 110, 110), -1)
        cv2.line(c, (x0 + 230, y0 - 4), (x0 + 230, y0 + 34 * 6), (200, 200, 200), 1)
        last = window_preds[-1] if window_preds else None
        txt = "board answer (last window): %s" % (P.CLASSES[last] if last is not None else "-- (window not finished)")
        cv2.putText(c, txt, (x0, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120) if last is not None else (160, 160, 160), 1)
        cv2.putText(c, "%s   step %d / %d   windows %d" % (title[:34], res.wcount, T, res.windows), (x0, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(c, "events %d   spikes L1 %d L2 %d" % (res.events, res.spk1, res.spk2), (x0, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(c, "latency %.1f ms  (SNN step %.1f ms)" % (res.latency_cycles / 25000.0, res.step_cycles / 25000.0), (x0, 344),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if self.save:
            if self.writer is None:
                self.writer = cv2.VideoWriter(self.save, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
            self.writer.write(c)
        if self.enabled:
            cv2.imshow("event camera + SNN", c)
            while True:
                k = cv2.waitKey(1 if not self.paused else 50) & 0xFF
                if k == ord("q"):
                    return False
                if k == ord(" "):
                    self.paused = not self.paused
                if not self.paused:
                    break
        return True

    def close(self):
        if self.writer:
            self.writer.release()
        if self.enabled:
            cv2.destroyAllWindows()


def events_image(words, size=128):
    """Colour image of the emulator events of one frame: green = ON (brighter), red = OFF (darker)."""
    from model.event_format import unpack
    img = np.full((size, size, 3), 30, np.uint8)
    if len(words):
        u = unpack(words)
        on = u["pol"] == 1
        img[u["y"][on], u["x"][on]] = (0, 255, 0)
        img[u["y"][~on], u["x"][~on]] = (0, 0, 255)
    return img


def run_clip(client, spec, args, disp, emu):
    client.reset()
    if emu is not None:
        emu.reset()
    results, window_preds, t_start = [], [], time.time()
    last_windows = 0
    period = 1.0 / 30.0
    n = 0
    for frame in frames_30fps(spec, args.fit):
        if args.max_frames and n >= args.max_frames:
            break
        t0 = time.time()
        client.send_frame(frame)
        res = client.recv_result()
        results.append(res)
        if res.windows > last_windows:
            last_windows = res.windows
            window_preds.append(res.pred)
            print("  %-34s window %d finished at frame %d: %s   scores %s" % (os.path.basename(spec)[:34], res.windows, n + 1, P.CLASSES[res.pred], res.win_score))
        words = emu.step(frame) if emu is not None else np.zeros(0, np.uint32)
        if not disp.show(frame, events_image(words), res, os.path.basename(spec), window_preds):
            return None
        n += 1
        if not args.fast:
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    if not results:
        raise IOError("no frames read from %s" % spec)
    # clip decision: sum of the finished windows' scores (as in sample accuracy); fall back to the running score of a short clip
    totals = np.zeros(P.NCLS, np.int64)
    seen = 0
    for r in results:
        if r.wcount == 0 and r.windows > seen:       # the frame that closed a window: its win_score is that window
            totals += np.array(r.win_score, np.int64)
            seen = r.windows
    short = seen == 0
    if short:
        totals = np.array(results[-1].score, np.int64)
    clip_class = int(np.argmax(totals))
    elapsed = time.time() - t_start
    print("CLIP %s: %d frames, %d windows -> %s%s   (%.1f frames/s, mean latency %.1f ms, step %.1f ms)" % (
        os.path.basename(spec), len(results), seen, P.CLASSES[clip_class], " (clip shorter than one window: running score)" if short else "",
        len(results) / max(elapsed, 1e-9), np.mean([r.latency_cycles for r in results]) / 25000.0, np.mean([r.step_cycles for r in results]) / 25000.0))
    return {"source": spec, "frames": len(results), "windows": seen, "window_classes": [P.CLASSES[p] for p in window_preds],
            "clip_class": P.CLASSES[clip_class], "totals": [int(v) for v in totals],
            "mean_events": float(np.mean([r.events for r in results])), "errors": int(any(r.status & 0x18 for r in results))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", required=True, help="video files and/or webcam:N, played one after another")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--connect", help="HOST[:PORT] of the board (sw/snn_streamer.c)")
    g.add_argument("--sim", action="store_true", help="in-process simulated board running the Python models")
    ap.add_argument("--weights", default="rtl/weights/kth6", help="network for --sim")
    ap.add_argument("--fit", default="squash", choices=["squash", "crop"], help="squash = what the network was trained on")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fast", action="store_true", help="do not pace to 30 fps")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--report", help="write the per-clip results to this JSON file")
    a = ap.parse_args()

    board = None
    if a.sim:
        from sw.host.snnboard_sim import SimBoard
        board = SimBoard(a.weights)
        host, port = "127.0.0.1", board.start()
    else:
        host, _, p = a.connect.partition(":")
        port = int(p) if p else 5002
    client = P.Client(host, port)
    print("build id 0x%08X" % client.hello())
    from model.emulator_int import EmuConfig, IntEmulator
    emu = IntEmulator(EmuConfig(theta=9, mode="step"))
    disp = Display(enabled=not a.headless, save=a.save)
    report = []
    try:
        while True:
            for spec in a.source:
                r = run_clip(client, spec, a, disp, emu)
                if r is None:
                    return
                report.append(r)
            if not a.loop:
                break
    finally:
        disp.close()
        client.close()
        if board:
            board.stop()
        if a.report:
            with open(a.report, "w") as f:
                json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
