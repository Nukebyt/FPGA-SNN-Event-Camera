"""Check recorded gesture clips before they go into the pipeline (run this on your first 2-3 clips!).

    python sw/check_clip.py data/own_clips                 # every video below the folder
    python sw/check_clip.py path/to/clip.mov --out work/clipcheck

For each clip: decode like the pipeline does (centre square crop, 128x128 gray, 30 fps), run the integer event
emulator (theta 9 = 0.2 ln units) and report PASS or the list of WARN items, plus a contact sheet PNG
(frame | events) so you can look at it. Thresholds come from the DVS128 Gesture statistics measured in
docs/recording_guide.md; they are guides, not laws. Exit code is always 0.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.emulator_int import EmuConfig, IntEmulator  # noqa: E402
from model.event_format import unpack  # noqa: E402
from model.log_lut import theta_from_ln  # noqa: E402
from model.video_io import read_video  # noqa: E402

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
MIN_ACTIVE_EVENTS = 30        # floor for calling a frame "active" (the real threshold adapts to the clip, see check())
PATCH = 24                    # corner patch size (128-px frame) used as static-background probes


def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 255) for i in range(4)) if v else "?"


def check(path, out_dir):
    warns, info = [], []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ["cannot open the file (unsupported codec?)"], ["-"], None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    info.append("%dx%d %.2f fps %s %.1f s" % (w, h, fps, codec, n / fps if fps else 0))
    if w < h:
        warns.append("portrait video (%dx%d): record in landscape" % (w, h))
    if min(w, h) < 720:
        warns.append("resolution below 720p")
    if not (29.5 <= fps <= 30.5):
        warns.append("%.2f fps: not 30 (60 fps is dropped to 30, anything else is resampled badly)" % fps)

    frames, _ = read_video(path, 128, 30.0)
    if len(frames) < 90:
        warns.append("only %.1f s of video" % (len(frames) / 30.0))
    f = frames.astype(float)
    lo, hi = np.percentile(frames, [1, 99])
    info.append("luma p1 %.0f p99 %.0f" % (lo, hi))
    if hi - lo < 60:
        warns.append("low contrast (p99-p1 = %.0f): flat/log/HDR profile, or an empty scene" % (hi - lo))
    if np.median(frames) < 50:
        warns.append("dark image (median luma %.0f): add light, noise will create false events" % np.median(frames))
    if (frames >= 250).mean() > 0.03:
        warns.append("%.1f%% of pixels are clipped white" % (100 * (frames >= 250).mean()))

    # static-background probes: top corners of the square crop
    patches = [f[:, :PATCH, :PATCH].mean((1, 2)), f[:, :PATCH, -PATCH:].mean((1, 2))]
    drift = max(np.abs(p - np.median(p)).max() / max(np.median(p), 1) for p in patches)
    k = np.ones(15) / 15
    fast = max(np.std((p - np.convolve(p, k, "same"))[8:-8]) / max(np.median(p), 1) for p in patches)   # interior only: 'same' zero-pads the ends
    info.append("background drift %.1f%% flicker %.2f%%" % (100 * drift, 100 * fast))
    if drift > 0.06:
        warns.append("background brightness drifts %.0f%% during the clip: auto exposure or auto white balance is on, or a light changed" % (100 * drift))
    if fast > 0.012:
        warns.append("background flickers (%.1f%% frame to frame): use shutter 1/100 s under artificial light at 50 Hz" % (100 * fast))

    emu = IntEmulator(EmuConfig(theta=theta_from_ln(0.2), mode="step"))
    counts, cx, cy, edge, corner = [], [], [], 0, []
    ev_img = []
    for fr in frames:
        u = unpack(emu.step(fr))
        x, y = u["x"].astype(int), u["y"].astype(int)
        counts.append(len(x))
        corner.append(((x < PATCH) & (y < PATCH)).sum() + ((x >= 128 - PATCH) & (y < PATCH)).sum())
        if len(x) > MIN_ACTIVE_EVENTS:
            cx.append(np.median(x)); cy.append(np.median(y))
            edge += ((x < 6) | (x > 121) | (y < 6) | (y > 121)).sum()
        ev_img.append((x, y, u["pol"]))
    counts = np.array(counts)
    act_thr = max(MIN_ACTIVE_EVENTS, 0.2 * np.percentile(counts, 90))     # adapts to how many events this scene produces
    active = np.nonzero(counts > act_thr)[0]
    bg = np.mean(corner) / (2 * PATCH * PATCH)
    info.append("events/frame median %.0f (active if > %.0f), background events %.2f%% of corner pixels per frame" % (np.median(counts), act_thr, 100 * bg))
    if bg > 0.005:
        warns.append("many events in the static background (%.1f%% of the corner pixels per frame): noise, compression or flicker" % (100 * bg))
    if len(active) < 30:
        warns.append("almost no motion detected (%d active frames)" % len(active))
    else:
        span = (active[-1] - active[0] + 1) / 30.0
        dens = len(active) / (active[-1] - active[0] + 1)
        info.append("active %.1f s (%.0f%% of that span)" % (span, 100 * dens))
        if span < 4.5:
            warns.append("gesture lasts only %.1f s: perform it for 6-8 s" % span)
        if dens < 0.75:
            warns.append("motion is present in only %.0f%% of the active span: keep moving without pauses" % (100 * dens))
        mx, my = 100 * np.median(cx) / 128, 100 * np.median(cy) / 128
        info.append("motion centre x %.0f%% y %.0f%%" % (mx, my))
        if not (30 <= mx <= 60) or not (40 <= my <= 75):
            warns.append("motion centre (%.0f%%, %.0f%%) is far from the DVS128 Gesture framing (x 36-53%%, y 53-66%%): move the camera / yourself" % (mx, my))
        tot = max(sum(counts[i] for i in range(len(counts)) if counts[i] > MIN_ACTIVE_EVENTS), 1)
        if edge / tot > 0.04:
            warns.append("%.0f%% of the events touch the border of the square crop: step back or re-centre" % (100 * edge / tot))

    sheet = None
    if len(active) >= 30:
        idx = np.linspace(active[0], active[-1], 6).astype(int)
        tiles = []
        for i in idx:
            g = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
            ev = np.zeros((128, 128, 3), np.uint8)
            for j in (i - 1, i):
                x, y, p = ev_img[max(j, 0)]
                ev[y[p == 1], x[p == 1]] = (0, 255, 0)
                ev[y[p == 0], x[p == 0]] = (0, 0, 255)
            tiles.append(np.vstack([g, ev]))
        sheet = cv2.resize(np.hstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / (Path(path).stem + ".png")), sheet)
    return warns, info, sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="work/clipcheck")
    a = ap.parse_args()
    p = Path(a.path)
    clips = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXT)
    if not clips:
        print("no video files found under", p)
        return
    bad = 0
    for c in clips:
        warns, info, _ = check(c, Path(a.out))
        print("%s  %s  [%s]" % ("PASS" if not warns else "WARN", c, "; ".join(info)))
        for w in warns:
            print("      - " + w)
        bad += bool(warns)
    print("%d clips, %d with warnings. Contact sheets (top row: frame, bottom row: events, green ON / red OFF): %s" % (len(clips), bad, a.out))


if __name__ == "__main__":
    main()
