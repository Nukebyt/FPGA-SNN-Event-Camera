"""Phone video of the live demo (laptop page + DE10 board) -> README GIF and a still image.

    python scripts/make_demo_gif.py --src data/user_clips/demo_video.mp4

Writes docs/media/live_demo_board.gif (width --width, --fps frames/s, one shared --colors palette so the GIF stays small and does
not flicker; the bottom --crop-bottom of the picture, the laptop keyboard, is cut) and docs/media/live_demo_board.jpg (one frame).
The phone video itself is not copied into the repository (16 MB; re-encoding with OpenCV's mp4v made it larger).
"""
import argparse
import os

import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "user_clips", "demo_video.mp4"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "media"))
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--crop-bottom", type=float, default=0.2)
    ap.add_argument("--still", type=float, default=5.0, help="time of the still image, seconds")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, still = [], None
    step, k, nxt = src_fps / a.fps, 0, 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = f[:int(round(f.shape[0] * (1 - a.crop_bottom)))]
        if still is None and k >= a.still * src_fps:
            still = f.copy()
        if k >= nxt:
            h2 = int(round(a.width * f.shape[0] / f.shape[1]))
            g = cv2.resize(f, (a.width, h2), interpolation=cv2.INTER_AREA)
            g = cv2.bilateralFilter(g, 5, 20, 5)                     # phone sensor noise costs a lot of GIF bytes
            frames.append(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)))
            nxt += step
        k += 1
    cap.release()
    # one palette for the whole GIF, from a strip of sampled frames
    sample = frames[:: max(1, len(frames) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, s in enumerate(sample):
        strip.paste(s, (0, i * s.height))
    pal = strip.quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    gif = os.path.join(a.out, "live_demo_board.gif")
    q[0].save(gif, save_all=True, append_images=q[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    cv2.imwrite(os.path.join(a.out, "live_demo_board.jpg"), cv2.resize(still, (1280, int(1280 * still.shape[0] / still.shape[1]))),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    for p in (gif, os.path.join(a.out, "live_demo_board.jpg")):
        print("%-60s %6.2f MB" % (os.path.relpath(p, ROOT), os.path.getsize(p) / 1e6))
    print("%d GIF frames at %.0f fps (%.1f s)" % (len(frames), a.fps, len(frames) / a.fps))


if __name__ == "__main__":
    main()
