"""PC-side frame feeder: video file -> 128x128 8-bit grayscale frames (roadmap 1.1, 1.2).

    python sw/frame_feeder.py clip.mp4 --raw out.y8                  # N*16384 raw bytes
    python sw/frame_feeder.py clip.mp4 --ring ring.bin --slots 64    # byte image of the DDR3 frame ring
    python sw/frame_feeder.py clip.mp4 --preview out_dir             # PNG montage of frames + events
    python sw/frame_feeder.py --synthetic bar --ring ring.bin        # no video needed

Ring slot layout (docs/frame_buffer.md), little endian, 32 KiB per slot:
    0x000  u32 magic 0x30 'FRM0' = 0x304D5246   0x004 u32 frame_counter   0x008 u64 timestamp_us
    0x010  u16 width  0x012 u16 height          0x014 u32 flags (bit0 = last frame of clip)
    0x100  pixels, width*height bytes, raster order
Slot i of the ring holds frame number k with k % slots == i; the reader verifies frame_counter
to detect dropped or repeated frames (roadmap 1.8).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import synth_video as sv  # noqa: E402
from model.video_io import read_video  # noqa: E402

SLOT_BYTES = 0x8000
PIX_OFFSET = 0x100
MAGIC = 0x304D5246


def ring_image(frames, fps, slots=64):
    """Byte image of a ring holding the first `slots` frames (frame k in slot k % slots)."""
    img = bytearray(SLOT_BYTES * slots)
    h, w = frames.shape[1:]
    n = len(frames)
    for k in range(min(n, slots)):
        base = (k % slots) * SLOT_BYTES
        ts = int(round(k * 1e6 / fps))
        img[base:base + 24] = struct.pack("<IIQHHI", MAGIC, k, ts, w, h, int(k == n - 1))
        img[base + PIX_OFFSET:base + PIX_OFFSET + w * h] = frames[k].tobytes()
    return bytes(img)


def preview(frames, out_dir, fps, theta_ln=0.2):
    import cv2
    from model.emulator_int import EmuConfig, IntEmulator
    from model.event_format import unpack
    from model.log_lut import theta_from_ln
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emu = IntEmulator(EmuConfig(theta=theta_from_ln(theta_ln), period_ticks=int(round(1e4 / fps))))
    for k, f in enumerate(frames):
        ev = unpack(emu.step(f))
        view = np.full((128, 128, 3), 128, np.uint8)       # grey = no event; ON = green, OFF = red (BGR)
        view[ev["y"][ev["pol"] == 1], ev["x"][ev["pol"] == 1]] = (0, 255, 0)
        view[ev["y"][ev["pol"] == 0], ev["x"][ev["pol"] == 0]] = (0, 0, 255)
        both = np.hstack([cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), view])
        cv2.imwrite(str(out / ("%04d.png" % k)), cv2.resize(both, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    print("wrote", len(frames), "preview frames to", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--synthetic", choices=["bar", "blob", "flash", "noise"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate (frames are dropped, never invented)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--raw")
    ap.add_argument("--ring")
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--preview")
    a = ap.parse_args()

    if a.synthetic:
        frames = getattr(sv, {"bar": "moving_bar", "blob": "moving_blob", "flash": "flash", "noise": "noise"}[a.synthetic])()
        fps = a.fps
    elif a.video:
        frames, fps = read_video(a.video, a.size, a.fps, a.max_frames)
    else:
        ap.error("give a video file or --synthetic")
    print("%d frames, %dx%d, %.2f fps" % (len(frames), frames.shape[2], frames.shape[1], fps))
    if a.raw:
        Path(a.raw).write_bytes(frames.tobytes())
    if a.ring:
        Path(a.ring).write_bytes(ring_image(frames, fps, a.slots))
    if a.preview:
        preview(frames, a.preview, fps)


if __name__ == "__main__":
    main()
