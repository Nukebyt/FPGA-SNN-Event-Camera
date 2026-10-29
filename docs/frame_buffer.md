# Frame buffer layout in HPS DDR3 (roadmap 1.2)

Reserved physical region (kernel `mem=` or a device-tree reserved-memory node, base address TBD on the board), a ring of 64 slots, 32 KiB per slot (2 MiB total). Little endian.

| Offset in slot | Size | Field |
|---|---|---|
| 0x000 | u32 | magic `0x304D5246` (bytes `F R M 0`) |
| 0x004 | u32 | frame counter, increments by one per frame, never repeats |
| 0x008 | u64 | timestamp in microseconds since the start of the clip |
| 0x010 | u16 | width |
| 0x012 | u16 | height |
| 0x014 | u32 | flags: bit 0 = last frame of the clip |
| 0x100 | w*h bytes | 8-bit grayscale pixels, raster order |

Frame k lives in slot `k % 64`. The producer (HPS) writes pixels first and the header last, then bumps a write-index CSR; the frame-reader DMA reads the header first and checks magic and `frame counter == expected`. A mismatch increments an error counter in the CSR (dropped or repeated frame, roadmap 1.8) and is never silent.

`python sw/frame_feeder.py clip.mp4 --ring ring.bin` writes the byte image of the first 64 frames in exactly this layout, so the layout can be tested on the PC before the board exists.
