# Sending video clips to the board: laptop -> Ethernet -> HPS -> DDR3 -> FPGA (`quartus/snn_soc`)

**Status: running on the board (2026-10-29).** Frames from the laptop were compared with the bit-accurate models on 3,600 frames of 12 KTH clips: 0 differ; an 8,000-frame soak ran clean; real-time at 29.4 frames/s. Measurements and the one bug the board showed (a 40 ms TCP delay, bug log 31): `docs/results/board_run.md`. Everything that was proven on hardware by the night-vision project
(`F:\Projects\FPGA-Adaptive-Night-Vision`: Linux boot from the SD card, Ethernet, `/dev/mem`, the HPS-to-FPGA bridge, DMA from DDR3) is
reused unchanged; what is new is the engine behind it.

```
laptop                                   board (DE10-Standard)
scripts/snn_viewer.py                    HPS (ARM, Linux)                     FPGA (25 MHz)
  video -> grey, squash 128x128, 30 fps    sw/snn_streamer.c  TCP 5002
  FRAME_IN (16 KiB) ------------------->  copy into DDR3 ring ------------>  snn_engine: dma_rd -> byte FIFO -> event emulator
  <-------------------- RESULT (112 B) <- read CSR snapshot  <--------------  -> 2x32x32 bins -> integer SNN -> class, scores
```

## What runs where
- **Laptop** (`scripts/snn_viewer.py`, `sw/host/snnproto.py`): any video file or webcam. Each frame is converted exactly like the
  training data (grey, whole frame squashed to 128x128 with area averaging; sources that are not 30 fps are blended to 30 fps with the
  same arithmetic as `train/kth.py`). One frame is sent, the answer is awaited, the next frame follows at 30 fps. The window shows the
  frame, the DVS events of that frame (computed on the laptop with the same bit-accurate emulator model, labelled as such), the running
  scores of the six classes and the class of the last finished 32-frame window. At the end of a clip it prints the clip class (largest
  sum of window scores).
- **HPS** (`sw/snn_streamer.c`): derived from the night-vision `nv_streamer.c`. Maps the lightweight bridge and the reserved DDR3
  (`0x30000000`, 4 slots of 16 KiB), releases the FPGA-to-SDRAM port reset as that project's daemon does (its bug 10), writes the frame
  into the next slot, increments `IN_WR_SEQ`, and sends a RESULT when `DONE_SEQ` advances. A `RESET` message restarts the engine for a new clip.
  The client can only write STATUS_CLR, THETA and SCRATCH; the daemon owns CONTROL, IN_BASE and IN_WR_SEQ so the ring cannot get out of step.
- **FPGA** (`rtl/snn_engine.v`, Platform Designer component `quartus/snn_soc/ip/snn_engine`): CSR slave at lightweight-bridge offset
  `0x10000` (physical `0xFF210000`), one 64-bit read master on the HPS FPGA-to-SDRAM port 0. A frame takes about 0.7 ms to stream into the
  emulator and 8.2 ms for the network; a new frame is only started after the previous result is final, so the network never overruns
  (latency DMA start to result: 222,233 clocks = 8.9 ms in simulation; the engine could process about 110 frames/s, the laptop sends 30).
  One network step is one frame, whatever the wall-clock arrival rate: the emulator stamps frames by count, as in training.

## CSR map (word addresses, `rtl/snn_engine.v` is the source of truth)
| word | name | | meaning |
|---|---|---|---|
| 0 | VERSION | RO | `0x534E0001` (software refuses a different build) |
| 1 | CONTROL | RW | [0] enable, [3] emulator reference update 0 step / 1 reset; writing bit 1 = soft reset pulse (everything behind the CSR block is reset, `IN_WR_SEQ` cleared), bit 2 = forget the emulator reference |
| 2 | STATUS | RO | [0] ready (builder swept, network cleared), [1] busy, [2] enable, [3] overrun (a frame arrived while the network was busy), [4] emulator error flag seen, [11:8] emulator error flags |
| 3 | STATUS_CLR | W | clears the sticky bits |
| 4 | IN_BASE | RW | physical byte address of the ring (slot s at +s*0x4000) |
| 6 | IN_WR_SEQ | RW | frames written by software |
| 7 | IN_RD_SEQ | RO | frames whose pixels the FPGA has consumed |
| 8 | DONE_SEQ | RO | frames fully processed; all result registers below belong to frame `DONE_SEQ` |
| 9 | THETA | RW | emulator threshold in LUT steps, default 9 (0.2 ln units), the value the network was trained with |
| 10 | CLOCK_HZ | RO | 25,000,000 |
| 11 | SCRATCH | RW | echo test |
| 12 | WINDOW_T | RO | 32 |
| 13, 14 | STEP_CYC, LAT_CYC | RO | clocks of the last network step (about 205,832) and from DMA start to result (about 222,233) |
| 16 | PRED | RO | class of the last finished window (valid once WINDOWS >= 1) |
| 17, 18 | WINDOWS, WCOUNT | RO | finished windows, steps done in the current window |
| 19, 20, 21 | SPK1, SPK2, EVENTS | RO | spikes of layer 1 and 2 in this frame, emulator events in this frame |
| 24..29 | WIN_SCORE[0..5] | RO | scores of the last finished window (signed) |
| 32..37 | CUR[0..5] | RO | class currents of this frame |
| 40..45 | SCORE[0..5] | RO | running score of the current window including this frame |

## Messages (16-byte header `magic 'SNNE', version 1, type, seq, length`, little endian; `docs` of the night-vision project describes the same framing)
HELLO (128, 128) -> HELLO_ACK (status, build id), FRAME_IN (16384 bytes), RESULT (28 words: done_seq, pred, windows, wcount, spk1, spk2,
events, step_cycles, latency_cycles, status, then 6 class currents, 6 running scores, 6 scores of the last finished window), RESET,
CSR_WRITE / CSR_READ / CSR_REPLY, ERROR (1 bad magic, 2 bad version, 3 bad size, 5 no memory, 6 busy, 7 FPGA timeout).

## Build (done on this PC)
`python scripts/make_snn_soc.py` derives `quartus/snn_soc` from the night-vision SoC project (the HPS, DDR3, bridges and clocking are
copied unchanged; only the engine instance and the top-level status wiring differ), then `bash scripts/build_snn_soc.sh --generate`
(Qsys generate, its known DDR3-sequencer crash on this machine is repaired by `scripts/patch_qip.py`, map, merge, pin assignments, fit,
sta, assembler, compressed `.rbf`, STA check including the DDR3 clocks). Results: `docs/results/snn_soc.md`.

## Bring-up on the board (in this order; stop at the first failure and write it in `docs/bug_log.md`)
1. **SD card.** Keep the night-vision / CFAR files. On the FAT partition copy the current `soc_system.rbf` to `soc_system_nightvision_BACKUP.rbf`
   (do not overwrite an existing backup), then copy `releases/snn_soc_build1/DE10_Standard_SNN.rbf` to the card as `soc_system.rbf`.
   Boot switch `SW10 MSEL = 01010`. Ethernet cable to the laptop (Windows connection sharing, the board is `192.168.137.200`).
   The board must be powered from the cold: do not program anything over JTAG afterwards (the preloader sets up the bridges for the image it loads).
2. **Linux is up:** `ping 192.168.137.200`, `ssh root@192.168.137.200`. `LEDR[0]` blinks (HPS led_pio heartbeat from the golden top).
3. **Copy and build the daemon on the board** (it has gcc): `scp sw/snn_streamer.c root@192.168.137.200:` then on the board
   `gcc -std=gnu99 -O2 -Wall -Wextra -o snn_streamer snn_streamer.c`.
4. **Run it:** `./snn_streamer`. It must print `CSR at 0xFF210000, VERSION 0x534E0001`, the `fpgaportrst` line and `listening on port 5002`.
   VERSION 0 or a failed scratch echo means the wrong image is loaded or the bridge is not enabled (check step 1).
5. **Bit-exact check against the Python models, first test of everything:**
   `.venv\Scripts\python.exe scripts\check_board.py --host 192.168.137.200 --source data\kth\videos\person22_boxing_d1_uncomp.avi --frames 200`
   must end with `RESULT: PASS` (every spike count, event count, current and window score identical to the model). If it fails it prints
   the first differing frame and fields. Then try other clips, e.g. a walking and a hand-clapping video.
6. **Look at it:** `.venv\Scripts\python.exe scripts\snn_viewer.py --source <clip> --connect 192.168.137.200`.
   HEX0 shows the class of the last finished window, HEX3..HEX2 the steps in the current window, HEX5..HEX4 the frames processed;
   `LEDR[1]` = ready, `LEDR[2]` = enabled, `LEDR[3]` = error.

## What can go wrong first (from the night-vision experience and from what is new)
- The ring must not be Linux memory: the device tree limits Linux to 768 MB, so `0x30000000` is free (measured there). `cat /proc/iomem` on the board.
- The FPGA-to-SDRAM port is in reset after the preloader: the daemon releases it (and prints the register); if reads return nothing this is the first suspect.
- The Quartus-generated image must be the one on the card: `VERSION` and `build id` (HELLO_ACK) are the check.
- Both items that were unproven before the first run turned out fine: the engine's DMA reads from the real DDR3 return the right data, and the DMA-to-result latency equals the simulation to within 12 clocks (`docs/results/board_run.md`).

## Limits (what "detects" means)
The network knows six actions (walking, jogging, running, boxing, hand waving, hand clapping) and always answers with one of them: there is no "none of these" class.
It was trained on KTH video: static camera, a whole person in a 160x120 picture, plain backgrounds, and scores 79.9% per clip (72.7% per 1-second window) on
unseen KTH people (`docs/results/kth6_accuracy.md`). Clips unlike that (a moving camera, a person filling the frame, a busy background, a different distance) are outside
what was measured and will do worse; jogging versus running is the weakest pair. The first thing to try is a KTH video; your own clips are the experiment.
