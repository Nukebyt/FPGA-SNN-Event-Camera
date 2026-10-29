# First run of the SNN SoC on the board (2026-10-29)

Board: DE10-Standard, Linux 4.5 (2017 Terasic image) booted from the SD card, FPGA image `releases/snn_soc_build1/DE10_Standard_SNN.rbf` loaded by U-Boot as `soc_system.rbf`, Ethernet to the laptop (Windows connection sharing, board `192.168.137.200`), daemon `sw/snn_streamer.c` compiled on the board with its gcc. Nothing was programmed over JTAG.

## Bring-up checks
| Check | Result |
|---|---|
| `devmem 0xFF210000` (VERSION) | `0x534E0001`: the new image was loaded from the SD card and the lightweight bridge answers |
| STATUS `0xFF210008` / CLOCK_HZ `0xFF210028` | `0x00000001` (ready) / `0x017D7840` = 25,000,000 |
| Linux memory | System RAM `00000000-2fffffff`, so the ring at `0x30000000` is free |
| daemon start | `CSR at 0xFF210000, VERSION 0x534E0001, DDR3 ring at 0x30000000`; the FPGA-to-SDRAM port reset was released by the daemon (`fpgaportrst 0x0000 -> 0x3FFF`, as in the night-vision project) |

## Board against the Python models (`scripts/check_board.py`)
Same frames through the real board and through the bit-accurate models; compared per frame: spike counts of both layers, emulator event count, the six class currents, the running window score, the window counters and the class and scores of every finished window.

| Clips (KTH, all six actions, people 22 02 03 05 07 08 10, conditions d1-d4) | Frames | Frames that differ |
|---|---|---|
| 12 clips x 300 frames | 3,600 | **0** |
| (first run, 40-frame boxing clip, and a repeat of all 12 after the daemon fix) | 40, 3,600 | 0, 0 |

Per-frame latency from the start of the DMA to the published result: 222,236 to 222,245 clocks (8.89 ms) on the board against 222,233 in simulation. Network step 205,831 clocks (8.23 ms). Logs: `board_check_2026-10-29.log`.

## Throughput and soak
- First measurement 20 frames/s: a 40 ms delay per message from sending each reply as two small TCP writes (bug log 31). After the fix: frame round trip 11.7 ms mean (85 frames/s possible), a CSR read 1.6 ms, the frontend paced at 30 fps sustains **29.4 frames/s**.
- Soak: 8,000 frames back to back in 94.5 s (84.7 frames/s); afterwards `DONE_SEQ = IN_RD_SEQ = 8000`, 250 windows finished, STATUS `0x5` (ready, enabled), no overrun, no emulator error flag.

## What the network said on whole videos (person 22, d1, all frames, the viewer's clip rule: largest sum of window scores)
walking -> walking, jogging -> jogging, **running -> jogging**, boxing -> boxing, hand waving -> hand waving, hand clapping -> hand clapping (5 of 6; running versus jogging is the known weak pair, `docs/results/kth6_accuracy.md`). Report: `board_clip_report_2026-10-29.json`. These are single clips, not an accuracy measurement; the measured accuracy is 79.9% per clip on all test people.

## Not measured yet
Power, an ARM-only software baseline, the VGA output of this design (it has none), photographs of the HEX/LED display, clips that are not from KTH.
