# Board notes (DE10-Standard, Cyclone V SoC 5CSXFC6D6F31C6)

Copied from the sister project `F:\Projects\FPGA-Adaptive-Night-Vision` (`docs/board_notes.md`, `quartus/pins/de10_nv_pins.tcl`, `quartus/bringup/`) and the CFAR project it references. Items from the CFAR project were not re-verified here. "TBD" needs the physical board.

## Toolchain on this PC (verified 2026-10-25)
- Windows 11, Quartus Prime Lite **21.1.0 Build 842** at `C:\intelFPGA_lite\21.1\quartus\bin64`, Questa FSE, Icarus Verilog 14 (devel). No Verilator, no ffmpeg (OpenCV reads mp4 on its own).
- Python 3.14, PyTorch 2.14.0+cu126, CUDA available (RTX 3050 Ti Laptop, 4 GB). Project venv `.venv` is created with `--system-site-packages` and adds opencv-python and tonic.
- Build flow script `scripts/build_quartus.sh` and `scripts/check_sta.py` are copies from the sister project (map -> fit -> sta -> asm -> compressed rbf, and a check that the clocks you care about actually appear in the STA tables).

## Device
- 5CSXFC6D6F31C6: 41,910 ALMs, 553 M10K (5.66 Mbit), 112 DSP, 15 PLLs, dual Cortex-A9 HPS with 1 GB DDR3, FPGA-side 64 MB SDRAM (`DRAM_*` pins in the golden top, the fallback frame store).

## Pins
- Reuse `F:\Projects\FPGA-Adaptive-Night-Vision\quartus\pins\de10_nv_pins.tcl` (96 assignments: CLOCK_50, KEY[3:0], SW[9:0], LEDR[9:0], HEX0-5, VGA). It is generated from the Terasic golden top (`platform_setup.tcl`, ACDS 17.0) by the sister project's `scripts/gen_pins.sh`, whose reference directory is `C:\Users\priya\OneDrive\Desktop\slop\projects\Matlab_projects\Weibull_CFAR\matlab\d10_standard_golden_top\extracted\`. Copy it to `quartus/pins/` when the first board build is created (task 0.8); do not hand-edit it.
- `CLOCK_50` PIN_AF14, `KEY` PIN_AJ4/AK4/AA14/AA15, `SW` are 2.5 V, everything else 3.3-V LVTTL.
- VGA: ADV7123 DAC, 8 bits per colour, `VGA_CLK` is the DAC clock, `VGA_BLANK_N` = active video, `VGA_SYNC_N` tied low.
- HPS pins and DDR3 come from the Platform Designer HPS component, not from the pin file.

## Boot and HPS (from the CFAR project, TBD on the board)
- Boot-mode switch `SW10 MSEL[4:0]` must be `01010` (FPPx32) for the HPS to load the FPGA image; the factory default `10010` silently boots AS mode.
- Login `root`, empty password; UART J4 (FT232R, 115200 8N1). `de10_ghrd/serial_console.ps1` in the CFAR folder works when PuTTY does not.
- OS image conflict to resolve at task 0.7: the checklist says Angstrom 2014 (Python 2.7 only, no pip), the CFAR bring-up notes say a Terasic Ubuntu 16.04 console image. Record what is on the SD card and which compilers exist. **Plan assumes C only on the HPS.**
- Proven HPS/DDR3 Qsys base: `de10_ghrd/soc_system.qsys` and `hps_common_board_info.xml` in the CFAR folder. Lightweight HPS-to-FPGA bridge at `0xFF200000`.
- `jtagconfig` on the sister project's PC showed no hardware (board not connected). TBD: record the JTAG chain and the FPGA device index `@N` here before the first `quartus_pgm`, and never use the bare default (checklist 2.4a).

## Emulator resource and timing numbers (Quartus 21.1, synthesis probe `quartus/emu_probe`, 128x128, FIFO 1024)
252 ALMs, 288 registers, 165,888 block-memory bits in 21 M10K, 0 DSP, Fmax 158 MHz (slow 85C). Details and history: `docs/results/emulator_synth.md`.

## Known gaps
- VGA output delay against the ADV7123 is not constrained in the sister SDC (same PLL clock for data and `VGA_CLK`, 40 ns period). Revisit with `set_output_delay` if the image shows edge artefacts.
- Gate every design reset on PLL `locked` (sister bring-up does `KEY[0] & pll_locked` through `reset_sync`).

## JTAG chain, measured 2026-10-28
`jtagconfig`: cable `DE-SoC [USB-1]`; device 1 = `4BA00477` (HPS DAP), device 2 = `02D020DD` (the FPGA, listed as 5CSEBA6/5CSEMA6, programs fine as 5CSXFC6D6F31). Program with `quartus_pgm -c "DE-SoC [USB-1]" -m jtag -o "p;<file>.sof@2"`. First configuration of `releases/snn_demo_build1/snn_demo.sof`: "Configuration succeeded -- 1 device(s) configured", 3 s.

## SD card changes, 2026-10-29
The card (D: on the laptop, 0.5 GB FAT32, `u-boot.scr` does `fatload soc_system.rbf; fpga load; bridge_enable_handoff; boot`) held a `soc_system.rbf` of 3,741,516 bytes (SHA-256 `d1629630...18ff`, dated 2026-10-27) that matched none of the other files on the card. It was saved as `soc_system_BEFORE_SNN_20261029.rbf` on the card and as `work/sd_backup/soc_system_as_found_20261029.rbf` on the PC (checksums equal). `soc_system.rbf` is now `releases/snn_soc_build1/DE10_Standard_SNN.rbf` (2,422,232 bytes, SHA-256 `418e93c6...12d3`). **To restore the previous design: copy `soc_system_BEFORE_SNN_20261029.rbf` over `soc_system.rbf`** (or the PC copy). Files not touched: `soc_system_CFAR_*`, the other backups, `socfpga.dtb`, `u-boot.scr`, `zImage`, `payload/`.

## First run of the SNN SoC, 2026-10-29
Board booted from the SD card with `DE10_Standard_SNN.rbf` as `soc_system.rbf`; `ssh root@192.168.137.200` (no password); `devmem 0xFF210000` = `0x534E0001`; `sw/snn_streamer.c` compiled on the board (`gcc -std=gnu99 -O2 -Wall -Wextra`) and started with `setsid ./snn_streamer > streamer.log 2>&1 < /dev/null &` (log in `/home/root/streamer.log`; it releases `fpgaportrst` on start). Results: `docs/results/board_run.md`. The daemon was left running on the board; to restore the previous design copy `soc_system_BEFORE_SNN_20261029.rbf` over `soc_system.rbf` on the card.
