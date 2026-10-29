# Power: Quartus estimates (2026-10-29). Nothing was measured on the board.

## Method
Quartus Prime Lite 21.1 Power Analyzer (`quartus_pow`) on the fitted designs, vectorless: default toggle rate 12.5% on all signals and on the inputs, no simulation activity file. The tool itself rates the estimate **"Low: user provided insufficient toggle rate data"**. The HPS (the ARM cores and their DDR3 controller) is not part of the number (the report lists "Enable HPS: Off"); only the FPGA fabric, its I/O, PLL and the HPS-to-FPGA interfaces are.

| Design | Total | Core static | Core dynamic | I/O | Notes |
|---|---|---|---|---|---|
| `snn_demo` (JTAG, pattern source / clip replay, VGA, no HPS) | 481 mW | 417 mW | 35.9 mW | 27.9 mW | no HPS in this design |
| `snn_soc` (HPS bridges, DMA, emulator, network) | **531 mW** | 416 mW | 65.5 mW | 49.0 mW | 25 MHz domain: 33.4 mW dynamic; DDR3 DLL 10.5 mW, DDR I/O 17 mW, M10K blocks 10.5 mW |

## What can and cannot be said
- **78-87% of the estimate is device static power** that the design does not control; the network's own dynamic power is a few tens of mW at these toggle assumptions. The real activity is lower than 12.5% for most signals (about 5-9% of layer-1 neurons spike per frame), but the estimate was not refined with simulated activity.
- **No comparison with the ARM is possible** from these numbers: the ARM's power was not measured, and the HPS is excluded here.
- Energy per frame, FPGA, as an upper-bound style estimate: 65.5 mW dynamic x 8.9 ms = 0.58 mJ if the whole core dynamic power were attributed to the engine for the whole frame time (it is not; the DDR3 interface and idle logic are in that number). It is quoted only to show the order of magnitude and must not be used to claim an advantage.

## To measure it properly (needs a USB power meter or a bench supply with a current display)
Put the meter between the 12 V supply and the board and write down (1) the board booted and idle (daemon running, no stream), (2) streaming a clip at 30 frames/s with `scripts/snn_viewer.py`, (3) the same with the ARM baseline running instead (`sw/arm_baseline.c` in a loop) and the FPGA idle. The differences (2)-(1) and (3)-(1) are the extra power of the FPGA path and of the ARM path; with a 12 V supply read the current to 1 mA.
