# Event emulator synthesis results

Quartus Prime Lite 21.1.0 Build 842, device 5CSXFC6D6F31C6, project `quartus/emu_probe` (all ports virtual pins, so no IO limits),
128x128, event FIFO 1024 x 32, SDC target 125 MHz. Regenerate: `bash scripts/build_quartus.sh quartus/emu_probe emu_probe clk`.

| Item | Value |
|---|---|
| ALMs | 252 / 41,910 (< 1 %) |
| Registers | 288 |
| Block memory | 165,888 bits in 21 RAM blocks (3 %): reference 131,072 + event FIFO 32,768 + log LUT 2,048 |
| DSP blocks | 0 |
| Fmax (slow 1100 mV 85 C) | 158.13 MHz |
| Setup slack at 125 MHz (slow 85 C / slow 0 C) | +1.676 ns / +1.556 ns |
| Hold slack (worst) | +0.119 ns (fast 0 C) |

History (so the lesson is recorded): the first version computed compare, reference update and RAM write data in one stage. Slack was **-1.499 ns** at 125 MHz (Fmax about 105 MHz), critical path RAM read -> 10-bit subtract -> threshold compare -> add/sub and mux -> RAM write data (8.4 ns). Splitting compare (stage 1, registered flags) from update and event emit (stage 2) fixed it; the bit-exact regression passed unchanged.

At the demo's 25 MHz VGA/pixel clock the emulator has more than 6x timing margin; one pixel per clock gives 25 Mpixel/s, i.e. 1,500 frames/s of 128x128.

## On-chip demo (`quartus/demo`, board build with the real pins)
288 ALMs, 442 registers, 214,016 block-memory bits in 27 RAM blocks, 2 DSP blocks (the x171 reciprocal for the 3x upscale), 1 PLL, 96 pins. All STA corners positive; `CLOCK_50`, PLL `vcoph[0]` and PLL `divclk` (25 MHz) all appear in the slack tables (setup slack on `divclk` +31.6 ns slow 85 C). No Critical Warnings; the two warning types left are 292013 (Lite-edition feature notice) and 15714 (default drive strength/slew on outputs, same as the Terasic golden top).
