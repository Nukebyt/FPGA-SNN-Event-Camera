# ARM-only software baseline versus the FPGA engine (2026-10-29)

**Question:** how does the FPGA engine compare with doing the same job on the board's own ARM? **Answer: the ARM is faster for this small network.** This is a result, not a defect: the engine is deliberately small and was built for exactness and a fixed latency, not for speed.

## What was run
`sw/arm_baseline.c`: the whole pipeline in C, integer arithmetic only, the same weights and the same function as the FPGA: log LUT, per-pixel reference compare (step mode, theta 9), 2x32x32 presence bins, two LIF conv layers with OR pooling, classifier, window of 32 frames. Two implementations of the network so that the comparison is fair:

- **dense**: output-stationary, every neuron sums all its inputs with plain loops: the algorithm the FPGA executes;
- **sparse**: input-stationary / event driven, what one writes for a CPU: each active input bit adds its int8 weights into the accumulators of the neurons it touches (channels last, so the inner loop is contiguous and NEON-friendly).

Built on the board with its own gcc: `gcc -std=gnu99 -O3 -mcpu=cortex-a9 -mfpu=neon`. Run on the board's ARM Cortex-A9 (Linux 4.5, the board's clock frequency was not readable from `/sys`), one thread, 362 frames of a KTH boxing clip, while the FPGA daemon sat idle. Both variants were checked against the bit-accurate Python model on every frame (spike counts, event count, six class currents, window class): **362 of 362 frames identical, both modes, on the board and on a PC** (`scripts/export_arm_baseline.py compare`; the test is in `sw/tests/test_arm_baseline.py`).

## Results (per frame)
| Implementation | Event emulator | Network | Total | Frames/s |
|---|---|---|---|---|
| **FPGA engine**, 25 MHz (measured on the board: DMA start to result 222,236-222,245 clocks) | 0.66 ms (16,384 pixels at one per clock) | 8.23 ms (205,831 clocks) | **8.89 ms** | 112 (84.7 through the Ethernet path, 29.4 at the real-time pace) |
| ARM Cortex-A9, dense (same algorithm as the FPGA) | 0.28 ms | 6.15 ms | 6.43 ms (median 6.40, p99 6.53) | 156 |
| ARM Cortex-A9, event driven (sparse) | 0.29 ms | 1.14 ms | **1.43 ms** (median 1.41, p99 1.77, max 1.99) | **702** |

The ARM needs no Ethernet and no DMA in this measurement (the frames are in RAM), so the whole system with a laptop attached is limited by the network path (84.7 frames/s measured for the FPGA path); the table compares the compute only.

## Reading it honestly
- The efficient CPU implementation is **6x faster** than the FPGA engine, and even the dense CPU version is 1.4x faster. The reason is simple: this engine processes 8 output channels at a time with one weight per clock at 25 MHz and uses only about 13% of the chip (HPS included), while the Cortex-A9 runs at several hundred MHz and exploits that only 5-9% of the inputs are active.
- Levers that would change it (none was built): a faster clock (the 25 MHz clock closes with 10.8 ns of slack, i.e. about 34 MHz without pipelining work, only 1.4x), more parallel channels (8 to 32 would be 4x on the network time), an event-driven engine that skips the dense passes. These are estimates of the potential, not measurements.
- What the FPGA path gives today is a **fixed latency**: 222,236 to 222,245 clocks over 8,000 frames (9 clocks of jitter), against 1.41 to 1.99 ms of spread on the CPU, plus a CPU that stays free. For a project whose goal is to show an event-camera emulator and a spiking network in hardware that is the honest value; it is not a speed record.
- Not compared: **power and energy** (the ARM's power was not measured; the FPGA number is a vectorless estimate, `power.md`), a multi-threaded or hand-vectorised ARM version, and other clock settings of the HPS.

Reproduce: `python scripts/export_arm_baseline.py export --video <clip> --frames 362 --out work/arm`, copy `sw/arm_baseline.c` and `work/arm/{w.bin,lut.bin,frames.u8}` to the board, build, run `./arm_baseline --weights w.bin --lut lut.bin --frames frames.u8 --n 362 --mode sparse --out res.bin`, copy `res.bin` back and `compare`.
