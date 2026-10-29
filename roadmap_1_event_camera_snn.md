# Roadmap 1: Event-Camera Emulator + Spiking Neural Network

**Board:** Terasic DE10-Standard (Cyclone V SoC 5CSXFC6D6F31C6: FPGA + dual-core ARM HPS, 1 GB HPS DDR3, ADV7123 VGA DAC). Pins, build flow and HPS notes come from the sister project, see `docs/board_notes.md`.
**Estimated duration:** 15-16 weeks (part-time)
**Toolchain (this PC, Windows 11):** Quartus Prime Lite 21.1 + Platform Designer, Icarus Verilog 14 and Questa FSE for simulation (no Verilator on Windows), Python 3.14 venv (`.venv`, `--system-site-packages`) with PyTorch 2.14+cu126 (RTX 3050 Ti, 4 GB), tonic, OpenCV, cocotb 2.1, pytest. Training uses a custom QAT LIF module (`train/snn_qat.py`) instead of snnTorch because the hardware neuron is custom and the forward pass must be integer-exact. **HPS software is C only** (see D7 in `docs/architecture_decisions.md`).

**How to use:** tick `- [x]` when done (or wrap the line in `~~strikethrough~~`). `[~]` means partly done, with the remainder stated. Every phase ends with an **Exit gate**. Do not start the next phase until the gate passes.

**Run the checks any time:** `.venv\Scripts\python.exe scripts\run_regression.py` (RTL benches against the Python golden models + pytest) and `... --selftest` (mutation self-test: broken RTL must fail).

---

## Status (2026-10-28, end of session 2)

**Direction change (2026-10-27):** the project is built around video that is practical to get. The real-DVS-trained network did not transfer to emulated KTH video (`docs/results/kth_domain_gap.md`: clapping 0.20-0.25, waving 0.52), so the demo network is trained on emulated KTH video (6 actions). DVS128 Gesture (Track A, 82.7% / 89.8%) stays as the real-event reference.

Done and verified here, without the board:
- Golden models, emulator RTL (bit-exact, 7 scenarios + mutation self-test), on-chip VGA demo (pixel-exact frames), QAT SNN equal to the integer model bit for bit, Track A on real DVS data: as in the sections below.
- **KTH network (6 actions, 32x32, conv 16/32, int8): 0.765 window / 0.807 sample on the validation people, 0.727 window / 0.799 sample on the test people (scored once)**. Jogging vs running is the weak pair. `docs/results/kth6_accuracy.md`.
- **SNN RTL written and bit-exact in simulation** (`rtl/snn_core.v`, `lif_conv.v`, `bin_builder.v`, `snn_ctrl.v`, `frame_sync.v`): every step's spike counts and class currents, both membrane RAMs, and the window class equal `model/snn_int.py` on a random net, a saturating net (both rails, int8 extremes, a score tie), the real KTH net on a real sample, and the whole chain pattern source -> RTL emulator -> bins -> SNN (32 frames). 205,832 clocks per step = 8.2 ms at 25 MHz. Mutation self-test on the SNN RTL. `docs/snn_arch.md`, `docs/results/snn_rtl.md`.
- **Board design `quartus/snn_demo`** (`rtl/snn_demo_top.v`): REPLAY of six real KTH clips (binned by the Python model) or LIVE pattern chain, class on HEX0 and LEDR[9:4]; top level simulated (keys, switches, 7-segment and LED outputs). Bitstream built, timing met, resources in `docs/results/snn_rtl.md`. What to do on the board: `docs/snn_bringup.md`.

**First hardware run (2026-10-28 evening): `snn_demo.sof` programmed over JTAG (cable `DE-SoC [USB-1]`, FPGA is device 2) and the user reported that it works.** The report was one line, so which clips and whether LIVE mode and the VGA picture were checked is not recorded yet.

**Video clips from a laptop (2026-10-29, RUNNING on the board: 3,600 frames bit-exact against the models, 8,000-frame soak clean, real time at 29.4 frames/s, `docs/results/board_run.md`):** `quartus/snn_soc` is the proven night-vision SoC (HPS, DDR3, bridges) with a new engine `rtl/snn_engine.v` (DMA reader -> event emulator -> bins -> SNN -> CSR results); `sw/snn_streamer.c` is the ARM daemon (derived from the one that ran on this board), `scripts/snn_viewer.py` the laptop frontend (any video or webcam), `scripts/check_board.py` compares the board with the Python models frame by frame. `tb/snn_engine_tb.v` runs real KTH frames through the engine with DMA wait states: bit-exact. How it was brought up: `docs/snn_soc.md`. Still open: power and ARM-only baseline, photographs of the display, non-KTH clips, an on-board overlay of the class. The earlier `snn_demo` (JTAG, no HPS) stays as the standalone demo.

Not done, needs the board or you: the rest of the hardware checks (all six clips against the table, LIVE mode, VGA), the HPS/Linux side, real video into the RTL emulator on the board (needs a host-written frame store: JTAG-to-Avalon bus or HPS TCP feeder, proven/planned in `F:\Projects\FPGA-Thermal-NUC_Motion`), v2e comparison on real v2e output, own phone recordings (optional now).

**Your next actions, in order**
1. Board: follow `docs/snn_bringup.md` (program `quartus/snn_demo/output_files/snn_demo.sof`; REPLAY first, then LIVE). The earlier `docs/demo_bringup.md` (VGA only) is still valid.
2. Report what the display shows for each of the six clips against the table in `docs/snn_bringup.md`.
3. Next engineering step after that: real video into the board (frame store + JTAG or Ethernet feeder), then a VGA overlay with the class name (6.3).

---

## Project definition

**One-line pitch:** Turn ordinary video files into DVS-style event streams in FPGA hardware, then classify motion or gestures with a spiking neural network (SNN) running on the same fabric.

**Target specification (as implemented; edit if needed)**

| Item | Target |
|---|---|
| Video source | Video files and datasets decoded on the PC (OpenCV), pushed to the HPS (C) and into DDR3 |
| Emulator resolution | 128x128 grayscale (start at 64x64 for bring-up if needed; `W`/`H` are RTL parameters up to 128) |
| Event format | 32-bit word `{polarity[31], y[30:24], x[23:17], timestamp[16:0]}`; timestamp tick = 100 us, one timestamp per frame |
| Emulator | log LUT (8 bit), 8-bit reference per pixel, threshold theta (9 ~ 0.2 ln units), <= 1 event/pixel/frame, "step" or "reset" reference update, backpressure not drops |
| SNN | Input 2x32x32 binary per time step; conv3x3(2->8) LIF, OR-pool, conv3x3(8->16) LIF, OR-pool, FC(1024->classes) LIF; int8 weights, 12-bit membrane, leak by shift, integer-only |
| Time step | One emulator frame (33.3 ms at 30 fps); Track A events binned to the same 33,333 us |
| Classes | DVS128 Gesture (11 classes) for real events; 4-6 own-recorded gesture classes for emulated events (use classes that exist in DVS128 Gesture) |
| Headline metrics | Accuracy, event-rate reduction vs. frame rate, latency, ALM/M10K usage, speedup vs. ARM-only |

**Two data tracks (keep them separate in your reports)**
- **Track A (real events):** DVS128 Gesture. Tests the SNN in isolation.
- **Track B (emulated events):** video file -> your hardware emulator -> SNN. This is the novel part.

---

## Phase 0: Environment and board bring-up (Week 1)

- [x] **0.1** Create the repo layout: `rtl/`, `tb/`, `model/`, `train/`, `sw/`, `quartus/`, `docs/`, `scripts/`, `data/` (gitignored).
- [x] **0.2** Quartus Prime Lite 21.1 with Cyclone V support runs from the shell (`quartus_map/fit/sta/asm/cpf` ran on this PC for `quartus/emu_probe`). Not checked: the GUI.
- [ ] **0.3** SoC FPGA EDS / preloader and device tree: do not rebuild, reuse the proven `de10_ghrd` base in the CFAR project folder (`docs/board_notes.md`) unless the HPS needs changes.
- [x] **0.4** Python venv: torch, tonic, numpy, opencv-python, cocotb, matplotlib, pytest, h5py (snnTorch replaced by `train/snn_qat.py`).
- [ ] **0.5** Verify the USB-Blaster II driver and programmer: `jtagconfig` sees the board. Record the chain and the FPGA device index `@N` in `docs/board_notes.md`.
- [ ] **0.6** Run a custom bitstream: the sister project's `quartus/bringup` (PLL + VGA test pattern + LEDs) proves board, cable and toolchain. Copy its `rtl/vga_timing.v`, `reset_sync.v`, `hex7seg.v`, `quartus/pins/de10_nv_pins.tcl` and `scripts/gen_pll.sh` rather than rewriting them.
- [ ] **0.7** Flash a prebuilt Linux image to an SD card (`SW10 MSEL = 01010` for FPGA load by the HPS). Boot, log in over UART (115200 8N1), confirm Ethernet gives you SSH. Record which OS image is on the card and which compilers exist.
- [~] **0.8** Board references: `docs/board_notes.md` written from the sister project, pin file in `quartus/pins/de10_pins.tcl`, Terasic DE10-Standard user manual kept locally in `docs/ref/` (not redistributed in the repository; download it from terasic.com). Still to collect: HPS/FPGA bridge map from the Cyclone V HPS TRM.

**Exit gate:** a custom bitstream loads, Linux boots on the HPS, and you can SSH in and copy files.

---

## Phase 1: Video input infrastructure (Weeks 2-4)

Goal: files on the PC become a clean frame stream in the FPGA, and you can see them on VGA. **Order matters: do 1.10 and 1.11 first** (on-chip path, no HPS). The DMA path (1.3-1.5) is the highest-risk item in the project.

- [x] **1.1** `sw/frame_feeder.py` (runs on the PC, not the HPS): video -> centre-crop -> 128x128 grayscale, raw output, ring-buffer image, PNG preview with the event view. Tested on synthetic and MJPG clips.
- [~] **1.2** Frame buffer layout defined in `docs/frame_buffer.md` (64-slot ring, 32 KiB slots, header with magic/counter/timestamp) and written byte-exact by `frame_feeder.py --ring`. Still to do on the board: the reserved physical region (`mem=` / reserved-memory) and its base address.
- [ ] **1.3** Build a Platform Designer system: HPS, FPGA-to-HPS SDRAM port (Avalon-MM master), lightweight HPS-to-FPGA bridge for control/status registers. Start from `de10_ghrd/soc_system.qsys`.
- [ ] **1.4** Write a **frame-reader DMA** in RTL: burst-reads one frame from DDR3 and emits the pixel stream `pix_valid/pix_ready/pix_data/pix_sop/pix_eop` that `event_emulator.v` already consumes. Checks header magic and frame counter.
- [ ] **1.5** CSR block (start, frame base address, frame count, status, error flags, emulator theta/mode/period, clear bits) and a small **C** userspace tool using `/dev/mem` (or UIO) to poke it. A C daemon on the HPS that receives frames from the PC over TCP/UDP and fills the ring replaces the "read a video on the HPS" idea.
- [ ] **1.6** **VGA output path** (640x480@60): upscale 128x128 to a window, dual-clock FIFO or double buffer between the stream clock and the pixel clock. Reuse the sister project's `vga_timing.v`.
- [ ] **1.7** Test with a known pattern (gradient, checkerboard, moving bar) pushed from the HPS. Check pixel-exact output against the source.
- [ ] **1.8** Stress test: 10,000+ frames with no dropped or duplicated frame (frame counter in the header, mismatches counted in a CSR).
- [ ] **1.9** Pacing control (fixed fps, e.g., 30 fps) so frames arrive at a deterministic rate.
- [~] **1.10** **(do first)** On-chip frame source: `rtl/pix_source.v` (procedural moving square on a gradient, 30 fps, same pixel-stream interface as the future DMA, bit-exact to `model/pattern_source.py`) is done. Still to do: a small frame ROM in M10K loaded from a `.mif` so a real clip can be played without the HPS.
- [~] **1.11** **(new, do first)** On-chip demo top `rtl/demo_top.v`: source -> `event_emulator` -> double-buffered event view (`rtl/event_view.v`) -> VGA, KEY/SW controls for theta, update mode and window, HEX/LED status. Verified in simulation (pixel-exact VGA frames); `quartus/demo` builds it with timing met. Still to do: program the board and look at the screen, following `docs/demo_bringup.md` (record any difference in `docs/bug_log.md`).

**Exit gate:** a real video clip plays on VGA through HPS -> DDR3 -> FPGA -> VGA with no frame errors.

---

## Phase 2: Golden model and datasets (Weeks 2-4, parallel with Phase 1)

- [~] **2.1** Datasets: **DVS128 Gesture downloaded** (raw IBM release in `DVSGesture/`, verified and prepared, see `docs/datasets.md`). Still to do: v2e, your own recordings, KTH/UCF101 only if needed.
- [ ] **2.2** Install v2e in a separate Python 3.10 env and run it on a few clips (`--dvs_text`, `--disable_slomo` is fine). Save the settings used (threshold, noise, bandwidth).
- [x] **2.3** `model/emulator_ref.py`: float reference (log intensity, per-pixel reference, +/- threshold, multiple events per pixel, optional interpolated timestamps and shot noise).
- [x] **2.4** `model/emulator_int.py`: **bit-accurate integer model** of the RTL (`model/log_lut.py` LUT, 8-bit reference, signed compare). Used as golden by `scripts/gen_emulator_vectors.py`.
- [x] **2.5** Timestamps: option (a), per-frame timestamp only, 100 us ticks, 17 bits (decision D3).
- [~] **2.6** `scripts/compare_vs_v2e.py` written (event counts per frame, ON fraction, event-image correlation, plot). Self-tested with a fake v2e file only. First observation on a fast synthetic clip: the one-event-per-pixel cap yields about half the events of the multi-event float reference (39.6k vs 84.4k). **Run it on real v2e output and your clips.**
- [~] **2.7** Threshold and update policy: theta 9 (~0.2 ln) with "step" provisional (D2); both policies are in RTL and verified. Decide from 2.6 with real data and record the trade-off.
- [ ] **2.8** Own gesture clips (Track B): now optional. Real-video substitute for hand clapping and hand waving: KTH (`docs/datasets.md` 2b, downloading). For the other classes, recording instructions are in `docs/recording_guide.md` and the checker is `sw/check_clip.py`; loader `train/eval_int.py --dataset own` labels by the `NN_` class-number folder prefix.
- [~] **2.9** Dataset loaders: Track A `train/aedat.py` + `train/prep_dvsgesture.py` + `train/datasets.py` **work on the real dataset** (1176 / 288 samples). Track B `train/eval_int.py --dataset own` written, untested on real clips; synthetic Track B toy set `model/toy_gestures.py` works.

**Exit gate:** the integer model reproduces v2e-like statistics, and its output is frozen as the RTL golden reference.

---

## Phase 3: SNN design, training, quantization (Weeks 4-8)

- [x] **3.1** Input representation: 128x128 -> 32x32 presence (OR) per polarity, one bin per 30 fps frame (33,333 us), identical on both tracks (D1, D5; `model/snn_input.py`).
- [x] **3.2** Architecture in `model/snn_config.py`, now size-configurable (`in_hw` 32 or 64, `c1`, `c2`, readout). **Default: 32x32 input, conv(2->16) LIF, OR-pool, conv(16->32) LIF, OR-pool, FC(2048->11) summed-current readout**: 27.4k int8 weights, 24.6k membrane words (~295 kbit), >= 72 M10K (13%) incl. the emulator. Size study (6 runs, 2 seeds for the main two) in `docs/results/trackA_accuracy.md`, decision D11.
- [x] **3.3** Float SNN on DVS128 Gesture (`--float`): **79.6% window / 87.8% sample** accuracy (60 epochs, single seed). `docs/results/trackA_accuracy.md`.
- [x] **3.4** ANN baseline (`--ann`) on the same windows: **81.9% window / 88.5% sample**.
- [x] **3.5** Hardware-friendly neuron: leak as shift (`v -= v >> 4`), integer threshold 64, reset by subtraction, saturating 12-bit membrane, no multipliers.
- [x] **3.6** Quantization-aware training (int8 STE weights, 12-bit membrane): **80.9% window / 88.2% sample**, no measurable drop against the float SNN (single seed, 288 test samples, noise about +-4 points). Needed three fixes, all recorded in D6/D10 and `docs/results/trackA_accuracy.md`: non-spiking sum readout, 1/thr surrogate slope, smaller initial weights.
- [x] **3.7** `model/snn_int.py` bit-accurate integer inference model, proven equal to the QAT forward pass (`train/tests/test_qat_vs_int.py`, per-layer spikes and counts).
- [x] **3.8** Export done: `snn_config.json` + `weights_int8.npz` from `train/train_snn.py`; ROM images `w1.hex`, `w2.hex`, `wfc.hex` in the layout the RTL reads (`scripts/gen_snn_vectors.py`, `scripts/export_kth6.py` -> `rtl/weights/kth6/`; layouts round-trip tested).
- [x] **3.9** `train/eval_int.py` on the full DVS128 test set (1377 windows, 288 samples): integer model **80.90% / 88.19%**, identical to the QAT model; ~37.7k spikes per 48-step window (layer 1 8.6%, layer 2 1.9% of neurons per step); confusion matrix in `docs/results/trackA_accuracy.md`.
- [~] **3.10** Domain-gap test: run the Track A SNN on emulator output of real video. **KTH run done (2026-10-27, `docs/results/kth_domain_gap.md`): the network does not transfer.** Test people, window accuracy: hand clapping 0.20 to 0.25 (real DVS recall 0.80), hand waving 0.52 (real 0.94 to 0.98); the best of 10 emulator settings on validation people reaches only 0.37 mean. Cause not isolated: KTH differs in scale (person small, active input 0.04 to 0.06 vs 0.166), gesture style and frame rate. Still open: own clips (`train.eval_int --dataset own`), which are framed to match DVS128.
- [~] **3.11** Not needed for the KTH demo network (it is trained on emulated events). For the DVS-trained network the gap was large (`docs/results/kth_domain_gap.md`); fine-tuning on own clips is optional now.
- [x] **3.12** Frozen 2026-10-28: 32x32 input, conv 16/32, sum readout, T = 32, trained on KTH (6 classes). `docs/snn_arch.md` has shapes, bit widths, state sizes, cycles per step.

**Exit gate:** a frozen, bit-accurate integer SNN with documented accuracy on Track A and a measured result on Track B.

---

## Phase 4: Event-emulator RTL (Weeks 5-8)

- [x] **4.1** `rtl/log_lut.v` (ROM from `rtl/log_lut.hex`, generated by `python -m model.log_lut`).
- [x] **4.2** Reference memory: `rtl/sdp_ram.v` as a standalone M10K-inferred module (checklist 1.1), same-address hazard ruled out by construction (documented in `event_emulator.v`).
- [x] **4.3** Compare and event-generate stage (signed difference, +/- threshold, polarity, step or reset update, one event per pixel per frame).
- [x] **4.4** Coordinate counters driven by `pix_sop/pix_eop/pix_valid`, with sop/eop sanity checks.
- [x] **4.5** Per-frame timestamp (`cfg_period`, restarts at 0 after `cfg_clear_ref`).
- [x] **4.6** Event packer and `rtl/event_fifo.v` with overflow pulse and counter. Backpressure on `pix_ready` makes overflow impossible by construction; the counter proves it.
- [~] **4.7** Runtime-configurable threshold, update mode and period via ports (`cfg_*`); to be wired to a CSR block in Phase 1/6. Refractory and noise-event injection not done (stretch S.4).
- [x] **4.8** Testbench `tb/event_emulator_tb.v` compares **every event word** to `emulator_int.py` over 7 scenarios (first frame, saturated pixels, zero-change frames, full-frame bursts, source gaps, stalled consumer, theta 1/255, timestamp wrap). Icarus self-checking benches instead of cocotb (D8).
- [~] **4.9** Never-silent error flags (`err_flags`, `ev_overflow_cnt`) are checked by the benches, and `run_regression.py --selftest` shows every injected fault caught. No SymbiYosys properties yet.
- [x] **4.10** Synthesis probe `quartus/emu_probe`: 252 ALMs, 288 registers, 21 M10K (165,888 bits), 0 DSP, **Fmax 158 MHz**, setup slack +1.68 ns at 125 MHz. First version missed timing by 1.5 ns (RAM read -> compare -> update -> RAM write in one stage); fixed by splitting compare and update into two pipeline stages. History in `docs/results/emulator_synth.md`.
- [~] **4.11** VGA event view (ON green, OFF red, accumulated over a 1/2/4/8-frame window, no tearing) implemented and verified in simulation (= task 1.11). Still to do on hardware: look at it, then compare visually against the v2e output of a real clip.

**Exit gate:** the RTL emulator is bit-exact to the integer model in simulation, and the live event view looks right on VGA.

---

## Phase 5: SNN core RTL (Weeks 8-12)

Start only after 3.12 (frozen architecture), otherwise the RTL chases a moving target.

- [x] **5.1** Time-stepped, one layer sweep per frame; output-stationary, 8 channels at a time (D13).
- [x] **5.2** `rtl/bin_builder.v` (two banks, swap at the end of the frame, back-pressure while clearing) + `rtl/frame_sync.v`.
- [x] **5.3** In `rtl/lif_conv.v`; saturation and reset checked on both rails (`snn_sat` case).
- [x] **5.4** `rtl/lif_conv.v`: output-stationary instead of per-input-spike updates (D13).
- [x] **5.5** Pooling by OR into the pooled word; layer 2 is the same module.
- [x] **5.6** `rtl/snn_core.v` classifier: 6 28-bit accumulators (sum readout).
- [x] **5.7** `rtl/snn_ctrl.v`: argmax, ties to the lowest index; confidence margin not implemented (the 6 window scores are available as `win_score`).
- [x] **5.8** `rtl/snn_ctrl.v`: step start, window of T steps, clear between windows, sticky overrun flag; CSR hooks come with the HPS integration (Phase 6).
- [x] **5.9** `tb/snn_core_tb.v`, `tb/snn_chain_tb.v`, `tb/snn_demo_top_tb.v`: per-step spike counts, class currents, membranes, window class.
- [~] **5.10** Bulk regression of hundreds of samples not run: each sample window takes about 1 minute in Icarus. Instead: random, saturating and real-network cases plus the chain and top-level benches, 0 mismatches, and a mutation self-test (`scripts/snn_regression.py`). A long run over real test samples is still open.
- [x] **5.11** Quartus 21.1 numbers for the whole board design in `docs/results/snn_rtl.md` (25 MHz timing met).
- [~] **5.12** Not done: cycles per step are known (205,832) from simulation, not yet exposed as hardware counters.

**Exit gate:** the full SNN RTL matches the integer model on a large regression set, and fits the device with timing met.

---

## Phase 6: System integration (Weeks 12-14)

- [ ] **6.1** Connect `frame-reader DMA -> emulator -> time-bin accumulator -> SNN -> class register` in Platform Designer.
- [ ] **6.2** Expose results (class, confidence, counters) to the HPS via the lightweight bridge. C tool prints the prediction live.
- [ ] **6.3** VGA overlay: video frame (or event view) with the predicted class text/colour bar, and a live events-per-second counter.
- [ ] **6.4** Demo mode selector using switches/keys: raw video / event view / event view + prediction.
- [ ] **6.5** Input source selector: file playback (PC stream over TCP/UDP to the HPS C daemon), dataset event replay (Track A bypasses the emulator; the replay engine inserts the bin markers).
- [ ] **6.6** End-to-end test on Track A (DVS128 replay into the SNN) and Track B (your clips through the emulator). Log accuracy.
- [ ] **6.7** Long-run soak test (1+ hour of looped playback). No hangs, no counter overflow, no FIFO overflow.
- [ ] **6.8** Fix timing closure issues for the full design (multi-clock domains, CDC on CSRs and VGA path).

**Exit gate:** a clip plays, events appear, and a class label updates on screen, repeatably, from a cold boot.

---

## Phase 7: Evaluation and benchmarking (Weeks 14-15)

- [ ] **7.1** **Sparsity:** events per second vs. pixels per second for several clips. Report the reduction factor. (Event counts per frame are already available from `emulate()`.)
- [ ] **7.2** **Accuracy table:** float SNN, quantized SNN (Python), SNN RTL (hardware), ANN baseline, on Track A; plus Track B results with and without fine-tuning.
- [ ] **7.3** **ARM baseline:** the emulator + SNN in C (NEON optional) on the HPS. Measure fps, latency per inference, and CPU load.
- [ ] **7.4** **Latency:** time from last event of a gesture window to class output in cycles and microseconds (hardware counters).
- [ ] **7.5** **Resources:** ALMs, M10K, DSP, Fmax from the final Quartus report.
- [ ] **7.6** **Power:** Quartus Power Analyzer estimate (and, if you have a USB power meter, measured board power for FPGA-on vs. ARM-only runs). State the method honestly.
- [ ] **7.7** **Ablations:** threshold sweep vs. accuracy, bit width vs. accuracy, time-bin size vs. accuracy and latency.
- [ ] **7.8** All plots in `docs/results/` with the scripts that generated them.

**Exit gate:** every headline number in your README has a script and a log behind it.

---

## Phase 8: Documentation and demo (Weeks 15-16)

- [ ] **8.1** README: problem, architecture diagram, how to build, how to run, results table, limitations.
- [ ] **8.2** Block diagram and dataflow diagram (HPS, DMA, emulator, SNN, VGA).
- [ ] **8.3** Short design doc: bit widths, memory map, CSR table, timing diagram for the event interface.
- [ ] **8.4** 2-3 minute demo video: clip -> event view -> predicted gesture, with the ARM-only comparison side by side.
- [ ] **8.5** Tag a release with the bitstream, SD card image or scripts, trained weights, and one-command run instructions.
- [ ] **8.6** One-page summary for portfolio and resume (problem, what you built, numbers).
- [ ] **8.7** Clean up: remove dead code, pin tool versions, confirm a fresh clone builds.

---

## Stretch goals (only after Phase 8)

- [ ] **S.1** Fully event-driven SNN (asynchronous neuron updates with lazy leak using timestamps).
- [ ] **S.2** Live camera input (ADV7180 composite or a D8M on GPIO) through the same emulator.
- [ ] **S.3** On-chip online learning (e.g., a simple STDP or delta rule in the output layer).
- [ ] **S.4** Per-pixel threshold variation, refractory period and noise events to mimic a real sensor more closely.
- [ ] **S.5** Interpolated per-event timestamps inside a frame (decision D3, option b) and several events per pixel per frame.

---

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| SNN accuracy too low after quantization | Widen the network a little, increase bit width, or reduce classes to 6 |
| Domain gap between emulated and real events | Train on emulated events from your own clips (Phase 3.11); the shared 33 ms bin (D1) already removes the biggest mismatch |
| One-event-per-pixel cap changes the statistics too much (seen on fast synthetic motion) | Raise the frame rate of the source clips, or add a second event per pixel per frame in RTL (cost: FIFO bandwidth) |
| 32x32 input too coarse for DVS128 Gesture | Use 64x64 input (4x the layer-1 state, still small) |
| Direction-of-motion classes (cw/ccw) and drums/guitar/"other" stay at 60-80% recall (seen: right arm ccw 0.62-0.77, drums 0.49-0.63, guitar 0.62-0.70) | Depth, regularisation, 64x64 and longer training did not help; untried: longer leak or window, an explicit temporal path, per-subject data |
| DMA or HPS bridge bring-up eats weeks | On-chip path first (1.10, 1.11); finish the emulator on hardware before the HPS work |
| HPS image has no usable compiler or Python | C cross-compile on the PC; the PC does all video decoding |
| M10K runs out | Lower resolution (64x64), fewer channels, or shorter membrane width |
| Timing closure fails | Add pipeline stages (already needed once in the emulator: compare and update are separate stages) |
