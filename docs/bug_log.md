# Bug log

Format (checklist 7.3, same as the sister project): **Symptom / Root cause / Fix / Lesson**, exact numbers, wrong hypotheses kept. Newest entries at the bottom. Dates are 2026-10-25 / 26 unless stated. **Every new problem gets an entry here when it is found, not when it is fixed.**

Status key: fixed = verified fixed; open = still to resolve.

---

### 1. Emulator missed timing by 1.5 ns at 125 MHz (RTL, 10-25) - fixed
**Symptom:** first Quartus probe (`quartus/emu_probe`): setup slack **-1.499 ns** (slow 85C), TNS -338.6, Fmax about 105 MHz against a 125 MHz target.
**Root cause:** worst path (from `report_timing`): reference-RAM output register -> 10-bit subtract -> `>= theta` compare -> add/sub and mux -> reference-RAM write data, 8.42 ns of data delay, all in one pipeline stage.
**Fix:** split into stage 1 (subtract and compare, flags registered) and stage 2 (reference update and event emit). Bit-exact regression passed unchanged. Result: slack **+1.676 ns**, Fmax **158 MHz**.
**Lesson:** read the path report before guessing; a RAM-output-to-RAM-input loop with arithmetic in between always needs a register in the middle. Pipelining changed the in-flight depth, so the FIFO headroom (8 entries) was re-checked by the mutation self-test.

### 2. Quartus syntax error 10170 on the probe wrapper (RTL tooling, 10-25) - fixed
**Symptom:** `quartus_map` failed: `Verilog HDL syntax error at emu_probe_top.v(12) near text "*)"`.
**Root cause:** the wrapper used the SystemVerilog implicit port connection `.*` but was a `.v` file, so Quartus parsed it as plain Verilog.
**Fix:** renamed to `.sv` and declared it with `SYSTEMVERILOG_FILE`.
**Lesson:** file extension selects the language in Quartus; SystemVerilog-only syntax needs `.sv`.

### 3. `inside` is a SystemVerilog keyword (RTL, 10-25) - fixed
**Symptom:** `iverilog -g2012`: `pix_source.v:30: syntax error / Invalid module item` at `wire inside = ...`.
**Root cause:** I named a signal `inside`, which is reserved in SystemVerilog (the benches compile with `-g2012`).
**Fix:** renamed to `in_sq`.
**Lesson:** compile RTL under the strictest language mode that any tool uses, not only the one the synthesiser needs.

### 4. Latch warning 10240 on `sy[0]` in the pattern source (RTL, 10-25) - fixed (harmless, silenced)
**Symptom:** `Inferred latch for "sy[0]"` in the demo build, the only 10240 warning.
**Root cause:** `sy` advances by 2 from 0, so bit 0 is a constant 0; the synthesiser saw a register that only holds its value and reported it as a latch. No real latch in hardware (also `sx[0]` was merged with `sy[1]`, as expected).
**Fix:** `sy_h[5:0]` counts by 1 and `sy = {sy_h, 1'b0}`; warning gone (0 of 10240).
**Lesson:** keep the warning list clean so that a real latch warning would stand out.

### 5. VGA frame loader read 345,600 values instead of 307,200 (verification, 10-25) - fixed
**Symptom:** `scripts/check_demo_vga.py`: `AssertionError: 345600`.
**Root cause:** Icarus `$writememh` inserts a `// 0x...` address comment every 8 words (38,400 extra tokens); my loader split on whitespace and counted them.
**Fix:** loader skips tokens starting with `//`, `0x` or `@`.
**Lesson:** look at the first lines of a tool-written file before writing its parser.

### 6. Hardware-model frame ring image was 96 bytes too large (software, 10-25) - fixed
**Symptom:** `ring.bin` was 2,097,248 bytes instead of 2,097,152 (64 slots x 32 KiB).
**Root cause:** the slot header packs to 24 bytes (`<IIQHHI`) but I assigned it into a 20-byte slice, so the `bytearray` grew by 4 bytes per frame (24 frames = 96 bytes).
**Fix:** slice `[base:base + 24]`. Checked: size 2,097,152, magic `0x304D5246`, counter and timestamp read back for slot 5.
**Lesson:** assert the size of a binary image after building it; a silently growing buffer shifts every later slot.

### 7. Regression script died with a SyntaxError after an edit (tooling, 10-25) - fixed
**Symptom:** `run_regression.py`: `SyntaxError: unterminated string literal` at `"COMPILE ERROR`.
**Root cause:** I patched the script from a Python heredoc; the `\n` inside my replacement text became a real newline in the file.
**Fix:** fixed the literal by hand; `ast.parse` check added to my routine after scripted edits.
**Lesson:** after any scripted edit of a source file, parse or run it before moving on.

### 8. Stale or non-matching mutation patterns in the self-test (verification, 10-25) - fixed
**Symptom:** `--selftest` printed `SELFTEST ERROR: pattern not found` twice (after the pipeline split renamed `ref_q`/`s1_*` to `s2_*`; and for `wbank <= ~wbank;` whose real text has extra spaces) and ended `OVERALL: FAIL`.
**Root cause:** mutations are plain text replacements and the RTL text had changed or was spaced differently.
**Fix:** patterns updated; the harness correctly refused to count a mutation it could not apply. All 12 real mutations are now caught; the one mutation that only halves throughput (still functionally correct) is correctly reported as MISSED and expected so.
**Lesson:** a self-test that fails loudly on its own stale data is the behaviour we want (sister project bug 2 / checklist 3.9); update mutations in the same change as the RTL.

### 9. v2e comparison: int model showed 0.004 correlation (analysis script, 10-25) - fixed
**Symptom:** `compare_vs_v2e.py` on a fake v2e file: event-image correlation int model vs "v2e" **0.004**, float reference vs "v2e" 0.974.
**Root cause:** frame-index binning. Events generated by frame k lie in ((k-1)dt, k dt]; I binned with `floor(t/dt)`, which put them in frame k-1, while the integer model reports them at frame k. The float and the fake-v2e data shared the same off-by-one, which hid it.
**Fix:** `ceil(t/dt - 1e-9)`. Correlation int vs float-based file became **0.591** (the remaining gap is the one-event-per-pixel cap, see roadmap 2.6).
**Lesson:** when two things agree with each other but not with a third, check the alignment convention before blaming the model. Still to confirm with real v2e output.

### 10. Toy network stuck at 50% accuracy (training, 10-25) - fixed
**Symptom:** QAT SNN on the 6-class toy set plateaued at exactly 0.500 train and test (3 of 6 classes) for 50 epochs.
**Root cause:** boxcar surrogate gradient `max(0, 1 - |v - thr| / thr)` is exactly zero for a neuron resting at v = 0, so silent neurons never received a learning signal.
**Fix:** fast-sigmoid shape with non-zero tails. Toy test accuracy 100%.
**Lesson:** a surrogate must have a gradient where the neuron actually sits at rest.

### 11. DVS128 Gesture: network at chance, loss above ln 11 (training, 10-26) - fixed (three causes)
**Symptom:** first real-data runs: train and test 8-9% (chance 9.1%), loss 3.3 to 6.0; the toy-set recipe had worked.
**Wrong hypotheses, in the order tested:** (a) initial weights too large for dense input - smaller std did not help alone; (b) learning rate / logit scale - changed behaviour but not the failure; (c) weight resolution too coarse for about 500 active FC inputs, so raise thresholds to (128,256,512) or (256,512,1024) with larger weights - output layer died instead (firing 0.000-0.007).
**Root causes (measured with a per-iteration probe, `work/diag.py`):**
1. Gradient explosion: the surrogate slope was about 1 per membrane unit but membranes are integers with thr 64, so the backward signal grew by about |w| per layer and per step. Layer-1 weight gradient 2.0e6 against 4.5e1 at the output (first iteration).
2. Spiking output layer with count readout: with random weights each output neuron has a fixed sign, so it fires on every step or never (out firing 0.82-0.89 or 0.000); the loss pushes everything to the dead state where the gradient vanishes.
3. Initial weights tuned on the toy set (4% input density) saturated layer 2 at 39% firing on DVS input (17% density).
**Fix:** surrogate slope scaled by 1/thr (gradients 7.5 / 2.6 / 0.7 across layers); non-spiking sum readout (decision D10); calibrated initial weights (`--init-std auto`). 250 iterations later: 56-70% training accuracy. Final: 80.9% window / 88.2% sample (8/16 channels).
**Lesson:** instrument first (per-layer firing and gradient magnitude per iteration), then change one thing at a time; two of my three first hypotheses were wrong.

### 12. Exported toy weights no longer matched their config (software, 10-26) - fixed
**Symptom:** after adding the `readout` field, `train/runs/toy/snn_config.json` had no `readout` key, so `SNNConfig.from_json` silently defaulted it to "sum" for weights that had been trained with the spike readout.
**Root cause:** a new config field with a default silently changes the meaning of old exports.
**Fix:** retrained the toy export with the sum readout (still 100%). `from_json` now also ignores unknown keys (older exports contain `down`).
**Lesson:** exports should carry a format version; for now, retrain or re-export whenever a config default changes meaning.

### 13. Download landed in a different place and format than planned (data, 10-26) - fixed
**Symptom:** `data/` was empty; the dataset was in `DVSGesture/` at the project root, and it was the raw IBM release (122 `.aedat` + csv + trial lists), not tonic's preprocessed copy my prep script assumed.
**Root cause:** the IBM page offers the raw release. My docs recommended tonic's figshare route (which I could not verify: HTTP 202, no size) and listed the IBM Box download only as a fallback, noting that tonic would expect its own folder layout; no code existed for the raw format.
**Fix:** wrote an AEDAT 3.1 parser (`train/aedat.py`), made raw the default in `prep_dvsgesture.py`, added `DVSGesture/` to `.gitignore`. Verified: coordinates 0..127, monotonic timestamps, ON fraction 49.8%, 1176 train / 288 test samples (matches SpikingJelly and snnTorch counts), no mirroring (right-hand gestures at x about 40).
**Lesson:** verify the data's format and layout before building on assumptions about it; check orientation empirically.

### 14. Documentation claims that were wrong and were caught on re-check (docs, 10-25 / 10-26) - fixed
- `docs/demo_bringup.md` said the pattern's LUT values are 243 and edge contrast 46-72; real values: LUT[200] = 244, contrast 47-73 steps (computed, then corrected).
- `docs/results/trackA_accuracy.md` first said "hand clapping, both waves and arm roll are 90% or better"; computed per-class recall: clapping 84%. Also a garbled confusion sentence replaced by the actual counts.
- The network size table listed layer-1 spike rates "~700 / ~900" for runs where I had not measured them; replaced with "not measured".
- `--init-std` help text pointed at a `train/check_init.py` that never existed; replaced by `--init-std auto`.
**Lesson:** a number in a doc needs a command behind it. Compute, do not remember.

### 15. Environment notes (tooling, 10-25 / 10-26) - not bugs, kept so they cost no time again
- `python -I` (isolated mode) cannot import numpy from the `--system-site-packages` venv; run normally from the repo root (the dataset folder contains no `.py` files, so there is nothing to shadow).
- `tail -n N file1 file2` needs `-n`; the legacy `tail -4 a b` form fails in Git Bash.
- Quartus fitter on the 260-ALM emulator probe took about 1 h 45 min while a PyTorch training job and simulations were running at the same time, and about 12 min for the 288-ALM demo on a quiet machine. Advanced Physical Optimization is slow when the CPU is shared; run Quartus builds in the background and do not start them together with training.
- Training three 64x64 runs at once took 70-80 s per epoch each (about 20 s for one 32x32 run): the GPU was launch-bound, so parallel runs help much less than the core count suggests.

### 16. One of five parallel training runs crashed at startup with CUDA out of memory (tooling, 10-26) - fixed
**Symptom:** `r1_flip` printed only its config, then `torch.AcceleratorError: CUDA error: out of memory` in the init-calibration step; the other four runs were fine.
**Root cause:** five processes started at the same moment on a 4 GB laptop GPU. Each needs a CUDA context plus the calibration batch at the same time, and the sum of the start-up peaks exceeded 4 GB. Steady state for all five is only about 2.4 GB.
**Fix:** relaunched the failed run after the others had settled (no code change).
**Lesson:** stagger launches on a small GPU (or limit to 3-4 jobs); a missing first log line is the sign, so check every log after launching a batch.

### 17. Synaptic-work estimates were inconsistent between configurations (analysis script, 10-26) - fixed
**Symptom:** after re-reading `scripts/snn_resources.py` output for the third-conv comparison, the work figures I had quoted for the size study (0.9M / 0.22M / 0.06M synaptic additions per step for 64x64 16/32, 32x32 16/32 and 32x32 8/16) did not agree with the measured spike counts.
**Root cause:** the script's default rates were one fixed set (input density 0.07, L1 0.086, L2 0.019): the input density is the 64x64 value (32x32 input is 16.6% dense) and the L1/L2 rates belong to the 8/16 net, so every other configuration was estimated with the wrong rates.
**Fix:** the script now has a table of firing rates measured with `train.eval_int` per configuration (32x32 8/16, 32x32 16/32, 64x64 16/32) and falls back to the 32x32 16/32 rates with the input density chosen by size. Corrected figures: about 0.65M / 0.18M / 0.08M. The qualitative conclusion (64x64 costs about 3.5x the work of 32x32 16/32) is unchanged.
**Lesson:** derived numbers that I quote in prose need the same discipline as measured ones: regenerate them from the script after any change to its inputs, and grep the docs for the old value.

### 18. Mirror augmentation lowered accuracy by 4.6 points; was my class mapping wrong? (training, 10-27) - fixed (not a bug in the mapping)
**Symptom:** training with 50% left/right mirroring (right wave <-> left wave, right arm cw <-> left arm ccw, right arm ccw <-> left arm cw) gave 78.2% window / 85.1% sample against 82.8% / 90.3% without it, and the train accuracy fell too (88% against 93%).
**Wrong hypothesis tested:** the class mapping is wrong. Check: baseline model (no augmentation) scored on mirrored test windows: **64.5% with the mapped labels against 28.2% with unmapped labels**; the wave classes keep 97% / 92% recall when mirrored, so the mapping is right.
**Root cause (probable, not verified):** mirrored arm-circling windows are recognised 20-30 points worse than real ones (right arm cw 0.78 -> 0.61, left arm ccw 0.93 -> 0.47; drums 0.51 -> 0.38, guitar 0.68 -> 0.28), i.e. real left-arm and right-arm gestures in this dataset are not mirror images of each other (handedness, how the 29 subjects perform them). Mirroring then adds mislabelled-looking examples instead of new valid ones.
**Fix:** none needed in code; mirror augmentation stays available (`--aug-flip`) but is not used. Result recorded in `docs/results/trackA_accuracy.md`.
**Lesson:** before concluding that an augmentation is bad, test the invariance it assumes directly on a trained model.

### 19. Own-clip loader would have labelled clips by folder rank, not by DVS class (software, 10-27) - fixed before first use
**Symptom:** found while writing the recording instructions: `train/eval_int.py --dataset own` took `sorted(folder names)` and used the position as the label.
**Root cause:** correct only if all 11 classes are recorded. With the 6 core classes (DVS ids 0, 1, 2, 5, 6, 7) the folders would have been labelled 0-5, so a perfect classifier would have scored about 17%; no error would have been raised. (The earlier note in `docs/datasets.md` only warned about this instead of preventing it.)
**Fix:** class folders must be named `NN_name` where NN is the DVS128 Gesture class number 00-10; the loader parses it and refuses other names. It also trims the still seconds before and after a gesture. Covered by `train/tests/test_own_clips.py`.
**Lesson:** when a doc has to warn about a trap, remove the trap instead.

### 20. Clip checker reported flicker on every clip and used an unusable fixed activity threshold (tooling, 10-27) - fixed before use
**Symptom:** first test of `sw/check_clip.py` on synthetic clips: the clean clip and all bad ones showed "background flickers 6.6% frame to frame", and "motion present in only 41% of the active span" although the synthetic hand moved continuously.
**Root causes:** (1) the flicker metric subtracted a moving average computed with `np.convolve(..., "same")`, which zero-pads, so the first and last 7 frames had a huge residual (constant 6.65% for any clip); (2) the active-frame threshold was a fixed 150 events per frame, but a soft-edged moving object produced fewer events than that on many frames, and a real phone clip will have a different event level again.
**Fix:** flicker is measured on the interior frames only (clean clip now 0.01%, a +-5% alternating-brightness clip 5.35%); the active threshold adapts to the clip (20% of its 90th-percentile event count, floor 30). After the fix the clean synthetic clip passes and each of seven deliberately bad clips (flicker, dark, drift, 25 fps, off-centre, pauses, too short) is flagged for the right reason.
**Open:** all thresholds are tuned on synthetic video. First real phone clips may need recalibration (the guide asks for two test clips and their checker output before the full recording).
**Lesson:** test a checker with a known-good and a set of known-bad inputs before handing it over; the known-good case is what exposes false alarms.

### 21. Dataset doc pointed at a KTH address that no longer serves the data (docs, 10-27) - fixed
**Symptom:** `docs/datasets.md` listed KTH at `http://www.nada.kth.se/cvap/actions/` with a "not verified, may have moved" note and described a 1 GB third-party mirror.
**Root cause:** I took the address from search results without testing it. Tested now: `nada.kth.se/.../handclapping.zip` redirects to an HTML page (19 KB), while `https://www.csc.kth.se/cvap/actions/handclapping.zip` returns the real zip (176,901,831 bytes); sizes of all six per-action zips are on the dataset page.
**Fix:** doc rewritten with the verified URLs, sizes, terms and the real condition list (the dataset page says d4 is "lighting variations"; my earlier note said "indoors").
**Lesson:** HEAD-request every download link before putting it in a doc; "unverified" in a doc means the work is not finished.

### 22. Draft junk left in the new KTH evaluator (code, 10-27) - fixed before running on real data
**Symptom:** reading `train/eval_kth.py` after the synthetic smoke test: three leftover lines at the end of `main()` (a loop that computed nothing and an `owner_action` expression guarded by `if False`) from an abandoned first idea for the per-action prediction summary; a walrus assignment inside a call was also unnecessary.
**Root cause:** wrote the summary code twice in one edit and kept the first attempt.
**Fix:** replaced by the real predicted-class distribution per action; syntax and smoke test rerun.
**Lesson:** reread new scripts once before running them on the data that matters.

### 23. A dropped download looked like a finished one and the chained job ran on nothing (tooling, 10-27) - fixed
**Symptom:** the background job "wait for download, extract, evaluate" ended with `FileNotFoundError: data\kth\00sequences.txt`; `work/kth_download.log` held `curl: (56) Recv failure: Connection was reset`; `handclapping.zip` was 65,281,481 of 176,901,831 bytes; no video was extracted.
**Root cause:** (1) the KTH server resets long connections (measured speed only about 115 KB/s); (2) my wait condition was `DOWNLOAD_DONE or any line containing "curl:"`, so an error ended the wait as if it were success, and the following steps (written with `&&` and `;` mixed) partly ran anyway.
**Fix:** resumable download (`curl -C -`) in a retry loop that succeeds only when the file size equals the `Content-Length` the server reported (176,901,831 and 218,981,954 bytes); extraction and evaluation start only after the explicit `FETCH_COMPLETE` line. No data was corrupted: the partial file is simply resumed.
**Lesson:** a wait condition must wait for success, not for "something was printed"; verify sizes against the server before using a download.

### 24. Two downloaders appended to the same file (tooling, 10-27) - handled, verified at the end
**Symptom:** after starting the parallel fetch of the four remaining KTH zips, `pkill` printed "command not found" (Git Bash on Windows has none), so the earlier sequential fetch script kept running and two `curl -C -` processes were both appending to `boxing.zip`.
**Root cause:** I assumed `pkill` existed and did not check the process list before starting the second script.
**Fix:** listed processes with PowerShell (`Get-CimInstance Win32_Process`), stopped the old script and its curl. Two appenders may have duplicated bytes in `boxing.zip`; the size check against Content-Length and `zipfile.testzip` before extraction will catch that, and the file is re-fetched from scratch if either fails.
**Lesson:** list what is running before starting a second copy of anything that writes to the same file.

### 25. `boxing.zip` had the right size but 32 corrupt videos (tooling, 10-27) - repaired by range requests
**Symptom:** after the download reached the server's size (194,498,294 bytes) the pipeline's `zipfile.testzip()` raised `zlib.error: too many length or distance symbols` and stopped before extracting. Opening each member showed 32 of 100 videos bad (`person01_boxing_d2` to `person09`), all in one block, bytes 1,605,772 to 59,916,058; the other 68 were fine.
**Root cause:** bug 24 was real. Overlapping `curl -C -` writers (the old sequential script plus the parallel one, then several restarts) wrote the first ~60 MB of the file out of order; the size check alone could not see it because the total length still matched.
**Fix:** a copy of the bad file is kept (`work/boxing_corrupt_backup.zip`); only the bad byte range is re-fetched with HTTP `Range` requests (1 MB chunks, retried, status 206 required) and written into the file in place, then the whole zip is tested again. The pipeline starts only if `testzip` passes.
**Lesson:** a size match proves nothing about content when more than one process wrote the file. The CRC test before use is what caught it; keep it. One writer per file.

### 26. Scripted edits turned `\n` into real newlines again, twice (tooling, 10-28) - fixed
**Symptom:** `scripts/gen_snn_vectors.py` failed with `SyntaxError: unterminated string literal` right after an edit made from a heredoc Python snippet (the same failure as bug 7).
**Root cause:** the replacement text was written inside a Python string in the snippet, so `"\n"` became a real newline in the new source. It happened twice in one session (the `clips` function, then the membrane dump).
**Fix:** repaired with the Edit tool. New rule for myself: source files with escapes are written or patched with the Write/Edit tools, not through a Python string inside a shell heredoc. The same shell heredoc also failed outright when the text contained an apostrophe (`unexpected EOF while looking for matching`), which is why `rtl/snn_demo_top.v` and `scripts/snn_regression.py` were written with the Write tool.

### 27. `bins` is a reserved word in SystemVerilog (RTL bench, 10-28) - fixed
**Symptom:** `tb/snn_core_tb.v` failed to compile under `iverilog -g2012`: "syntax error ... Syntax error in variable list" on `reg [31:0] bins [...]`.
**Root cause:** `bins` is a covergroup keyword in SystemVerilog (the same class of mistake as bug 3, `inside`).
**Fix:** renamed to `binw`. While renaming I also rewrote the file name inside the `$readmemh` string (`binw.hex`), so every step read an empty file and the bench reported 47 mismatches with all-zero outputs; found because the model's spike counts were nonzero and the RTL's zero. Restored `bins.hex`.
**Lesson:** a bench that fails with "all zeros" is first a stimulus problem; check the input files are read before suspecting the DUT.

### 28. The SNN self-test showed two checks that could not see a clipping error (test quality, 10-28) - fixed
**Symptom:** the first mutation run of the SNN bench: the membrane-clip mutations (`VMAX` and `VMIN` off by one) and the tie-break mutation (`>` to `>=` in the argmax) were MISSED.
**Root cause:** the bench compared per-step spike counts and class currents only. A membrane that is 1 LSB off at a rail rarely changes a spike count, and random weights almost never produce an exact score tie, so those two properties were effectively untested.
**Fix:** the bench now also compares the full contents of both membrane RAMs after the last step with the model (`expect_v1.hex`, `expect_v2.hex`), and the saturating network now has two identical dominant output classes so that an exact tie occurs and class 0 must win.
**Lesson:** a passing bench says nothing about a property it cannot observe; the mutation self-test is what shows which properties those are.

### 29. Streaming resampler differed from the training blend by 1 grey level on rounding ties (software, 10-28) - fixed before use
**Symptom:** the new test `sw/tests/test_snn_stream.py::test_resampler_matches_training_blend` failed for 25 fps and 29.97 fps (maximum difference 1 grey level, on about 10% of the frames), while 24 and 50 fps matched.
**Root cause:** the laptop-side resampler advanced its output position by repeated addition (`pos += src/30`), the training code (`train.kth.resample_linear`) computes `m * src/30`. Accumulated float error moved exact ties (`a = 0.5`) to either side of the rounding boundary.
**Fix:** `Resampler30` now computes the position as `m * ratio` and blends with the same expression, so it equals the training resampler exactly.
**Lesson:** two implementations of "the same" resampling are only the same if a test compares them; the test caught it the first time it ran.

### 30. Declaration after use in `snn_engine.v` (RTL, 10-28) - fixed
**Symptom:** the first elaboration of the engine bench failed with "Unable to bind wire/reg/memory `pix_fire`".
**Root cause:** `pix_fire` and the other pixel-handshake wires were declared below the FIFO instance that uses `pix_fire`; Verilog needs the declaration first.
**Fix:** moved the declarations above the instance. No functional change.

### 31. 40 ms per message from the board: reply sent as two small TCP writes (software on the board, 10-29) - fixed, measured
**Symptom:** first run of the frontend on the real board kept only 20.0 frames/s whether it was paced at 30 fps or not; a micro-benchmark showed 39 ms for a single 4-byte CSR read round trip and 48.7 ms per frame (12 to 64 ms), although the engine needs 8.9 ms and ping answers in under 1 ms.
**Root cause (probable, not proven by a packet capture):** `send_msg` wrote the 16-byte header and the payload with two separate `write` calls, so every reply left the board as two tiny segments; the ~40 ms is the signature of a delayed-ACK/Nagle interaction on one of the two TCP stacks (TCP_NODELAY was set on the board's socket).
**Fix:** one buffer, one `write` per message (`sw/snn_streamer.c`, replies are at most 112 bytes). Measured on the board afterwards: CSR read 1.57 ms, frame round trip 11.7 ms (min 11.5, max 17.4), i.e. 85 frames/s possible and 29.4 frames/s sustained at the real-time pace. The comparison against the model passed before and after the change.
**Lesson:** a fixed per-message delay of about 40 ms means a TCP small-write interaction; send each protocol message with one write. The simulation could not show this, only the board did.

### 32. `wsl` from Git Bash turned `/mnt/f/...` into `C:/Program Files/Git/mnt/f/...` (tooling, 10-29) - worked around
**Symptom:** `wsl -d Ubuntu -- bash /mnt/f/.../arm_run.sh` failed with "No such file or directory: C:/Program Files/Git/mnt/f/...". Earlier the same call with a quoted loop variable (`--mode \$m`) lost the variable and printed the usage line instead of running.
**Root cause:** MSYS path conversion of arguments that look like POSIX paths when a Windows program is started from Git Bash; variables inside a `wsl ... bash -lc "..."` string are expanded in the wrong shell.
**Fix:** `MSYS_NO_PATHCONV=1` in front of `wsl`, and the commands written into a small script file that WSL runs (`work/arm_run.sh`); the pytest uses `bash -lc` with the path built in Python.
**Lesson:** put multi-step WSL commands in a script file; a failing path or a missing variable is a tooling problem before it is a design problem.

### 33. A figure claimed a clock-speed lever the timing report does not support (documentation, 10-29) - fixed before publishing
**Symptom:** while drawing the ARM-versus-FPGA figure I wrote "timing has more than 10 ns of slack, so a higher clock ... would be a large speed-up".
**Root cause:** the slack of 10.8 ns is on a 40 ns clock period, so the longest path is about 29 ns, i.e. the design can run at about 34 MHz without pipelining work, a 1.4x gain, not a large one.
**Fix:** text corrected in the figure, in `docs/results/arm_baseline.md` and in the README; the real levers are more parallel channels and an event-driven engine.
**Lesson:** convert slack to Fmax before using it as an argument.

### 34. Web demo draft: two threads could use the one board socket, and a stopped clip could close the next source's connection (software, 10-29) - fixed before first run
**Symptom:** found by reading the never-run draft of `scripts/snn_web.py` before running it. Nothing serialised `Backend.infer`: webcam frames are handled in HTTP threads (`ThreadingHTTPServer`), so two overlapping POSTs (two tabs, or a webcam POST while a clip thread still runs) would interleave FRAME_IN/RESULT on the single TCP connection and read each other's answers. Also, all sources shared `hub.backend`: if `stop()` gave up joining an old clip thread (5 s timeout, shorter than the 8 s board timeout), that thread's `finally: close_backend()` would close the backend of the source started after it.
**Root cause:** the draft kept per-source state (backend, window list, counters) on the shared hub and had no lock around the request/answer pair.
**Fix:** a `Run` object per source owns its backend, window state and a lock held for each frame's send-and-receive. Every frame checks that its run is still the current one (`Stopped` otherwise). A thread only closes its own run. `stop()` joins for 12 s, longer than one board timeout. In my first rewrite, `run_clip`'s `finally` took the start/stop lock, which `stop()` holds while it joins that same thread, so Stop could have stalled for 12 s. I caught this on re-reading, and it now uses a compare-and-clear under the status lock.
**Lesson:** with a single-connection protocol, each request/answer pair needs its own lock, and source-specific state belongs to the source, not to a shared object.

### 35. Web demo draft: webcam bookkeeping (software, 10-29) - fixed, tested with the model backend
**Symptom (from review):** `hub.web_source` and `hub.webcam_idx` were attributes created on the fly (set in `main`, not in `Hub`). A webcam POST from any tab was accepted, so two tabs' webcams would be mixed into one frame stream. A tab closed without Stop left the board connection open forever, which blocks `check_board.py` because the daemon serves one client. A looping clip with no page open would also hold the board forever.
**Fix:** `/api/webcam/start` returns a session token and frames must carry it (`X-Session`); a new start in any tab makes the old token answer 409, and the page then stops its camera. A watchdog stops a webcam source after 6 s without a frame and any source after 15 s with no page connected. The page also sends a beacon to `/api/stop` on `pagehide`. Tested: wrong token 409, old tab's token 409 after a new start, watchdog message after 6 s, looping clip stopped 15 s after the last viewer left.

### 36. Web demo draft: unchecked inputs, a loop that could spin, wrong frame total, short-clip answer (software, 10-29) - fixed, tested
**Symptom (from review, then confirmed by tests):** `/api/start` would open any path the client sent. `/api/upload` read the whole body into memory with no size limit and wrote it under any `X-Filename`, which was only stripped by `basename`. With loop on, a file that yields 0 frames would publish `clip_done` in a tight loop. `total` came from `CAP_PROP_FRAME_COUNT`, i.e. source frames (302 for a 25 fps KTH clip), although the clip yields 362 frames at 30 fps. For a clip shorter than 32 frames the draft returned no answer, where `snn_viewer.py` uses the running score.
**Fix:** only the listed clips can be started (400 otherwise). Uploads need a Content-Length of at most `--max-upload-mb` (default 200 MB; 413 before reading), a video extension and a sanitised name; they are streamed to `work/uploads` in 1 MB chunks, and a file OpenCV cannot decode is deleted (415). A pass with 0 frames raises an error. `total` uses the resampler's arithmetic (`(n-1)*30/fps + 1`, 362 = the frames actually produced). Short clips follow the viewer's rule. Bad JSON gives 400. Tests: a 10 GB Content-Length is refused with 413, `..\` paths and README.md with 400, junk `.mp4` with 415 and deleted, frame total 362 = 362.

### 37. Test-harness mistakes while testing the web demo (tooling, 10-29) - fixed
**Symptom:** two failures in my own API driver, not in the server: (1) it expected the sanitised upload name `my_walk_test_.avi`, but the sanitiser strips trailing `_`, so `my_walk_test.avi` is correct; (2) `SSE.close` raised `AttributeError: 'NoneType' object has no attribute 'shutdown'` because `http.client` detaches `conn.sock` once `getresponse()` returns.
**Fix:** corrected the expectation; the driver keeps its own reference to the socket before `getresponse()`.
**Lesson:** when a test fails, first check which side is wrong.

### 38. Zooming the phone clips made the answers worse: handheld shake magnified (preparation, 10-29) - fixed by shake removal and a milder crop
**Symptom:** the first crop of the four phone clips (320x240 window around the person, 2.7x zoom) went through the board and scored 0 of 4 clips (hand waving answered running / jogging), against 2 of 4 for the uncropped originals; events rose from about 1% to 3-6% of pixels.
**Root cause:** the clips are handheld: the picture drifts 15-25 px over a clip and jitters 0.4-0.6 px per frame. After a tight crop that jitter is several pixels at 128x128 and every wall edge of the corridor produces events. Zooming further also changed the answers on its own (see `docs/results/user_clips.md`; the reason was not investigated).
**Fix:** `scripts/crop_user_clips.py --stabilize` shifts every frame onto the first (phase correlation) before cropping, and the default crop is 640 wide (1.3x), the best of the variants in `scripts/crop_sweep.py`. A first version of the sweep crashed on the 640 window (centre y 235 minus half the height is negative, an empty slice); the window is now clamped inside the picture.
**Lesson:** zoom is not free: it magnifies camera motion, so remove shake first and measure the crop on the model before using it; a plausible-looking preparation step can make results worse.

### 39. The board top level says HEX1 shows the window count; it shows a fixed F (documentation, 10-29) - documented, RTL not changed
**Symptom:** while describing the phone video of the board for the README, the display read `b8 18 F4`; the comment in `quartus/snn_soc/DE10_Standard_SNN.v` says "HEX1: window count", but HEX1 showed `F` with five windows finished.
**Root cause:** the code is `hex7seg hx1 (.d(4'hF), ...)`: HEX1 is tied to a constant F (a separator). The comment describes an earlier plan, not the code.
**Fix:** the README and the photo-session notes describe HEX1 as a fixed `F`. The RTL comment is left as it is on purpose: the archived bitstream in `releases/` records the SHA-256 of its sources, and editing even a comment would break that record; fix it with the next real rebuild.
**Lesson:** describe hardware outputs from the code (or a photo), not from comments.

### 40. "Identical to the KTH blending" was too strong for the recorder's resampler (test, 10-29) - fixed
**Symptom:** the new test `test_resample_times_equals_kth_blending_for_an_even_rate` failed: `train.recordings.resample_times` with even 25 fps capture times differed from `train.kth.resample_linear` on 142 of 3,904 values, all by 1 grey level, all on odd output frames.
**Root cause:** both blend the same two neighbours with the same weight, but the weight is computed with different floating-point operations; where a blend is exactly half-way (odd frames of a 25 -> 30 fps conversion) the two round the .5 differently.
**Fix:** the test now checks at most 1 grey level everywhere and exact equality on even frames; the docstring says so. No change to the code.
**Lesson:** state an equivalence only as strong as it is; "same algorithm" and "bit-identical output" are different claims.
