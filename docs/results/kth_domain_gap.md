# Track A network on emulated real video (KTH), 2026-10-27

Network: `train/runs/dvsg_32_16_32` (trained on real DVS128 Gesture events only, 32x32, conv 16/32, sum readout).
Input: KTH hand clapping and hand waving videos, through the integer emulator (`model/emulator_int.py`), same binning as training. Emulator setting (theta, step or reset) chosen on validation people only; test people reported. Windows: 48 time steps (1.6 s), stride 24. Correct = class 0 for clapping, class 1 or 2 for waving. Raw output: `kth_domain_gap_raw.log`. Reproduce: `python -m train.eval_kth --weights train/runs/dvsg_32_16_32 --kth data/kth --sweep`.

## Result: the network does not transfer

| | Real DVS test (Track A) | KTH, default 0.20 ln step | KTH, chosen 0.30 ln reset |
|---|---|---|---|
| Hand clapping, window acc | recall 0.80 | 0.25 (n=143 seqs) | 0.20 |
| Hand waving, window acc | right 0.98, left 0.94 | 0.52 (n=144 seqs) | 0.53 |
| Active input fraction | 0.166 (train) | 0.062 | 0.036 |

- Validation sweep (5 thetas x step/reset): the best mean window accuracy is 0.37, so no emulator setting rescues it. Clapping stays between 0.07 and 0.23 for every setting; waving rises with theta (0.40 at 0.10 ln to 0.64 at 0.40 ln) while clapping falls.
- Per condition (chosen setting): clapping d1 0.07, d2 0.32, d3 0.39, d4 0.03; waving d1 0.68, d2 0.44, d3 0.65, d4 0.35.
- Predicted classes: clapping is called left wave 39%, clap 20%, other 18%, drums 12%, right wave 7%. Waving is called left wave 46%, clap 23%, other 17%, right wave 7%.
- Waving at about 0.5 is above chance for 11 classes (0.09) but nowhere near the 0.94 to 0.98 on real events; clapping is at about chance level among the plausible classes.

## What this does and does not show

It shows that this network, trained on real DVS events only, is not usable on emulated KTH video as it stands. It does **not** isolate the emulator as the cause, because KTH differs from DVS128 Gesture in ways unrelated to the emulator:
- The person is small in the frame (160x120 source, person fills a fraction of it), while DVS128 gestures fill most of the sensor. The active input fraction after the emulator is 0.04 to 0.06 against 0.166 in training, so the network sees a much sparser map.
- Different gestures: KTH waving is a two-arm wave and KTH clapping is a different motion from the DVS128 clap, at 25 fps against the continuous DVS time axis (resampled to 30 fps by blending).
- Backgrounds in d2 to d4 (zoom, clothes, lighting changes); d1 and d4 clapping are the worst conditions.
- One clip is absent from the zip: `person13_handclapping_d3` (99 clapping files, 100 waving).

To separate those causes, the next experiments (not run yet) are:
1. Fine-tune on KTH training people (11 to 18) and test on the test people: shows whether emulated KTH events are learnable at all with this network.
2. Crop and zoom the KTH person to the central region before the emulator, to match the scale of the DVS128 gestures.
3. The same test on your own recordings, which are shot to match the DVS128 framing (`docs/recording_guide.md`). This is the cleaner test of the emulator itself.
