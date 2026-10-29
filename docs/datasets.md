# Datasets and downloads

Everything goes under `data/` (gitignored). Items marked **(not verified)** are things I could not confirm from this machine; check them before relying on them. Read each licence before using results in anything public.

## 1. Required

| # | What | Used for | How to get it |
|---|---|---|---|
| 1 | **DVS128 Gesture** (IBM, 11 classes, real DVS events) | Track A: train/test the SNN (roadmap 3.3, 3.6, 3.9) | **Downloaded (2026-10-26)** from IBM as the raw release into `DVSGesture/` at the project root (5.0 GB, 122 `.aedat` trials from 29 subjects under 5 lighting combinations, plus `*_labels.csv`, `trials_to_train.txt` (98 trials) and `trials_to_test.txt` (24 trials), `gesture_mapping.csv`; CC BY 4.0, cite IBM Research). It is gitignored. `python -m train.prep_dvsgesture` parses the AEDAT 3.1 files (`train/aedat.py`) and writes `data/dvsg32/{train,test}.npz`: **1176 train and 288 test samples**, one per labelled gesture window (the same counts SpikingJelly and snnTorch report). The tonic route (`--source tonic`, figshare, preprocessed copy) is no longer needed. |
| 2 | **KTH actions** (6 actions, 599 videos) | The demo network: video -> emulator -> SNN, trained and tested on emulated events (roadmap 3.x, `docs/results/kth6_accuracy.md`) | **Downloaded and verified (2026-10-28)**, section 2b. |
| 3 | Your own gesture clips (optional) | Extra test of the emulator on your own camera | You record them. Protocol in section 4; no longer needed for the demo. |

Sample counts (1176 / 288) differ from the 1342 instances quoted in the paper; the difference comes from how the labelled windows are split between train and test lists. Report the counts you actually used. Both arm-roll directions share class 8, as in the dataset. Recording-level facts checked on this copy: coordinates are 0..127, timestamps are monotonic microseconds, polarity is balanced (49.8% ON), and **no mirroring**: right-hand gestures sit at image x about 40 and left-hand gestures at x about 80, the normal view of a person facing the camera. Phone video of a person facing the camera needs no flip.

## 2. Needed for the emulator comparison (roadmap 2.2, 2.6)

| What | Notes |
|---|---|
| **v2e** (https://github.com/SensorsINI/v2e) | Reference simulator. Its README uses a **separate conda env with Python 3.10** and PyTorch with CUDA; install with `python -m pip install -e .` from a clone. Do not install it into this project's venv. It writes events with `--dvs_text` as one event per line: `timestamp_seconds x y polarity`. |
| **SuperSloMo checkpoint** `SuperSloMo39.ckpt` (~151 MB) | Google Drive link in the v2e README. Only needed for frame interpolation; run v2e with `--disable_slomo` to skip it (fine for a first comparison, since the emulator itself is frame-based). |
| ~10 short clips (3-10 s) | Your own clips are enough. |

Run v2e on the same clips you push through `emulator_int.py`, then `python scripts/compare_vs_v2e.py --clip clip.mp4 --v2e v2e_events.txt` (event counts per frame, polarity balance, event-frame correlation). Use the settings you pass to v2e as the emulator's `theta` (0.2 ln units ~ theta 9).

## 2b. KTH actions: the dataset of the demo network

Source: per-action zips at `https://www.csc.kth.se/cvap/actions/` (the old `nada.kth.se` address returns an HTML page, not the zips). Downloaded into `data/kth/` (gitignored), each zip checked against the size the server reports **and** by CRC (`zipfile.testzip`) before use: `walking.zip` (242 MB), `jogging.zip` (169 MB), `running.zip` (150 MB), `boxing.zip` (194 MB), `handwaving.zip` (219 MB), `handclapping.zip` (177 MB), plus `00sequences.txt`; extracted to `data/kth/videos/` (599 `.avi` files; one clapping video, person13_d3, is not in the zip). The server is slow (12 to 100 KB/s) and drops connections; `boxing.zip` once reached the right size with 32 corrupt videos (bug log 24, 25).

- 25 subjects, static camera, 160x120, **25 fps**, DivX AVI; four conditions: d1 homogeneous background, d2 + scale change, d3 + different clothes, d4 + lighting changes. About four repetitions per video (frame ranges in `00sequences.txt`; 2,390 repetitions in all). Split by person as in the paper: train 11-18, validation 19-21 23-25 01 04, test 22 02 03 05-10.
- Terms (from the page): publicly available for non-commercial use; cite Schuldt, Laptev, Caputo, "Recognizing human actions: a local SVM approach", ICPR 2004.
- Pipeline: `train/kth.py` (frame ranges, 25 -> 30 fps by linear blending, `fit` = whole frame squashed to 128x128 or centre crop), `train/prep_kth.py` (integer emulator + binning -> `data/kth32/{train,val,test}.npz`), `train/eval_kth6.py` (scores).
- The earlier domain-gap test of the DVS-trained network on KTH hand clapping and waving: `train/eval_kth.py`, `docs/results/kth_domain_gap.md`.

## 3. Optional

| What | Used for | Where |
|---|---|---|
| **KTH actions** | the dataset of the demo network, see section 2b | |
| **UCF101 subset** | Same purpose, more variety | https://www.crcv.ucf.edu/data/UCF101.php **(not verified)** |

UCF101 is not needed; your own clips are optional (section 4).

## 4. Recording your own gesture clips (decides whether Track B is usable)

The full, exact instructions (what to record, phone and Blackmagic Camera settings, framing, file layout, a checker script) are in **`docs/recording_guide.md`**. In short: 6 core classes (`00_hand_clapping`, `01_right_hand_wave`, `02_left_hand_wave`, `05_left_arm_cw`, `06_left_arm_ccw`, `07_arm_roll`), 30 fps, manual exposure with shutter 1/100 s, fixed white balance and focus, tripod, plain wall, about 10 s per take, 12 takes per class over two lighting sessions, folders named with the DVS128 class number (`NN_name`) because that number is the label. Check clips with `python sw/check_clip.py data/own_clips`.

## 5. Board-side downloads (only when you reach Phase 0.6, 0.7 and Phase 1)

You already have most of these from the sister projects (`F:\Projects\FPGA-Adaptive-Night-Vision` and the CFAR folder, see `docs/board_notes.md`):

- Terasic **DE10-Standard System CD** (golden top, demos, user manual PDF, Linux SD images).
- A Linux SD image: record which one is actually on your card (the sister notes disagree: Angstrom 2014 vs Terasic Ubuntu console).
- A microSD card (8 GB+), and a USB-UART terminal (PuTTY or the sister project's `serial_console.ps1`).

No other downloads are needed: Quartus 21.1 Lite, Icarus Verilog and Questa FSE are already installed on this PC, and torch 2.14 with CUDA, cocotb, numpy, h5py, opencv-python and tonic are in the project venv.
