# KTH 3-class network (hand clapping, hand waving, boxing), validation people, 2026-10-28

Data: KTH videos (whole 160x120 frame squashed to 128x128, 25 -> 30 fps) -> integer emulator (theta 0.20 ln, step) -> 2x32x32 presence bins (`train.prep_kth --actions handclapping,handwaving,boxing --out data/kth32_3`). Train people 11-18 (376 repetitions), validation 19-21 23-25 01 04 (384), test 22 02 03 05-10 (430, **not used yet**).
Network: 32x32 input, conv 16/32, sum readout, T = 32 bins, 60 epochs, lr 0.008, no augmentation (`train/runs/kth3_32_16_32`, log `kth3_train.log`). The exported final epoch is scored; no epoch or setting was picked on the validation people (the emulator setting is the hardware default and the hyperparameters were copied from the DVS128 run).

| | window accuracy | sample accuracy |
|---|---|---|
| Validation people | **0.882** | **0.904** |
| Training windows (last epoch) | 0.984 | |

Per class (window / sample recall): boxing 0.702 / 0.734, hand waving 0.966 / 1.000, hand clapping 0.954 / 0.977.
Per condition (window): d1 0.881, d2 0.882, d3 0.891, d4 0.875 (no condition stands out).
Sample confusion (rows true, columns predicted: boxing, waving, clapping): boxing [94, 24, 10], waving [0, 128, 0], clapping [1, 2, 125].
The integer model (`model/snn_int.py`) gives the same predictions as the GPU evaluation on 40 samples (0 differences).

Reading: training accuracy is 98% against 88% on new people, so the network overfits the 376 training repetitions; boxing is the weak class and is mostly mistaken for waving. Three classes only; walking, jogging and running are still downloading. Not done: the test people, augmentation, other emulator thresholds.
