# KTH 6-class network (walking, jogging, running, boxing, hand waving, hand clapping), 2026-10-28

Data: all six KTH actions, 599 videos (one clapping video, person13_d3, is missing from the zip), 2,390 repetitions listed in `00sequences.txt`. Whole 160x120 frame squashed to 128x128, 25 -> 30 fps by blending, integer emulator (theta 0.20 ln, step mode), 2x32x32 presence bins, `python -m train.prep_kth` -> `data/kth32`. Person split of the original paper: train 11-18 (760 repetitions), validation 19-21 23-25 01 04 (768), test 22 02 03 05-10 (862).
Network: 32x32 input, conv 16/32, sum readout, window T = 32 steps (1.07 s), 60 epochs, lr 0.008, no augmentation, int8 weights, QAT (`train/runs/kth6_32_16_32`, log in `kth6_train.log`). Window accuracy = fraction of non-overlapping 32-step windows classified correctly; sample accuracy = the repetition is scored by the summed scores of its windows.

## Result (the network that is frozen into the board design)

| | window accuracy | sample accuracy |
|---|---|---|
| Validation people | 0.765 | 0.807 |
| **Test people (scored once, after the choice)** | **0.727** | **0.799** |

Per class on the test people (window / sample recall): walking 0.79 / 0.92, jogging 0.47 / 0.67, running 0.45 / 0.65, boxing 0.88 / 0.92, hand waving 0.81 / 0.88, hand clapping 0.73 / 0.76.
Per condition (window): d1 0.81, d2 (scale change) 0.58, d3 (clothes) 0.77, d4 (lighting) 0.79.
Test confusion by repetition (rows true; walking, jogging, running, boxing, waving, clapping):

```
walking   [132  12   0   0   0   0]
jogging   [ 31  96  17   0   0   0]
running   [  8  42  94   0   0   0]
boxing    [  8   1   0 132   2   0]
waving    [  4   2   4   0 126   8]
clapping  [ 10   0   2  21   1 109]
```

The integer model (`model/snn_int.py`) predicts exactly what the GPU evaluation predicts on 100 test repetitions (0 differences), and the RTL matches the integer model bit for bit (`docs/results/snn_rtl.md`).

## Reading

- Jogging and running are mistaken for each other (and jogging for walking): they differ mainly in speed, and the 2x32x32 presence maps with a 1 s window carry little of it. This is the known hard pair of KTH and it is where almost all the lost accuracy sits; the other four actions are at 73-88% per window.
- The d2 condition (the camera zooms, so the person changes size) is the weakest at 0.58.
- Sample accuracy of 80% is below what published methods working on raw video reach on KTH. This network is deliberately tiny (17 k int8 weights), sees 32x32 binary event maps and uses integer arithmetic only; the number is what that system achieves, not a claim about the state of the art. Protocol differences (this split by person, windows of 1 s, no per-video normalisation) also make a comparison with published numbers unreliable.

## What else was tried (validation people only; none is better beyond noise)

| Variant | window | sample |
|---|---|---|
| baseline (frozen) | 0.765 | 0.807 |
| event dropout 0.1 | 0.765 | 0.800 |
| window T = 48 steps (different windows, not directly comparable) | 0.770 | 0.811 |
| weight decay 0.01 + dropout 0.05 | 0.759 | 0.807 |

One seed each, so differences below about 1.5 points are noise (the DVS128 two-seed runs differed by that much). Not tried: longer temporal memory (a slower leak) or speed-aware features for jogging versus running; a higher frame-rate input (the 25 fps source limits motion information); training on train+val.

## Earlier 3-class stage (clapping, waving, boxing)

Validation 0.882 window / 0.904 sample (`kth3_accuracy.md`). Boxing was the weak class there (0.70).

## Reproduce

```
python -m train.prep_kth                                            # needs data/kth/videos (all six zips extracted)
python -m train.train_snn --dataset kth --data data/kth32 --T 32 --epochs 60 --lr 0.008 --out train/runs/kth6_32_16_32
python -m train.eval_kth6 --weights train/runs/kth6_32_16_32 --data data/kth32 --split val --int-check 40
```
