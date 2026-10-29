# Track A: DVS128 Gesture accuracy (real events)

Data: raw IBM release, `python -m train.prep_dvsgesture` -> 1176 train / 288 test gesture windows (IBM train/test trial lists, 98 / 24 trials, no subject overlap by construction of the lists). Input: 2 x 32 x 32 binary presence maps, one per 33,333 us.
Training: 60 epochs, Adam, cosine schedule, batch 16, random 48-step window per sample + random +-2 cell shift. Test: every non-overlapping 48-step window of every test sample (1377 windows; the last window of a sample is zero-padded); "sample" = scores summed over a sample's windows. **Single seed (0), 288 test samples: differences of 1-2 points are within noise (a 95% interval on 288 samples is about +-4 points).**

| Model | Weights | Test window acc | Test sample acc | Command |
|---|---|---|---|---|
| ANN baseline (conv-ReLU-pool x2 + FC on time-averaged frames) | float | 81.9% | 88.5% | `train_snn --ann --lr 0.002` |
| SNN, float weights | float | 79.6% | 87.8% | `train_snn --float --lr 0.008` |
| **SNN, QAT** (PyTorch forward) | int8 | 80.9% | 88.2% | `train_snn --lr 0.008` |
| **SNN, integer model** (`model/snn_int.py`, numpy int arithmetic, exported weights) | int8 | **80.90%** | **88.19%** | `eval_int --weights train/runs/dvsg --dataset dvsg` |

- The integer model reproduces the QAT numbers exactly on all 1377 windows, so quantization to int8 costs nothing measurable here and the export is bit-faithful. This is the number the SNN RTL must match.
- The ANN baseline sees the time-averaged frame, so it has no temporal order information; it is a reference, not an upper bound.
- Architecture: conv(2->8) LIF, OR-pool, conv(8->16) LIF, OR-pool, FC(1024->11) with non-spiking sum readout (decision D10), thr 64, leak shift 4, 12-bit membrane, 12.6k int8 weights.

## Sparsity (what drives hardware work), 48-step window, integer model
Layer 1: 33,911 spikes per window (8.6% of neurons per step). Layer 2: 3,765 spikes (1.9%). About 37.7k spikes per inference, roughly 800 per time step, against 8,192 + 4,096 neurons.

## Where it fails (integer model confusion, rows = true class)
Largest off-diagonal counts (windows): air guitar -> other 32, air drums -> hand clapping 26 and -> arm roll 23 (drums is correct in only 51 of 118 windows), left arm cw -> ccw 20, right arm ccw -> cw 14, right arm cw -> ccw 12, other -> air guitar 14. So the weak spots are the direction of arm circling, air drums, and the catch-all "other" class, as the roadmap risk table anticipated. Per-class window recall: clapping 84%, right wave 99%, left wave 91%, right arm cw 84%, ccw 78%, left arm cw 81%, ccw 89%, arm roll 94%, air drums 43%, air guitar 63%, other 71%.

## Things that went wrong on the way (kept so they are not repeated)
1. Spiking output layer with count readout: stuck at chance, loss above ln 11 (outputs fire always or never). Fixed by the sum readout (D10).
2. Surrogate gradient slope not scaled by 1/thr: exploding backward signal (layer-1 gradient 10^6 vs 10^-3 at the output). Fixed in `train/snn_qat.py`.
3. Toy-tuned initial weights saturate layer 2 on dense DVS input; use `--init-std 20,10,3` and check 5-20% firing at init.


# Network size study (2026-10-26)

All runs: DVS128 Gesture, 60 epochs, lr 0.008, init std calibrated per layer to ~8% / 10% firing, sum readout, int8 QAT. The integer model (`eval_int`) reproduced the QAT test numbers for every export that was checked (32x32 8/16, 32x32 16/32, 64x64 16/32).

| Input | Channels C1/C2 | Seed | Test window | Test sample | Train acc | Weights (int8) | Membrane words | Memory (lower bound, incl. emulator) | L1 spikes per step |
|---|---|---|---|---|---|---|---|---|---|
| 32x32 | 8/16 | 0 (hand-set init) | 80.9% | 88.2% | 89% | 12.6k | 12.3k | 46 M10K (8%) | 706 |
| 32x32 | 8/16 | 1 | 80.9% | 87.5% | 92% | 12.6k | 12.3k | 46 M10K | not measured |
| **32x32** | **16/32** | 0 | **82.8%** | **90.3%** | 93% | 27.4k | 24.6k | 72 M10K (13%) | 914 |
| **32x32** | **16/32** | 1 | **82.6%** | **89.2%** | 94% | 27.4k | 24.6k | 72 M10K | not measured |
| 64x64 | 8/16 | 0 | 78.6% | 86.5% | 91% | 46.4k | 49.2k | 115 M10K (21%) | not measured |
| 64x64 | 16/32 | 0 | 81.3% | 90.3% | 92% | 95.0k | 98.3k | 211 M10K (38%) | 3,934 |

Reading it:
- Doubling the channels at 32x32 gives a small but consistent gain (about +1.8 points window accuracy, +1.8 sample accuracy, same direction in both seeds). A single comparison on 288 test samples has a 95% interval of roughly +-3.5 points for the sample figure, so only the consistency across two seeds supports it.
- 64x64 input did not help with this architecture: 8/16 channels is worse than the 32x32 baseline, and 16/32 matches 32x32 16/32 at about 3x the memory and about 4x the layer-1 spike work. The information the network can use is not limited by 32x32 resolution here. Only one seed was run at 64x64, so a small 64x64 advantage cannot be excluded, but it would have to be paid for in hardware.
- Every network overfits (train 90-94% against test 78-83%), so the next gain is more likely to come from regularisation, temporal modelling or data (subjects) than from more pixels. Not tried yet: input event dropout and stronger augmentation, a longer window, a third conv layer, a longer-memory path for arm-circling direction.
- Hardware work, event-driven estimate with the firing rates measured on the test set (`scripts/snn_resources.py`, rough): about 0.65M synaptic additions per 33 ms step at 64x64 16/32, 0.18M at 32x32 16/32 and 0.08M at the old 32x32 8/16. (An earlier version of this line said 0.9M / 0.22M / 0.06M; those came from an estimator that used the 64x64 input density for every size, see bug log 17.)

Decision: the default network is now **32x32 input, 16/32 channels** (`SNNConfig` defaults). 64x64 stays supported by the code (`--hw 64`, prep `--down 2`) and tested bit-exact. The architecture is not frozen (roadmap 3.12).


# Regularisation and third-conv-layer study (2026-10-27)

Same setup as the size study (32x32 input, DVS128 Gesture, lr 0.008, calibrated init, sum readout, int8 QAT, seed 0 unless stated). Reference: 32x32 16/32, 60 epochs: **82.8% window / 90.3% sample** (seed 1: 82.6 / 89.2).

## Round 1: regularisation (60 epochs, one change at a time)
| Change | Test window | Test sample | Train acc |
|---|---|---|---|
| none (reference) | 82.8% | 90.3% | 93% |
| event dropout 0.2 (each active input cell removed) | 80.4% | 86.8% | 91% |
| left/right mirror 50% with class swaps | 78.2% | 85.1% | 88% |
| 2x speed (OR-merged pairs of bins) for 50% of windows | 81.3% | 89.2% | 92% |
| AdamW weight decay 0.3 | 79.8% | 87.5% | 87% |
| dropout + mirror + speed together | 77.8% | 85.8% | 88% |

All worse than the reference, and the train-test gap did not shrink (9-11 points in every row), so these augmentations reduce fitting without improving generalisation within 60 epochs. Mirroring hurt most. Checked that the class mapping was not the cause: the reference model scores 64.5% on mirrored test windows with mapped labels against 28.2% with unmapped labels (waves keep 97% / 92% recall), but mirrored arm circles lose 20-30 points of recall (e.g. left arm ccw 0.93 -> 0.47), which suggests real left-arm and right-arm gestures are not mirror images in this dataset (bug log 18; the cause is probable, not verified).

## Round 2: longer training, mild regularisation, third conv layer
| Network / training | Epochs | Test window | Test sample | Train acc | int8 weights | membrane words | spikes per window (integer model) |
|---|---|---|---|---|---|---|---|
| 2 conv 16/32 (reference) | 60 | 82.8% | 90.3% | 93% | 27.4k | 24.6k | 47.4k |
| 2 conv 16/32, control | 90 | 83.1% | 90.3% | 96% | 27.4k | 24.6k | not measured |
| 2 conv 16/32, dropout 0.05 + speed 0.25 | 90 | 82.2% | 88.2% | 95% | 27.4k | 24.6k | not measured |
| **3 conv 16/32/32** | 60 | 82.9% | 90.3% | 94% | **19.7k** | 26.6k | 76.1k |
| 3 conv 16/32/64 | 60 | 83.1% | 91.0% | 96% | 34.6k | 28.7k | not measured |

The integer model reproduced the QAT test accuracy for the 3-conv 16/32/32 export (82.93% / 90.28%).

Per-class window recall (`python -m train.per_class`):
| Run | clap | R wave | L wave | R cw | R ccw | L cw | L ccw | roll | drums | guitar | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 conv reference s0 | .80 | .98 | .94 | .78 | .77 | .94 | .93 | .97 | .51 | .68 | .64 |
| 2 conv reference s1 | .78 | .99 | .97 | .79 | .73 | .88 | .93 | .94 | .58 | .68 | .68 |
| 2 conv, 90 epochs | .81 | .98 | .95 | .83 | .74 | .92 | .97 | .95 | .49 | .62 | .73 |
| 3 conv 16/32/32 | .78 | .98 | .95 | .84 | .77 | .91 | .88 | .91 | .63 | .64 | .74 |
| 3 conv 16/32/64 | .88 | 1.00 | .96 | .83 | .62 | .91 | .89 | .93 | .63 | .70 | .70 |

## Conclusions
- **Accuracy plateau at about 83% per window / 90% per sample** for this input and neuron model: more depth, more epochs, mild augmentation and 64x64 input all land within about half a point of it (the seed-to-seed spread of the reference is 0.2 points window / 1.1 sample). None of the changes tried is a measurable improvement.
- The residual errors are concentrated in air drums (recall 0.49-0.63), air guitar (0.62-0.70), "other" (0.64-0.74) and the right-arm circling pair (0.62-0.84). A third conv layer moves errors between these classes but does not remove them. The first three are ambiguous by nature (drums versus clapping and arm roll, guitar versus "other"); improving them probably needs a different kind of change (explicit temporal modelling, more subject diversity, or a longer or smarter decision window), none of which has been tried.
- A third conv layer with 32 channels matches the accuracy with 28% fewer weights (the FC input shrinks from 2048 to 512) but fires 1.6x more spikes per window and adds a layer to implement, so it is not adopted.
- Not tried: a longer decision window than 48 steps, a recurrent or longer-memory path (leak shift), per-class thresholds, a different readout (e.g. last-quarter accumulation), subject-level cross-validation, more than two seeds.
