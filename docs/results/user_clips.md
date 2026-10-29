# The first phone clips, through the real board (2026-10-29)

Four clips shot by the author (`hand_clapping1/2`, `hand_waving1/2`, 854x480, about 30 fps, 15 s each): handheld phone, a person 8-10 m away at the end of a long corridor, strongly backlit (mostly a silhouette), two different light conditions. Not KTH footage: the person is small, the background is full of edges, the camera moves.

`python scripts/user_clip_test.py --connect 192.168.137.200 --out docs/results/user_clips_board.json` (3,640 frames; every frame also run through the bit-accurate model: **board and model agree on all of them**).

## Preparation: crop, but only a little
`scripts/crop_user_clips.py --stabilize --width 640`: a fixed 640x480 window around the person (centre 430, 235), camera shake removed first (each frame shifted onto the first by phase correlation), scaled to 320x240. Originals are kept in `data/user_clips/originals/`; the demo's clip list shows the `_crop` files.

Why only a little: `scripts/crop_sweep.py` (model only, first 320 frames of each clip, 4 clips, so indicative only) tried five crop widths with and without shake removal.

| Crop width (854 = no crop) | Shake removal | Events (% of pixels) | Windows naming the true action | Clip answers right |
|---|---|---|---|---|
| 854 | off | 1.3% | 17 / 40 (42%) | 2 / 4 |
| 854 | on | 0.4% | 16 / 40 (40%) | 1 / 4 |
| **640** | off | 2.0% | 14 / 40 (35%) | 2 / 4 |
| **640** | **on** | 0.6% | **18 / 40 (45%)** | 2 / 4 |
| 520 | off / on | 2.7% / 0.9% | 0 (0%) / 3 (8%) | 0 / 0 of 4 |
| 420 | off / on | 3.3% / 1.1% | 2 (5%) / 7 (18%) | 0 / 1 of 4 |
| 320 | off / on | 4.1% / 1.6% | 7 (18%) / 10 (25%) | 1 / 1 of 4 |

- **Camera shake is real and zoom magnifies it**: the cameras drift 15-25 px over a clip and jitter 0.4-0.6 px per frame. Shake removal cuts the event rate 2.5-3.5x at every width.
- **Zooming in further is worse, not better**: with a 320-pixel crop (the person fills the picture, closer to KTH's scale) hand waving is answered as running or jogging. The first crop I made (320 wide, no shake removal) scored 0 of 4 in the board run before this sweep: hand waving answered "running" (8 of 14 windows) and "jogging", clapping "running" / "jogging". Why the network reacts that way was not investigated; a plausible but untested reason is that a big dark silhouette against a bright background produces long strong edges unlike anything in KTH's grey-on-grey recordings.
- A 640-wide crop with shake removal is the best of the variants tried, but it is not clearly better than the uncropped clip (45% against 42% of windows on 40 windows is inside noise).

## Result on the board (final crops and originals)
| Clip | True | Original: clip answer, windows | Cropped (640, stabilised): clip answer, windows |
|---|---|---|---|
| hand_clapping1 | hand clapping | hand waving (14 of 14 windows) | hand waving (12 of 14, walking 2) |
| hand_clapping2 | hand clapping | hand waving (11 of 14; boxing 1, walking 2) | hand waving (10 of 14, walking 4) |
| hand_waving1 | hand waving | **hand waving** (12 of 14, boxing 2) | **hand waving** (14 of 14) |
| hand_waving2 | hand waving | **hand waving** (11 of 14, walking 3) | **hand waving** (14 of 14) |

- **Hand waving is recognised** (clip answer right 2 of 2; 100% of windows on the crops, 82% on the originals).
- **Hand clapping is never recognised**: all four clapping clips are answered hand waving, and not a single window out of 56 says clapping. Clapping and waving are the two hand actions of KTH and look alike from far away; here the hands are tiny (a few pixels at 128x128) and the person is a silhouette. Two clips of one person are far too few to say more than that.
- Overall 4 of 8 clip answers right, on a set that is one person, one place and two actions.
- The expectation that phone clips do worse than KTH holds for clapping. For waving they did not do worse, but two clips of one person say little (KTH per-clip accuracy over all six actions is 79.9%).

## Not done
- No walking, jogging, running or boxing clips yet; no clips with the person larger or in front of a plain wall (the guide in `ood_test.md`).
- No claim about which of the preparation choices is best beyond this sweep of four clips.
