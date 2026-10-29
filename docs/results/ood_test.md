# Footage that is not from KTH, through the real board (2026-10-29)

`python scripts/ood_test.py --connect 192.168.137.200 --out docs/results/ood_board.json` (figure: `docs/media/fig_ood.png`). Ten clips, 6,280 frames, every frame also run through the bit-accurate model: **the board matched the model on all 6,280 frames**, including very noisy dark video, so the hardware is as exact on this data as on KTH. The interesting part is what the network says.

Sources, all from the author's other projects: thermal pedestrian sequences of CDnet 2014 (`park`, `corridor`, `lakeSide`, `library`, `diningRoom`: static thermal camera, 240x320 or 288x352 JPEG frames) and hand-in-a-dark-room clips of the night-vision project (1080p, 58 fps resampled to 30, ISO 2000, very noisy). Frames are prepared exactly like any other clip (grey, squashed to 128x128); the frame rate of the CDnet thermal sequences is not given, 30 fps is assumed. Labels are **weak**: where people walk past the camera the expected class is walking; where the footage shows none of the six actions there is no correct answer.

| Clip | What it shows | Events per frame | Window answers (1 s each) | Clip answer |
|---|---|---|---|---|
| KTH walking (reference) | person 22, in distribution | 2.2% | walking 78%, jogging 17%, running 6% | walking |
| KTH boxing (reference) | person 22, in distribution | 2.5% | boxing 73%, walking 27% | boxing |
| thermal park | pedestrians on a path, side view | 1.4% | walking 33%, running 28%, jogging 22%, hand waving 17% | walking |
| thermal corridor | people far away in a corridor | 1.1% | walking 61%, boxing 36% | walking |
| thermal lakeSide | small distant people | 0.1% | walking 76%, hand waving 16%, boxing 8% | walking |
| thermal library | a person sits down (**no correct answer**) | 2.1% | walking 79%, hand waving 11%, jogging 7% | walking |
| thermal diningRoom | a person moving indoors (**no correct answer**) | 2.8% | walking 96% | walking |
| hand, fast, dark room (**no correct answer**) | a hand moving, noisy | **24.0%** | running 100% | running |
| hand, slow, dark room (**no correct answer**) | slow hand, noisy | 13.4% | running 52%, walking 24%, jogging 19% | running |
| empty static scene, dark (**nothing moves**) | only sensor noise | 2.3% | running 58%, walking 42% | running |

## What this shows
1. **It transfers only partly.** On the walking footage that is not KTH the clip answer is right every time, but per window the walking share is 33% (park), 61% (corridor) and 76% (lakeSide) against 78% on KTH walking; the park sequence (a clear side view, the most KTH-like) is the worst, most likely because of thermal contrast, scale and frame-rate differences.
2. **There is no "none of the above".** A person sitting down is answered "walking" in 79% of windows, a person moving about "walking" in 96%, and **an empty, static dark scene is answered "running" or "walking" in every window**: the network turns sensor noise into an action (for the thermal clips possibly also the intensity changes of the cameras' automatic gain control; not checked). A real use would need a reject class or a minimum-activity gate; neither exists.
3. **Dense event input is far outside training.** The noisy 1080p clips produce 13-24% of pixels as events against 1-5% in KTH; the answer is "running" almost throughout.
4. Hardware is not the limit here: the board equals the model frame for frame on all of it. The limits are the network, its training data and the lack of a reject class.

## Your own clips (phone)
The network was trained on KTH: a **static camera**, a **whole person seen from the side** in a roughly 4:3 picture, plain background, 30 fps, one action for 5-10 s. To get a fair try:
1. Prop the phone (landscape, 30 fps, no zoom, no stabilisation, fixed exposure if the app allows) 3-4 m from the person, whole body in frame, plain wall or open ground behind, nothing else moving.
2. One action per clip, 8-10 s, walking across the frame, jogging/running across the frame, boxing and hand waving/clapping standing still in the middle of the frame.
3. Put the files in `data/user_clips/` and run `python scripts/snn_viewer.py --source data/user_clips/yourclip.mp4 --connect 192.168.137.200` (or load them in the live demo frontend).
Expect it to do worse than on KTH when the framing differs; the clips above show how it degrades. If the person is small in the picture, crop with `python scripts/crop_user_clips.py --stabilize` (mild crop, shake removed); do not zoom in tightly, it made things worse. First results: `docs/results/user_clips.md`.
