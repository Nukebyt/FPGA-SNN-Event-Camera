# Recording your own gesture clips (Track B)

**Goal:** short phone videos of the same gestures as DVS128 Gesture, shot so that my emulator turns them into events that look like the real dataset. They are used to (1) test the trained network on emulated events (does it still work? roadmap 3.10), (2) fine-tune it if it does not (3.11), and (3) drive the final demo.

**Optional, and no longer needed for the demo.** The demo network is trained on the public KTH videos instead (`docs/datasets.md`, section 2b; `docs/results/kth6_accuracy.md`). Recording your own clips is still useful as an independent test of the emulator on your own camera, and the classes in the table below are the DVS128 Gesture classes (a network trained on KTH does not know arm circles, arm roll or air drums).

Written for: you, recording with the Blackmagic Camera app. App menu names can differ between app versions, so settings are given by function. Everything marked *measured* comes from the real DVS128 Gesture recordings (`docs/results/dvs_reference/` shows what they look like).

---

## 1. What to record

**Core set, 6 classes** (the ones the network recognises best on real data, so a failure on your clips is the domain gap and not a weak class):

| Folder (exact name) | What you do | Pace (*measured*) |
|---|---|---|
| `00_hand_clapping` | Clap in front of your chest, continuously | about 3 claps per second |
| `01_right_hand_wave` | Wave **your right** hand at about shoulder-to-chest height | about 3 waves per second |
| `02_left_hand_wave` | Same with **your left** hand | about 3 per second |
| `05_left_arm_cw` | Whole **left** arm drawing circles in front of you, **clockwise from your own point of view** (as if drawing on a wall in front of you) | about 1 circle per second |
| `06_left_arm_ccw` | Same, **counter-clockwise from your own point of view** | about 1 circle per second |
| `07_arm_roll` | Roll your forearms around each other in front of your chest ("wheels on the bus"); do half the takes rolling forward, half backward, both are the same class | about 3 to 4 rolls per second |

Direction convention, *measured*: in the real dataset the "clockwise" classes trace a **counter-clockwise** path **in the video image** in 122 of 122 samples (for each arm). The camera sees you from the front, so your clockwise looks counter-clockwise on screen. Record by your own point of view and do not think about the screen. Right and left are **your** right and left arm; your right arm appears on the left side of the image (*measured*: right-hand gestures are centred at 36 to 38% of the image width, left-hand at 52 to 53%).

**Optional extras** (only if you want the full 11-class demo; the network is weaker on these even on real data, 50 to 75% recall): `03_right_arm_cw`, `04_right_arm_ccw`, `08_air_drums`, `09_air_guitar`, `10_other`.

**How many.** Per class: 6 takes in each of 2 sessions = 12 clips. 6 classes = 72 clips, about 12 minutes of video, about 1.5 hours with setup.
- **Session A:** daytime, bright natural light. **Session B:** evening, room LED lights on, different day or at least a different hour. Different lighting is the point: I train on one session and test on the other, so a good result means something.
- **A second person** doing the same takes (call them `p2`) is the single most valuable addition, because DVS128 Gesture has 29 subjects and one-person data cannot show whether the network works on other people. Optional, but if you can get 20 minutes of a friend, do it.

**Each take is its own file, about 10 seconds:** press record, stand still with arms down for 1 second, perform the gesture **continuously for about 7 seconds** (the real samples last 5 to 8 s, *measured* mean 6.5 s), arms down for 1 second, stop. No pauses in the middle.

---

## 2. Phone and app settings

Do this once, then do not touch it between takes (re-check only if the light changes a lot).

| Setting | Value | Why |
|---|---|---|
| Camera / lens | Rear **main (1x)** lens. No ultra-wide, no zoom. | Ultra-wide distorts; digital zoom softens. |
| Orientation | **Landscape**, phone on a **tripod or fixed stand**. Never handheld. | A moving camera makes events everywhere. |
| Resolution | 1920x1080 (HD). 4K also works but gives no benefit. | The pipeline shrinks to 128x128 anyway. |
| Frame rate | **30 fps** | The whole system is built on 30 fps frames. 60 fps is accepted (I drop every other frame); avoid 24 and 25. |
| Codec and quality | H.264 if offered (HEVC is probably fine, the checker will tell you), **highest bitrate / quality** setting. | Compression blocks create false events. |
| Colour profile | **Standard SDR / Rec. 709. No Log, no HDR, no LUT.** | Log and HDR flatten contrast and change the brightness-to-event relationship. |
| Shutter | **1/100 s** (if the app shows shutter angle instead: about 108 degrees at 30 fps) | Your mains is probably 50 Hz (your PC clock is on IST), which makes LED and fluorescent light flicker at 100 Hz; 1/100 s cancels it. Avoid 1/60 and 1/120 indoors. |
| ISO | **Manual**, as low as possible while the wall is bright and not clipped; aim for ISO 800 or lower. If it needs more, add light instead. | Sensor noise in dark areas becomes false events. |
| White balance | **Manual** Kelvin: about 5500 K for daylight, about 4000 to 5000 K for your LED bulbs; fixed. | Auto white balance drifts and changes every pixel. |
| Focus | **Manual**, set on your chest at the distance you will stand; check with focus peaking if the app has it. | Autofocus hunting changes the whole image. |
| Exposure | **Auto exposure off** (manual ISO and shutter, no exposure lock tricks). Watch the histogram: the wall in the middle to bright part, nothing clipped; zebras, if available, must not show on your skin or the wall. | Auto exposure changes brightness while you move, creating events everywhere. |
| Stabilisation | **Off** if the app has an electronic stabilisation option. | It warps and crops the image. |

---

## 3. Scene and framing

- **Background:** one plain, matte, light or mid-tone wall, no pattern, no posters, no window in the shot (a window with daylight behind you turns you into a silhouette and its light changes). Nothing may move behind you: no fan, TV, curtains, plants, pets, other people.
- **Clothes:** plain solid-colour **long-sleeve** top that contrasts clearly with the wall (dark top on a light wall works best); sleeves that fit, nothing flapping, no watch or bracelet that flashes. The emulator only reacts to brightness changes of about 20%, so a dark arm in front of a dark wall produces nothing.
- **Light:** bright, even, from the front or the sides. Not from behind you. No strong shadows moving across the wall.
- **Camera position:** about **1.5 to 2 m** from you, lens at chest height, pointing level. Start at 1.8 m and adjust.
- **Framing (the important part):** the system uses only the **middle square** of the frame (width equal to the height, i.e. the middle 56% of the width of a 16:9 video) and shrinks it to 128x128. So frame yourself **waist-up, body centred, head in the upper third**, and make sure every gesture stays inside the middle square with some margin. In the real recordings (*measured*) the motion is centred at about 45% of the width and the hands work at 53 to 66% of the height. If the app has a 1:1 (square) frame guide, turn it on; otherwise put two bits of tape on the wall at the edges of the middle square.
- Look at the reference images first: `docs/results/dvs_reference/gestures_1p6s.png` and `sequence_67ms_steps.png` (green = brightness increase, red = decrease). Your gesture should fill the picture about like that.

---

## 4. Do these two test clips first (10 minutes)

Record one take of `00_hand_clapping` and one of `05_left_arm_cw`, copy them to the PC, and run

```
.venv\Scripts\python.exe sw\check_clip.py data\own_clips
```

It prints PASS or a list of WARN items for each clip (frame rate, resolution, brightness drift = auto exposure/white balance still on, flicker, dark image, low contrast = Log/HDR, background noise, gesture too short or with pauses, motion off-centre or touching the crop edge) and writes a contact sheet to `work\clipcheck\<name>.png` (top row your frames, bottom row the events: green brighter, red darker). Look at the PNG: the bottom row should show clean moving shapes on black.

**The warning thresholds were tuned on synthetic clips only, I have no real phone footage yet.** If the checker warns about a clip that looks right to you, or passes one that looks wrong, send me the output and the PNG and I will fix the thresholds before you record the rest. Also useful to me: a screenshot of the app's settings screen and your phone model.

---

## 5. Files

Copy the **original files from the phone** (USB cable, AirDrop or a cloud drive set to original quality; **not** WhatsApp, Telegram or any messenger, they recompress and change the frame rate). Do not trim, convert or edit them. Put them here, with the exact folder names from section 1:

```
data\own_clips\00_hand_clapping\A_p1_01.mov      A = session A (daylight), B = session B (LED)
data\own_clips\00_hand_clapping\A_p1_02.mov      p1 = you, p2 = second person
data\own_clips\01_right_hand_wave\A_p1_01.mov    01, 02, ... = take number
...
```

The folder prefix is the DVS128 class number and **is** the label; the loader (`train/eval_int.py --dataset own`) refuses folders without it. Any video extension (.mov, .mp4, .m4v, .avi, .mkv) is read. Then run the checker on the whole folder and re-shoot whatever it flags.

---

## 6. What I do with them

1. Run the Track A network (82.7% window accuracy on real events) on your clips through the emulator, per session and per person: the domain-gap number (3.10).
2. If it drops a lot: fine-tune on session A, test on session B (and on p2 if you have one), keeping the whole-clip split (3.11).
3. Compare the statistics of your emulated events with the real dataset (event rate, motion centre, repetition rate) and with v2e if you install it (2.6).
4. Use the best clips for the board demo.
