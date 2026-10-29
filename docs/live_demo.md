# Live demo in the browser (`scripts/snn_web.py` + `sw/web/index.html`)

Play sample clips, upload your own video or use your webcam, and watch the FPGA classify it frame by frame.

![the live demo page while the board classifies a boxing clip](media/live_demo.gif)

*Recorded from the page (headless Edge) while the real board classified `person22_boxing_d1` at 30 frames/s: input frame, the DVS events of that frame, the running scores of the six classes, the window answers, and the live check of the board's answers against the bit-accurate model (362 / 362 frames identical). The last three windows say "walking" because the person walks out of the picture at the end of the clip; the clip answer (sum of the window scores) is still boxing.*

## Start it
```
.venv\Scripts\python.exe scripts\snn_web.py --board 192.168.137.200      # the board answers; opens http://127.0.0.1:8080
.venv\Scripts\python.exe scripts\snn_web.py                              # no hardware: the bit-accurate Python models answer
```
Options: `--port 8090`, `--no-browser`, `--max-upload-mb 500` (default 200), `--bind 0.0.0.0` (let a phone on the same network open the page; see the webcam notes). Only the Python standard library, OpenCV and NumPy are needed, and the page loads nothing from the internet.

The board must be running `sw/snn_streamer.c` (`docs/snn_soc.md`, bring-up step 4). The demo opens the board connection when a source starts and closes it when the source stops, so `scripts/check_board.py` or `snn_viewer.py` can use the board between two runs. The daemon serves one client at a time.

## Controls
| Control | What it does |
|---|---|
| Sample clip + **Play clip** | KTH clips of two test people (persons 22 and 02, never seen in training), all six actions and four conditions, with the true action in the list. Also lists `data/user_clips/` and earlier uploads. |
| **Choose file** + **Upload & play** | Sends any video file OpenCV can decode (AVI, MP4, MOV, MKV, WEBM, ...) to the server (`work/uploads/`) and plays it. Files over the size limit are refused before they are sent. |
| **Webcam** | Uses the browser camera: each picture is centre-cropped to 4:3 (the KTH shape), scaled to 320x240, sent as JPEG, and the next one follows when the server has answered, at most 30 per second. Press it again (or Stop) to end. |
| **Board / Model** + host | Who answers: the FPGA at `host[:port]` (default port 5002) or the bit-accurate Python models. |
| **check every board answer against the model** | Board only: every frame also goes through the model, and the page counts identical frames ("board == model: N / N frames match"). Every field of the RESULT is compared: spike counts, events, currents, running and window scores, window class and counters. The first difference, if any, is shown. |
| **loop** | Replays the clip; before each pass the board engine, the model and the event picture are reset, so every pass gives identical answers. |
| **fast** | Unpaced: as fast as the backend answers (about 70 frames/s with the board, several hundred with the model). The page then shows at most about 45 pictures per second, and every finished window is always shown. |
| **Stop** | Stops the source and releases the board. |

Backend, loop and fast apply to the next start. If the board cannot be reached (wrong address, daemon not running, another client connected), the error is shown with a **Use the Model backend instead** button that restarts the same source on the model.

## What the page shows
- **Input frame**: the 128x128 grey frame the board receives, after the same preparation as the training data: grey, the whole frame squashed with area averaging, sources that are not 30 fps blended to 30 fps with the training arithmetic.
- **DVS events of this frame**: green = ON (brighter), red = OFF (darker). They are drawn on the laptop by the same bit-exact emulator model, because the board returns counts, not pictures.
- **Running window score**: the six class scores of the current 32-frame window (summed class currents). The class ahead is highlighted. With all scores still at 0, nothing is highlighted, although the board's tie rule would name class 0.
- **Answer of the last finished window** (the board's `PRED`), the step in the window, and the history of window answers.
- **Counters**: step n / 32, finished windows, events and spikes (layer 1 / layer 2) of this frame, FPGA latency (DMA start to result from the engine's cycle counter, 8.89 ms), round trip (laptop: frame sent to answer received, about 13 to 15 ms), measured frames/s over the last 30 frames, frame n / total.
- **Clip answer** at the end of a clip: the class with the largest sum of the finished windows' scores (the rule of `snn_viewer.py` and of the per-clip accuracy), compared with the true action when it is known. A clip shorter than one window uses its running score.

## Webcam notes (what to expect)
- The network knows six KTH actions and always answers with one of them. KTH is a fixed camera, a whole person at some distance, and a plain background. Stand back so your whole body is in the picture, keep the camera still, and use a plain wall behind you. A face filling the picture is outside anything the network has seen.
- Webcam pictures are JPEG compressed in the browser, so they are noisier than a decoded video file, and compression flicker becomes events. Measured with the KTH clips sent the way the page sends webcam pictures (original frames at 30 fps, scaled to 320x240, JPEG quality 85, decoded and squashed on the server): on 12 clips (person 22, all actions, d1 and d3), the clip answer was the same as from the file in **12 / 12** cases and the window answer in **192 / 224** windows (86 %). This used OpenCV's JPEG encoder, not a browser's; a real webcam adds sensor noise and auto-exposure changes on top.
- One frame is one network step, whatever its arrival time. A webcam that delivers fewer than 30 frames/s (the "measured fps" counter shows it) moves further per step than the training video did. With the board, one webcam frame took 22 ms per request including JPEG decoding, so the server is not the limit.
- Browsers allow the camera only on `http://127.0.0.1` / `localhost` or HTTPS. With `--bind 0.0.0.0` a phone can open `http://<laptop-ip>:8080`, play clips and upload a video recorded on the phone, but not use its camera over plain HTTP.

## Sessions, tabs and releasing the board
- One session per server: every open tab sees the same stream, and the last button pressed in any tab wins. A webcam session has a token; when another tab starts a source, the first tab's camera stops with a message.
- A webcam that sends no frame for 6 s (tab closed or hidden) is stopped, and so is any source with no page open for 15 s, so a forgotten tab cannot hold the board. Closing the page with the webcam on also sends a stop right away.
- If the board is busy with another client, starting fails after 8 s with a message saying so; an unreachable address fails after 3 s.

## How it works
```
browser (sw/web/index.html)                 scripts/snn_web.py (laptop)                       board
  GET /api/clips, POST /api/start  ------>  clip thread: decode, prepare, 30 fps pacing
  POST /api/webcam/frame (JPEG)    ------>  prepare (squash 128x128)  --FRAME_IN-->  sw/snn_streamer.c -> FPGA
  EventSource /stream  <---- SSE: status, frame, clip_done, error <--  RESULT (112 B)  <--
                                            + optional SimBoard (bit-accurate model) for the live check
```
| Endpoint | |
|---|---|
| `GET /` | the page |
| `GET /api/clips` | clip list, default backend and host, current status, upload limit |
| `POST /api/start` | `{path, backend, host, verify, loop, fast}`; only listed clips can be played |
| `POST /api/stop` | stop the current source |
| `POST /api/upload` | raw file body, `X-Filename` header; size-limited, name sanitised, rejected if OpenCV cannot decode it |
| `POST /api/webcam/start` | `{backend, host, verify}` -> `{ok, token}` (or `{ok: false, msg}` if the board cannot be reached) |
| `POST /api/webcam/frame` | one JPEG, `X-Session: token`; 409 when the session is no longer active |
| `GET /stream` | Server-Sent Events; the page reconnects by itself if the server goes away |
| `POST /api/record` | a recording: uint32 LE header length, JSON header (`action`, `subject`, `note`, `sizes`, `stamps_ms`, `missed`, `timing`, `camera`), then the JPEG frames; saved as AVI + JSON, 400 with a reason for bad input |
| `GET /api/recordings` | the saved recordings (newest first) and counts per action |
| `POST /api/recordings/delete` | `{path}`; moves the recording to `_trash/`; 409 if the file is in use (playing) |

Each source is a `Run` that owns its backend, so a stopping source can never close the next one's board connection. Each frame's send-and-receive holds the run's lock, so two requests can never interleave on the board's single TCP connection (bug log 34 to 36 lists what the first draft got wrong).

## What was verified (2026-10-29)
| Test | Result |
|---|---|
| API driver against the Model backend (clips, loop, stop, upload, bad requests, webcam sessions, two tabs, missing board, watchdogs) | all checks pass; clip answers and window totals equal an independent model run |
| Same driver against the **real board** | board == model on every frame: 362 / 362 (boxing), 1,236 / 1,236 (three loop passes, identical totals per pass), 64 / 64 (webcam JPEGs); FPGA latency 8.89 ms, round trip 13 to 15 ms, paced 30.0 frames/s, fast about 70 frames/s |
| The page itself in headless Edge, Board backend | `person22_handwaving_d1`: 899 / 899 frames match, clip answer hand waving (correct); `person22_boxing_d1`: 362 / 362, boxing (correct) |
| Releasing the board | `check_board.py --frames 40` right after Stop: PASS (0 differ); with another client holding the daemon, the demo reports it after 8 s and works once that client has gone |
| After a reset pressed on the board (Linux kept running, daemon unchanged) | STATUS `0x1`, IN_RD_SEQ = DONE_SEQ; `check_board.py --frames 200` PASS, and the demo checks again 362 / 362, 1,236 / 1,236, 64 / 64 |
| JavaScript | `node --check` on the extracted script |

**Not verified:** a real webcam through `getUserMedia` (only the server side of the webcam path was tested, with JPEGs made from KTH clips), the page's upload and webcam buttons in a real browser (both endpoints were tested from a script), browsers other than Edge, and a phone.

**Filmed on the real setup (2026-10-29):** the page playing the phone clip `hand_waving1_crop.mp4` on the board: board == model 462 / 462 frames, every window hand waving, FPGA latency 8.89 ms, round trip 14 ms, 30 frames/s; the board's 7-segment display agrees with the page (README, top GIF).

## Recording training clips
The "Record training clips (webcam)" panel saves labelled clips for later training; it does not need the board or the classifier, and it can run while the Webcam source is classifying (both use one camera stream).

- **What is saved**: every frame the camera delivers (`requestVideoFrameCallback`; a 30 frames/s timer in browsers without it), centre-cropped to 4:3, at most 640x480, browser JPEG quality 0.92, written as MJPG AVI at the measured frame rate (OpenCV's FFmpeg writer keeps a rate such as 29.7; its own MJPEG writer would round it to 30). The sidecar JSON holds the action, `label` (index in the KTH order, `null` for none), subject, note, time, frames, duration, measured fps, frame-interval median and maximum, frames the page missed, the timing source, the camera and the capture time of every frame (`stamps_ms`).
- **Timing**: the camera's own frame times (`mediaTime`) are used when they rise and span the same time as the page clock; otherwise the times the page received the frames.
- **Folders**: `data/recordings/<action>/<subject>_<action>_<YYYYmmdd-HHMMSS>.avi` + `.json`; a file being written is named `_writing_...` and hidden from all lists; Delete moves to `data/recordings/_trash/`. `--recordings DIR` puts them elsewhere (used by the tests).
- **Checks on the server**: action one of the six KTH folders or `none`; subject 1-32 letters, digits or `-` (no `_`, which separates the parts of the file name); at least 15 frames; frame times finite and in order; at most 60 s; frame sizes add up to the body; every JPEG decodes and all have one size; the written file is read back and must hold every frame.
- **Warnings** (shown in the page and stored): fewer than 25 frames/s, a gap of more than three frame intervals, frames the page could not grab.
- **For training**: `train/recordings.py` (`load_recordings`, `resample_times`) gives 128x128 grey frames at exactly 30 fps from the capture times, prepared like the KTH network's training data; `train.prep_kth` does not read recordings yet.

| Test | Result |
|---|---|
| `sw/tests/test_web_record.py` (13 tests) | save + list + playable in the demo + loader at exact 30 fps + trash; `none` class and warnings; 9 kinds of bad input refused with nothing written; loader equals the KTH preparation; resampler equals the KTH blending for an even rate (to 1 grey level on exact half-way blends, bug 40) |
| `scripts/web_record_check.mjs` (headless Edge, Chromium's fake camera, Model backend) | empty subject refused; 5 s with countdown saved (100 frames at the fake camera's 20 frames/s, 4.96 s, warning shown); until-stopped + Stop after 2 s (2.05 s); cancel during the countdown saves nothing; recording while the Webcam source classifies (classifier frames kept counting); stopping the classifier keeps the recorder's camera; Play of a recording gives a clip answer; Delete moves it to `_trash`; Close camera stops it. Camera frame times were used; no identical consecutive frames in the files |
| Not verified | a real webcam, browsers other than Edge, a training run on recordings |
