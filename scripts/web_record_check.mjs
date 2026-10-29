// Browser test of "Record training clips" in the live demo page: headless Edge/Chrome with Chromium's fake camera, driven over the
// DevTools protocol (Node 22+, no packages).
//
//   python scripts/snn_web.py --no-browser --port 8124 --recordings work/rec_test     (in another terminal)
//   node scripts/web_record_check.mjs --port 8124 [--browser "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]
//
// Checks: a fixed-length recording with the countdown, "until stopped" + Stop, Cancel during the countdown, the subject check,
// recording while the classifier's Webcam source runs (Model backend), Play of a recording, Delete (to _trash), and screenshots.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const PORT = arg("--port", "8124");
const BROWSER = arg("--browser", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe");
const SHOTS = arg("--shots", "work/rec_shots");
const DBG = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const check = (ok, what) => { console.log((ok ? "PASS " : "FAIL ") + what); if (!ok) failures++; };

const prof = mkdtempSync(join(tmpdir(), "rec-edge-"));
const br = spawn(BROWSER, ["--headless=new", `--remote-debugging-port=${DBG}`, `--user-data-dir=${prof}`, "--use-fake-device-for-media-stream",
  "--use-fake-ui-for-media-stream", "--window-size=1280,2000", "--no-first-run", "about:blank"], { stdio: "ignore" });

let ws, nextId = 1;
const pending = new Map();
function cdp(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((res, rej) => pending.set(id, { res, rej }));
}
async function js(expr) {
  const r = await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error("page error: " + JSON.stringify(r.exceptionDetails).slice(0, 400));
  return r.result.value;
}
async function waitFor(expr, ms, what) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    const v = await js(expr);
    if (v) return v;
    await sleep(100);
  }
  throw new Error("timeout waiting for " + what);
}
async function shot(name) {
  const r = await cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  writeFileSync(join(SHOTS, name), Buffer.from(r.data, "base64"));
}
const status = () => js(`document.getElementById("recStatus").textContent`);
const api = async (path) => (await fetch(`http://127.0.0.1:${PORT}${path}`)).json();

try {
  mkdirSync(SHOTS, { recursive: true });
  let targets;
  for (let k = 0; k < 50; k++) {
    try { targets = await (await fetch(`http://127.0.0.1:${DBG}/json/list`)).json(); break; } catch (e) { await sleep(200); }
  }
  const page = targets.find((t) => t.type === "page");
  ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(m.error.message)) : p.res(m.result); }
  };
  await cdp("Page.enable");
  await cdp("Runtime.enable");
  await cdp("Page.navigate", { url: `http://127.0.0.1:${PORT}/` });
  await waitFor(`document.readyState === "complete" && !!document.getElementById("recBtn")`, 10000, "page");
  await waitFor(`document.getElementById("sseText").textContent === "live"`, 10000, "SSE");
  const before = (await api("/api/recordings")).recordings.length;

  // 1. subject check
  await js(`document.getElementById("recSubject").value = ""; document.getElementById("recBtn").click(); true`);
  check(/Enter a subject/.test(await status()), "empty subject is refused: " + (await status()));

  // 2. fixed length (5 s) with the 3 s countdown
  await js(`(() => { const $ = (i) => document.getElementById(i); $("recSubject").value = "fake cam"; $("recAction").value = "walking";
            $("recLen").value = "5"; $("recCountdown").checked = true; $("recNote").value = "browser test"; $("recBtn").click(); return true; })()`);
  await sleep(1500);
  const over = await js(`document.getElementById("recOver").textContent`);
  check(/^[123]$/.test(over), "countdown shows a digit: " + over);
  await shot("rec_countdown.png");
  await waitFor(`document.getElementById("recOver").textContent.includes("REC")`, 4000, "recording to start");
  await sleep(2000);
  await shot("rec_recording.png");
  check(await js(`document.getElementById("recBtn").textContent.includes("Stop")`), "button turns into Stop while recording");
  await waitFor(`/Saved|Not saved|failed|nothing saved/.test(document.getElementById("recStatus").textContent)`, 15000, "save");
  const s1 = await status();
  check(/^Saved fake-cam_walking_/.test(s1), "5 s clip saved, subject spaces turned into '-': " + s1);
  await sleep(500);
  await shot("rec_saved.png");
  let recs = (await api("/api/recordings")).recordings;
  const r1 = recs[0];
  check(recs.length === before + 1 && r1.action === "walking" && r1.subject === "fake-cam", "listed by the server");
  check(Math.abs(r1.duration_s - 5) < 0.25, "length about 5 s: " + r1.duration_s);
  console.log(`info: fake camera ${r1.fps} frames/s, ${r1.frames} frames, warnings: ${JSON.stringify(r1.warnings)}`);
  const rows = await js(`document.querySelectorAll("#recList tbody tr").length`);
  check(rows === recs.length, "table shows every recording: " + rows);
  check(await js(`document.getElementById("recCounts").textContent.includes("walking ")`), "counts per action shown");
  check(await js(`localStorage.getItem("snnRecSubject") === "fake-cam"`), "subject remembered");

  // 3. "until stopped", Stop after about 2 s
  await js(`(() => { const $ = (i) => document.getElementById(i); $("recAction").value = "none"; $("recLen").value = "0"; $("recCountdown").checked = false; $("recBtn").click(); return true; })()`);
  await waitFor(`document.getElementById("recOver").textContent.includes("REC")`, 4000, "recording 2");
  await sleep(2000);
  await js(`document.getElementById("recBtn").click(); true`);
  await waitFor(`/Saved|Not saved|failed|nothing saved/.test(document.getElementById("recStatus").textContent)`, 15000, "save 2");
  const s2 = await status();
  check(/^Saved fake-cam_none_/.test(s2), "until-stopped clip saved after Stop: " + s2);
  recs = (await api("/api/recordings")).recordings;
  check(recs.length === before + 2 && recs[0].duration_s > 1.5 && recs[0].duration_s < 3.5, "its length is the time until Stop: " + recs[0].duration_s);

  // 4. Cancel during the countdown saves nothing
  await js(`(() => { const $ = (i) => document.getElementById(i); $("recLen").value = "5"; $("recCountdown").checked = true; $("recBtn").click(); return true; })()`);
  await sleep(1200);
  await js(`document.getElementById("recBtn").click(); true`);
  await waitFor(`document.getElementById("recStatus").textContent === "Cancelled."`, 4000, "cancel");
  check((await api("/api/recordings")).recordings.length === before + 2, "cancel during the countdown saves nothing");

  // 5. record while the classifier's Webcam source runs on the Model backend
  await js(`document.querySelector("input[name=backend][value=sim]").checked = true; document.getElementById("camBtn").click(); true`);
  await waitFor(`document.getElementById("camBtn").textContent === "Webcam on"`, 8000, "webcam source");
  await waitFor(`document.getElementById("sIdx").textContent !== "–"`, 8000, "classifier frames");
  await js(`(() => { const $ = (i) => document.getElementById(i); $("recAction").value = "boxing"; $("recLen").value = "5"; $("recCountdown").checked = false; $("recBtn").click(); return true; })()`);
  await waitFor(`/Saved|Not saved|failed|nothing saved/.test(document.getElementById("recStatus").textContent)`, 20000, "save 3");
  const idxA = Number(await js(`document.getElementById("sIdx").textContent`));
  await sleep(1000);
  const idxB = Number(await js(`document.getElementById("sIdx").textContent`));
  check(/^Saved fake-cam_boxing_/.test(await status()), "recorded while the classifier ran: " + (await status()));
  check(idxB > idxA, `classifier kept running during and after the recording (frame ${idxA} -> ${idxB})`);
  await shot("rec_with_classifier.png");
  await js(`document.getElementById("stopBtn").click(); true`);
  await waitFor(`document.getElementById("camBtn").textContent === "Webcam"`, 5000, "webcam stop");
  check(await js(`document.getElementById("recVideo").srcObject !== null && document.getElementById("recVideo").srcObject.active`),
        "stopping the classifier leaves the recorder's camera open");

  // 6. Play a recording (Model backend) and wait for its clip answer
  recs = (await api("/api/recordings")).recordings;
  await js(`document.querySelector("#recList tbody tr button").click(); true`);
  await waitFor(`document.getElementById("clipBanner").classList.contains("show")`, 30000, "clip answer");
  const ans = await js(`document.getElementById("clipAns").textContent + " | " + document.getElementById("clipHow").textContent`);
  check(ans.includes(recs[0].file), "the newest recording plays and gets a clip answer: " + ans);

  // 7. Delete the newest recording (confirm answered yes)
  const gone = recs[0].file;
  await js(`window.confirm = () => true; document.querySelector("#recList tbody tr button:nth-of-type(2)").click(); true`);
  await waitFor(`!document.getElementById("recList").textContent.includes(${JSON.stringify(gone)})`, 5000, "delete");
  const after = (await api("/api/recordings")).recordings;
  check(after.length === recs.length - 1 && !after.some((r) => r.file === gone), "Delete removes it from the list (moved to _trash)");

  // 8. close the camera: tracks stopped
  await js(`document.getElementById("recCamBtn").click(); true`);
  check(await js(`document.getElementById("recVideo").srcObject === null && document.getElementById("recCamBtn").textContent === "Open camera"`), "Close camera");
} catch (e) {
  console.log("FAIL " + e.message);
  failures++;
} finally {
  try { ws && ws.close(); } catch (e) { /* closed */ }
  br.kill();
}
console.log(failures ? `${failures} check(s) failed` : "all checks passed");
process.exit(failures ? 1 : 0);
