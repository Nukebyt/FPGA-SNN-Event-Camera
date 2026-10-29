"""Regression runner: Icarus testbenches against the Python golden models, plus pytest.

Verdict rules (checklist 3.9): a bench passes only if its output has a "RESULT: PASS" line and no
"RESULT: FAIL" / "MISMATCH" / "FAIL" line. Compile error, timeout or no verdict line is FAIL; the
simulator exit code is never the signal.

--selftest re-runs the benches against deliberately broken copies of the RTL and requires FAIL.
Run it whenever this script or a testbench changes.

Usage:  python scripts/run_regression.py [--selftest] [--skip-python] [--only NAME]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.gen_emulator_vectors import SCENARIOS, write_scenario  # noqa: E402

IVERILOG = os.environ.get("IVERILOG", "iverilog")
VVP = os.environ.get("VVP", "vvp")

EMU_RTL = ["rtl/sdp_ram.v", "rtl/log_lut.v", "rtl/event_fifo.v", "rtl/event_emulator.v"]
EMU_TB = "tb/event_emulator_tb.v"

# (file, old text, new text) -- every one of these must make at least one scenario FAIL
MUTATIONS = [
    ("rtl/event_emulator.v", "(d >=  th)", "(d >   th)"),
    ("rtl/event_emulator.v", "(d <= -th)", "(d <  -th)"),
    ("rtl/event_emulator.v", "(cfg_mode_reset ? s2_cur : ref_step)", "(cfg_mode_reset ? ref_step : s2_cur)"),
    ("rtl/event_emulator.v", "{s2_on, s2_y, s2_x, s2_ts}", "{s2_on, s2_x, s2_y, s2_ts}"),
    ("rtl/event_emulator.v", "ts_cur + cfg_period", "ts_cur + cfg_period + 17'd1"),
    ("rtl/event_emulator.v", "- 8)", "- 0)"),                                  # no FIFO headroom -> overflow
    ("rtl/event_emulator.v", "ref_we   = s2_valid && (s2_init || ev_new)", "ref_we   = s2_valid && ev_new"),
    ("rtl/event_fifo.v", "wp <= wp + 1'b1;", "wp <= wp + 2'd2;"),
    ("rtl/event_fifo.v", "(!rd_valid || rd_ready)", "(!rd_valid)"),            # halves throughput but still correct -> must NOT be flagged
    # VGA demo path: caught only by the pixel-exact demo check
    ("rtl/event_view.v", "r <= 8'd0;   g <= 8'd255; b <= 8'd0;", "r <= 8'd0;   g <= 8'd254; b <= 8'd0;"),
    ("rtl/event_view.v", "8'd171", "8'd170"),
    ("rtl/event_view.v", "wbank      <= ~wbank;", "wbank      <= wbank;"),
    ("rtl/pix_source.v", "sx <= sx + 7'd3;", "sx <= sx + 7'd4;"),
]
EXPECT_CAUGHT = [True] * 8 + [False] + [True] * 4
DEMO_MUTATION_FILES = ("rtl/event_view.v", "rtl/pix_source.v")


def verdict(text):
    lines = text.splitlines()
    has_pass = any(re.match(r"\s*RESULT:\s*PASS", l) for l in lines)
    bad = any(re.match(r"\s*RESULT:\s*FAIL", l) or "MISMATCH" in l or re.match(r"\s*FAIL", l) for l in lines)
    return has_pass and not bad


def run_scenario(name, base):
    vec = ROOT / "work" / "vec" / name
    write_scenario(name, vec)
    exe = ROOT / "work" / ("%s_%s.vvp" % (base.name, name))
    cmd = [IVERILOG, "-g2012", "-Wall", "-I", str(vec), "-o", str(exe)] + [str(base / s) for s in EMU_RTL] + [str(ROOT / EMU_TB)]
    c = subprocess.run(cmd, capture_output=True, text=True, cwd=base if base != ROOT else ROOT)
    if c.returncode != 0:
        return False, "COMPILE ERROR\n" + c.stdout + c.stderr
    try:
        r = subprocess.run([VVP, str(exe)], capture_output=True, text=True, timeout=900, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return verdict(r.stdout + r.stderr), r.stdout + r.stderr


DEMO_RTL = ["tb/pll25_sim.v", "rtl/demo_top.v", "rtl/pix_source.v", "rtl/event_view.v", "rtl/event_emulator.v",
            "rtl/event_fifo.v", "rtl/sdp_ram.v", "rtl/log_lut.v", "rtl/vga_timing.v", "rtl/reset_sync.v", "rtl/hex7seg.v"]
# name -> (SW value, theta = 2*SW[4:0]+1, mode, number of VGA frames dumped)
DEMOS = {
    "demo_step_w1":  ("10'b0000000100", 9, "step", 6),
    "demo_reset_w4": ("10'b0010100111", 15, "reset", 5),
}


def run_demo(name, base=ROOT):
    sw, theta, mode, nd = DEMOS[name]
    d = ROOT / "work" / (name if base == ROOT else base.name + "_" + name)
    d.mkdir(parents=True, exist_ok=True)
    exe = ROOT / "work" / (d.name + ".vvp")
    cmd = [IVERILOG, "-g2012", "-Wall", "-DSWVAL=" + sw, '-DDUMP_DIR="%s"' % d.as_posix(), "-DNDUMP=%d" % nd,
           "-o", str(exe), str(ROOT / "tb/demo_top_tb.v")] + [str(base / s) if s.startswith("rtl/") else str(ROOT / s) for s in DEMO_RTL]
    c = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if c.returncode != 0:
        return False, "COMPILE ERROR\n" + c.stdout + c.stderr
    r = subprocess.run([VVP, str(exe)], capture_output=True, text=True, timeout=1800, cwd=ROOT)
    (d / "sim.log").write_text(r.stdout + r.stderr)
    if not verdict(r.stdout + r.stderr):
        return False, r.stdout + r.stderr
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    k = subprocess.run([str(py) if py.exists() else sys.executable, str(ROOT / "scripts/check_demo_vga.py"), "--dir", str(d),
                        "--log", str(d / "sim.log"), "--theta", str(theta), "--mode", mode], capture_output=True, text=True)
    return verdict(k.stdout), k.stdout + k.stderr


def run_all(skip_python, only, skip_demo=False):
    ok = True
    (ROOT / "work").mkdir(exist_ok=True)
    for name in sorted(SCENARIOS):
        if only and name != only:
            continue
        passed, text = run_scenario(name, ROOT)
        info = [l for l in text.splitlines() if l.startswith("info:")]
        print(("PASS  " if passed else "FAIL  ") + "event_emulator/" + name + "   " + (info[-1][6:] if info else "(no info line)"))
        if not passed:
            ok = False
            print(text[-2500:])
    if not skip_demo and not only:
        for name in DEMOS:
            passed, text = run_demo(name)
            last = [l for l in text.splitlines() if "mismatching pixels" in l]
            print(("PASS  " if passed else "FAIL  ") + "vga/" + name + "   (%d VGA frames pixel-exact vs model)" % len(last))
            if not passed:
                ok = False
                print(text[-2500:])
    if not skip_python:
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        exe = str(py) if py.exists() else sys.executable
        r = subprocess.run([exe, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        passed = r.returncode == 0 and " passed" in tail and "failed" not in tail and "error" not in tail
        print(("PASS  " if passed else "FAIL  ") + "pytest   " + tail)
        if not passed:
            ok = False
            print(r.stdout[-2500:])
    return ok


def selftest():
    print("selftest: broken RTL variants must make at least one scenario FAIL")
    good = True
    for i, ((rel, old, new), want) in enumerate(zip(MUTATIONS, EXPECT_CAUGHT)):
        mdir = ROOT / "work" / ("mut%d" % i)
        if mdir.exists():
            shutil.rmtree(mdir)
        (mdir / "rtl").mkdir(parents=True)
        for f in [q for q in (ROOT / "rtl").glob("*") if q.is_file()]:
            shutil.copy(f, mdir / "rtl" / f.name)
        p = mdir / rel
        src = p.read_text()
        if old not in src:
            print("  SELFTEST ERROR: pattern not found: %r in %s" % (old, rel))
            good = False
            continue
        p.write_text(src.replace(old, new, 1))
        caught = False
        if rel in DEMO_MUTATION_FILES:
            passed, _ = run_demo("demo_step_w1", mdir)
            caught = not passed
        else:
            for name in sorted(SCENARIOS):
                passed, _ = run_scenario(name, mdir)
                if not passed:
                    caught = True
                    break
        status = "CAUGHT" if caught else "MISSED"
        okk = caught == want
        print("  %s  %s: %s -> %s%s" % (status, rel, old, new, "" if okk else "   <-- UNEXPECTED"))
        good = good and okk
    assert verdict("") is False and verdict("hello") is False and verdict("RESULT: PASS\nMISMATCH x") is False
    print("  verdict(): empty / no-verdict / PASS+MISMATCH -> FAIL ok")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--skip-demo", action="store_true", help="skip the slow (~1 min each) VGA demo simulations")
    ap.add_argument("--snn-top", action="store_true", help="also simulate the SNN board top level (3 runs, ~6 min each)")
    a = ap.parse_args()
    from scripts import snn_regression
    ok = run_all(a.skip_python, a.only, a.skip_demo)
    if not a.only:
        ok = snn_regression.run_all() and ok
        if a.snn_top:
            ok = snn_regression.run_tops() and ok
    if a.selftest:
        ok = selftest() and ok
        ok = snn_regression.selftest() and ok
    print("OVERALL: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
