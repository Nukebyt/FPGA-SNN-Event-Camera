"""SNN RTL regression (called by scripts/run_regression.py): bit-exactness of snn_core / bin_builder / snn_ctrl / frame_sync
against model/snn_int.py, on random and saturating networks, the whole emulator-to-SNN chain, and (when exported) the real network.

Cases (vectors from scripts/gen_snn_vectors.py into work/vec/):
  snn_rand   random net, 12% dense bins              tb/snn_core_tb.v   (bin builder set_* path, core, controller)
  snn_sat    saturating net, 70% dense bins, 8 steps tb/snn_core_tb.v   (membrane clip at both rails, int8 extremes)
  snn_chain  random net, pattern source frames       tb/snn_chain_tb.v  (emulator -> frame_sync -> builder -> core)
  snn_kth6   exported KTH network, a real sample     tb/snn_core_tb.v   (only if rtl/weights/kth6/ holds it)
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IVERILOG = os.environ.get("IVERILOG", "iverilog")
VVP = os.environ.get("VVP", "vvp")
CORE_RTL = ["rtl/sdp_ram.v", "rtl/snn_core.v", "rtl/lif_conv.v", "rtl/bin_builder.v", "rtl/snn_ctrl.v"]
CHAIN_RTL = CORE_RTL + ["rtl/pix_source.v", "rtl/event_emulator.v", "rtl/event_fifo.v", "rtl/log_lut.v", "rtl/frame_sync.v"]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _gen(args):
    r = subprocess.run([str(PY) if PY.exists() else sys.executable, str(ROOT / "scripts/gen_snn_vectors.py")] + args,
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError("vector generation failed: " + r.stderr[-1500:])
    return r.stdout


ENGINE_RTL = CHAIN_RTL + ["rtl/snn_engine.v", "rtl/dma_rd.v", "rtl/fifo_64to8.v", "rtl/ram_sdp.v"]   # CHAIN_RTL has the rest

CASES = {
    "snn_rand":  (["random", "--seed", "1", "--steps", "6"], "tb/snn_core_tb.v", CORE_RTL),
    "snn_sat":   (["random", "--seed", "2", "--steps", "8", "--sat"], "tb/snn_core_tb.v", CORE_RTL),
    "snn_chain": (["chain", "--seed", "3", "--frames", "5"], "tb/snn_chain_tb.v", CHAIN_RTL),
    # the engine behind the HPS: DMA from a DDR3 model with wait states, CSR snapshots, 3 queued frames; pattern frames, random net
    "snn_engine": (["engine", "--seed", "4", "--frames", "5"], "tb/snn_engine_tb.v", ENGINE_RTL),
}


def verdict(text):
    lines = text.splitlines()
    return any(re.match(r"\s*RESULT:\s*PASS", l) for l in lines) and not any(
        re.match(r"\s*RESULT:\s*FAIL", l) or "MISMATCH" in l for l in lines)


def _simulate(vec, tb, rtl, base, tag):
    exe = ROOT / "work" / ("%s_%s.vvp" % (base.name, tag))
    cmd = [IVERILOG, "-g2012", "-DVEC=\"%s\"" % vec.as_posix(), "-I", str(vec), "-o", str(exe), str(ROOT / tb)] + [str(base / f) for f in rtl]
    c = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if c.returncode != 0:
        return False, "COMPILE ERROR\n" + c.stdout + c.stderr
    try:
        r = subprocess.run([VVP, str(exe)], capture_output=True, text=True, timeout=1500, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return verdict(r.stdout + r.stderr), r.stdout + r.stderr


def run_case(name, base=ROOT):
    args, tb, rtl = CASES[name]
    vec = ROOT / "work" / "vec" / name
    vec.mkdir(parents=True, exist_ok=True)
    _gen(args + ["--out", str(vec)])
    return _simulate(vec, tb, rtl, base, name)


def run_real(base=ROOT):
    """The exported KTH network on a stored real sample, if present."""
    d = ROOT / "rtl" / "weights" / "kth6"
    if not (d / "snn_config.json").exists() or not (d / "sample_bins.npz").exists():
        return None, "rtl/weights/kth6/{snn_config.json,weights_int8.npz,sample_bins.npz} not exported yet"
    vec = ROOT / "work" / "vec" / "snn_kth6"
    vec.mkdir(parents=True, exist_ok=True)
    _gen(["case", "--run", str(d), "--bins", str(d / "sample_bins.npz"), "--sample", "0", "--steps", "8", "--out", str(vec)])
    return _simulate(vec, "tb/snn_core_tb.v", CORE_RTL, base, "snn_kth6")


TOP_RTL = ["tb/pll25_sim.v", "rtl/snn_demo_top.v", "rtl/snn_core.v", "rtl/lif_conv.v", "rtl/snn_ctrl.v", "rtl/bin_builder.v",
           "rtl/clip_replay.v", "rtl/frame_sync.v", "rtl/pix_source.v", "rtl/event_view.v", "rtl/event_emulator.v", "rtl/event_fifo.v",
           "rtl/sdp_ram.v", "rtl/log_lut.v", "rtl/vga_timing.v", "rtl/reset_sync.v", "rtl/hex7seg.v"]


def run_top(kind):
    """Board top level (tb/snn_demo_top_tb.v), several minutes each: replay of clip 0 / 5 of the exported network, or the live chain."""
    d = ROOT / "rtl" / "weights" / "kth6"
    if not (d / "clips_pred.hex").exists():
        return None, "rtl/weights/kth6 not exported"
    macros = ["-DVEC=\"%s\"" % d.relative_to(ROOT).as_posix()]
    if kind == "live":
        vec = ROOT / "work" / "vec" / "top_live"
        vec.mkdir(parents=True, exist_ok=True)
        _gen(["chain", "--run", str(d), "--frames", "4", "--out", str(vec)])
        macros = ["-DVEC=\"%s\"" % vec.relative_to(ROOT).as_posix(), "-DLIVE=1", "-DTPAR=4"]
    else:
        macros += ["-DNPRESS=%d" % int(kind)]
    exe = ROOT / "work" / ("top_%s.vvp" % kind)
    c = subprocess.run([IVERILOG, "-g2012"] + macros + ["-o", str(exe), str(ROOT / "tb/snn_demo_top_tb.v")] + [str(ROOT / f) for f in TOP_RTL],
                       capture_output=True, text=True, cwd=ROOT)
    if c.returncode != 0:
        return False, "COMPILE ERROR\n" + c.stdout + c.stderr
    r = subprocess.run([VVP, str(exe)], capture_output=True, text=True, timeout=3000, cwd=ROOT)
    return verdict(r.stdout + r.stderr), r.stdout + r.stderr


def run_tops():
    ok = True
    for kind in ("0", "5", "live"):
        passed, text = run_top(kind)
        info = [l for l in text.splitlines() if l.startswith("info:")]
        print(("PASS  " if passed else "FAIL  ") + "snn_top/" + kind + "   " + (info[-1][6:] if info else text[:80]))
        if not passed:
            ok = False
            print(text[-1500:])
    return ok


def run_all():
    ok = True
    for name in CASES:
        passed, text = run_case(name)
        info = [l for l in text.splitlines() if l.startswith("info:")]
        print(("PASS  " if passed else "FAIL  ") + "snn/" + name + "   " + (info[-1][6:] if info else "(no info line)"))
        if not passed:
            ok = False
            print(text[-2000:])
    passed, text = run_real()
    if passed is None:
        print("SKIP  snn/snn_kth6   (" + text + ")")
    else:
        info = [l for l in text.splitlines() if l.startswith("info:")]
        print(("PASS  " if passed else "FAIL  ") + "snn/snn_kth6   " + (info[-1][6:] if info else ""))
        if not passed:
            print(text[-2000:])
        ok = ok and passed
    return ok


# (file, old, new, case that should catch it)
MUTATIONS = [
    ("rtl/lif_conv.v", "vl = vo - (vo >>> LEAK);", "vl = vo - (vo >> LEAK);", "snn_rand"),               # logical instead of arithmetic shift
    ("rtl/lif_conv.v", "(p_q | spk8)", "(p_q & spk8)", "snn_rand"),                                         # pooling AND instead of OR
    ("rtl/lif_conv.v", "(vs > VMAX) ? VMAX : vs", "(vs > VMAX - 1) ? VMAX - 1 : vs", "snn_sat"),            # clip off by one at the top rail
    ("rtl/lif_conv.v", "(vs < VMIN) ? VMIN : (", "(vs < VMIN + 1) ? VMIN + 1 : (", "snn_sat"),              # and at the bottom rail
    ("rtl/lif_conv.v", "(cx >= 0) && (cx < HW)", "(cx >= 0) && (cx < HW - 1)", "snn_rand"),                  # padding wrong on the right edge
    ("rtl/lif_conv.v", "vc = spk8[jc] ? (vc - THRS) : vc;", "vc = spk8[jc] ? vc : vc;", "snn_rand"),          # no reset by subtraction
    ("rtl/snn_core.v", "{{20{fc_q[7]}}, fc_q}", "{20'd0, fc_q}", "snn_rand"),                               # int8 not sign-extended in the classifier
    ("rtl/snn_core.v", "assign l1_nx_addr = ((l2_ci >> 3) * 256)", "assign l1_nx_addr = ((l2_ci >> 3) * 255)", "snn_rand"),
    ("rtl/snn_ctrl.v", "if (v > best_v)", "if (v >= best_v)", "snn_sat"),                                   # ties must go to the lowest class (np.argmax)
    ("rtl/bin_builder.v", "{ev_data[31], ev_data[30:26], ev_data[23:19]}", "{ev_data[31], ev_data[23:19], ev_data[30:26]}", "snn_chain"),
    ("rtl/bin_builder.v", "wbank <= ~wbank; cptr <= 12'd1;", "wbank <= wbank; cptr <= 12'd1;", "snn_chain"),
    ("rtl/frame_sync.v", "pend && emu_idle && idle_d && idle_dd", "pend && idle_dd", "snn_chain"),         # frame closed before its events are in
    ("rtl/snn_engine.v", "{in_rd_seq[1:0], 14'd0}", "{in_rd_seq[1:0], 13'd0}", "snn_engine"),            # wrong ring slot address
    ("rtl/snn_engine.v", "s_cur[k]   <= cur_cap[k*28 +: 28];", "s_cur[k]   <= score_cap[k*28 +: 28];", "snn_engine"),   # CSR snapshot of the wrong vector
    ("rtl/snn_engine.v", "8'd8:  csr_readdata <= done_seq;", "8'd8:  csr_readdata <= in_rd_seq;", "snn_engine"),        # DONE_SEQ reports "read", not "finished"
]
# None = no expectation:
# idle_dd alone may be enough in simulation because the extra idle clocks are margin, not a hard requirement
EXPECT_CAUGHT = [True, True, True, True, True, True, True, True, True, True, True, None, True, True, True]


def selftest():
    print("selftest (SNN): broken RTL variants must make their case FAIL")
    good = True
    for i, ((rel, old, new, case), want) in enumerate(zip(MUTATIONS, EXPECT_CAUGHT)):
        mdir = ROOT / "work" / ("snnmut%d" % i)
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
        passed, _ = run_case(case, mdir)
        caught = not passed
        okk = want is None or caught == want
        print("  %s  %s: %s -> %s   [%s]%s" % ("CAUGHT" if caught else "MISSED", rel, old, new, case, "" if okk else "   <-- UNEXPECTED"))
        good = good and okk
    return good


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--top" in sys.argv:         # slow: the board top level (about 6 minutes per run)
        sys.exit(0 if run_tops() else 1)
    sys.exit(0 if run_all() else 1)
