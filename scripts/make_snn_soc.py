"""Derive the SNN SoC project from the night-vision SoC project that ran on this board.

    python scripts/make_snn_soc.py        (needs F:\\Projects\\FPGA-Adaptive-Night-Vision next to this repository)

1. quartus/snn_soc/soc_system.qsys  = the night-vision system with nv_engine_0 replaced by snn_engine_0 (same CSR address 0x10000 on
   the lightweight bridge and the JTAG master, same read master on HPS F2H SDRAM port 0, the write master removed, the status
   conduit renamed snn_status).
2. quartus/snn_soc/DE10_Standard_SNN.v: status wiring and the HEX/LED mapping for the new engine.
The HPS, DDR3, bridges and clocking are untouched, so the proven part of that system stays as it was.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "FPGA-Adaptive-Night-Vision" / "quartus" / "soc"
DST = ROOT / "quartus" / "snn_soc"


def make_qsys():
    s = (SRC / "soc_system.qsys").read_text(encoding="utf-8")
    # drop the write-master connection (nv_engine_0.wr -> hps_0.f2h_sdram0_data)
    s, n = re.subn(r"\n <connection\n   kind=\"avalon\"\n   version=\"21.1\"\n   start=\"nv_engine_0\.wr\"\n.*?\n </connection>", "", s, flags=re.S)
    assert n == 1, "write-master connection not found"
    s = s.replace("nv_engine_0", "snn_engine_0").replace('kind="nv_engine"', 'kind="snn_engine"').replace('name="nv_status"', 'name="snn_status"')
    assert "nv_" not in s, "leftover nv_ references: %s" % sorted(set(re.findall(r"\bnv_\w+", s)))
    (DST / "soc_system.qsys").write_text(s, encoding="utf-8")


def make_top():
    p = DST / "DE10_Standard_SNN.v"
    s = p.read_text(encoding="utf-8")
    s = s.replace("""        .nv_status_lg                          ( nv_lg ),
        .nv_status_ready                       ( nv_ready ),
        .nv_status_enabled                     ( nv_enabled ),
        .nv_status_error                       ( nv_error ),
        .nv_status_frames                      ( nv_frames ),""", """        .snn_status_pred                       ( snn_pred ),
        .snn_status_wcount                     ( snn_wcount ),
        .snn_status_ready                      ( snn_ready ),
        .snn_status_enabled                    ( snn_enabled ),
        .snn_status_error                      ( snn_error ),
        .snn_status_done                       ( snn_done ),""")
    s = s.replace("""  // night-vision engine status (nv_engine_0 conduit, all on the 25 MHz clock)
  wire        clk25, pll_locked;
  wire [15:0] nv_lg;
  wire        nv_ready, nv_enabled, nv_error;
  wire [31:0] nv_frames;""", """  // SNN engine status (snn_engine_0 conduit, all on the 25 MHz clock)
  wire        clk25, pll_locked;
  wire [3:0]  snn_pred;
  wire [7:0]  snn_wcount;
  wire        snn_ready, snn_enabled, snn_error;
  wire [31:0] snn_done;""")
    s = s.replace("assign LEDR[9:1] = {fpga_led_internal[8:3], nv_error, nv_enabled, nv_ready};",
                  "assign LEDR[9:1] = {fpga_led_internal[8:3], snn_error, snn_enabled, snn_ready};")
    a = s.index("// HEX3..HEX0:")
    b = s.index("endmodule")
    s = s[:a] + """// HEX0: class of the last finished window (F until the first window ends), HEX1: window count, HEX3..HEX2: steps done in the
// current window (hex), HEX5..HEX4: frames processed (low byte).
hex7seg hx0 (.d(snn_pred),             .seg_n(HEX0));
hex7seg hx1 (.d(4'hF),                 .seg_n(HEX1));
hex7seg hx2 (.d(snn_wcount[3:0]),      .seg_n(HEX2));
hex7seg hx3 (.d(snn_wcount[7:4]),      .seg_n(HEX3));
hex7seg hx4 (.d(snn_done[3:0]),        .seg_n(HEX4));
hex7seg hx5 (.d(snn_done[7:4]),        .seg_n(HEX5));

""" + s[b:]
    s = s.replace("// Adapted for the Adaptive Night-Vision engine: PLL 50->25 MHz, nv_engine_0 status, LED/HEX use.",
                  "// Adapted from the night-vision SoC for the event-camera SNN engine: PLL 50->25 MHz, snn_engine_0 status, LED/HEX use.")
    assert "nv_" not in s, sorted(set(re.findall(r"\bnv_\w+", s)))
    p.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    make_qsys()
    make_top()
    # the qsf and sdc only need their names/comments adjusted
    for f in ("DE10_Standard_SNN.sdc",):
        p = DST / f
        p.write_text(p.read_text(encoding="utf-8").replace("night-vision SoC design", "event-camera SNN SoC design"), encoding="utf-8")
    p = ROOT / "scripts" / "patch_qip.py"
    t = p.read_text(encoding="utf-8").replace('"quartus", "soc"', '"quartus", "snn_soc"')
    p.write_text(t, encoding="utf-8")
    print("snn_soc derived from", SRC)
