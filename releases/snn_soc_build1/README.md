# snn_soc build 1 (2026-10-29)

`DE10_Standard_SNN.rbf` (compressed, 2.4 MB, for the SD card as `soc_system.rbf`) and `DE10_Standard_SNN.sof` (JTAG; do not use it for the HPS design, see `docs/snn_soc.md`), built by `bash scripts/build_snn_soc.sh --generate` with Quartus Prime Lite 21.1.
The HPS, DDR3 and bridge configuration is the night-vision SoC's unchanged; the engine is `rtl/snn_engine.v` with the KTH 6-class network in `rtl/weights/kth6/`. Sources and their SHA-256: `SOURCES.sha256`. Timing and resources: `*.sta.summary`, `*.fit.summary`, `docs/results/snn_soc.md`.
**Not run on hardware.** Bring-up: `docs/snn_soc.md`.
