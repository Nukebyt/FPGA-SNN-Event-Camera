#!/usr/bin/env bash
# Regenerates quartus/demo/ip/pll25 (altera_pll, 50 MHz -> 25.000 MHz, integer-N, with locked).
# Same recipe as the Adaptive-Night-Vision project. Generated IP is committed; this is the command that made it.
set -euo pipefail
cd "$(dirname "$0")/../quartus/demo"
export PATH="/c/intelFPGA_lite/21.1/quartus/sopc_builder/bin:/c/intelFPGA_lite/21.1/quartus/bin64:$PATH"
ip-generate --remove-qsys-generate-warning --component-name=altera_pll --output-directory=ip/pll25 \
  --file-set=QUARTUS_SYNTH --output-name=pll25 --system-info=DEVICE_FAMILY="Cyclone V" \
  --component-param=gui_reference_clock_frequency=50.0 --component-param=gui_number_of_clocks=1 \
  --component-param=gui_output_clock_frequency0=25.0 --component-param=gui_pll_mode="Integer-N PLL" \
  --component-param=gui_use_locked=true --component-param=gui_en_reconf=false \
  --component-param=gui_operation_mode=direct
