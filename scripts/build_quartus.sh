#!/usr/bin/env bash
# Quartus flow: map -> fit -> sta -> asm (+ compressed rbf if a .sof is produced), then verify STA.
# Usage: scripts/build_quartus.sh <project_dir> <revision> [required_clock_substring ...]
# Adapted from the Adaptive-Night-Vision project (same board/toolchain). Fails loudly on any step.
set -euo pipefail
export PATH="/c/intelFPGA_lite/21.1/quartus/bin64:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$1"; REV="$2"; shift 2
cd "$ROOT/$DIR"
mkdir -p output_files logs
for step in "quartus_map $REV" "quartus_fit $REV" "quartus_sta $REV" "quartus_asm $REV"; do
  name=$(echo "$step" | cut -d' ' -f1)
  echo "== $step"
  $step > "logs/${name}.log" 2>&1 || { echo "FAILED: $step (see logs/${name}.log)"; tail -20 "logs/${name}.log"; exit 1; }
done
[ -f "output_files/$REV.sof" ] || { echo "FAILED: output_files/$REV.sof missing"; exit 1; }
quartus_cpf -c -o bitstream_compression=on "output_files/$REV.sof" "output_files/$REV.rbf" > logs/cpf.log 2>&1
ls -l "output_files/$REV.sof" "output_files/$REV.rbf"
"$ROOT/.venv/Scripts/python.exe" "$ROOT/scripts/check_sta.py" "output_files/$REV.sta.summary" "$@"
