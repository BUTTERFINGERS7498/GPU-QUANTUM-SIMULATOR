#!/bin/bash
# Run the GHZ benchmark and write its output to results/<log-name>.
# Run it from the repo root. The log is appended to, so an older log is kept.
#
#   wsl -d CyberTronTraining --cd <repo> -- bash scripts/run_ghz.sh <log-name> [benchmark flags]
#
# Example: bash scripts/run_ghz.sh ghz_run4.log --resume --only nvidia
if [ -z "$1" ]; then
    echo "usage: bash scripts/run_ghz.sh <log-name> [benchmark flags]" >&2
    exit 2
fi
LOG="results/$1"
shift
mkdir -p results
. ~/gpu-quantum-sim/.venv-old/bin/activate
python -m src.ghz_benchmark --output-dir results "$@" >> "$LOG" 2>&1
rc=$?
echo "DONE rc=$rc" >> "$LOG"
exit "$rc"
