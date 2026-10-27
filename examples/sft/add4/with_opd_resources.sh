#!/usr/bin/env bash
# Execute on the training HOST; pass the existing docker/torchrun command as argv.
set -euo pipefail
: "${ADD_RUN_DIR:?Set the new experiment directory}"
if [ "$#" -eq 0 ]; then echo 'Pass the training command after this script' >&2; exit 2; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ADD_RUN_DIR"
python "$script_dir/add4_opd_resources.py" --once --output "$ADD_RUN_DIR/resources.json"
python - "$ADD_RUN_DIR/resources.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))['metrics']
if not all(metrics.get(f'resources/{role}_probe_ok') == 1 for role in ('training_host', 'teacher_host')):
    raise SystemExit('Host resource preflight failed; inspect SSH/nvidia-smi before training')
PY
python "$script_dir/add4_opd_resources.py" --interval 30 --output "$ADD_RUN_DIR/resources.json" &
collector_pid=$!
trap 'kill "$collector_pid" 2>/dev/null || true; wait "$collector_pid" 2>/dev/null || true' EXIT
"$@"
