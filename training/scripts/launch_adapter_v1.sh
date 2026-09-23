#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/LLM_HOME/src/localpilot
source /home/natha/.venvs/localpilot-training/bin/activate
source /etc/profile.d/rocdxg-amd-smi-lib.sh

config_path="training/configs/qlora_v1.yaml"
run_name="$(python -c 'import json, pathlib; c=json.load(open("training/configs/qlora_v1.yaml", encoding="utf-8")); print(pathlib.Path(c["output"]["directory"]).name)')"

# Do not inherit allocator experiments from an interactive shell or an earlier
# troubleshooting attempt. The approved ROCm/Unsloth path uses PyTorch defaults.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

# exec preserves this PID so the Windows memory guard can interrupt the exact
# training process without matching unrelated Python or WSL workloads.
printf '%s\n' "$$" > "training/reports/${run_name}.pid"
exec python training/scripts/train_adapter.py \
  --config "$config_path" \
  --train \
  --dry-run-report "training/reports/${run_name}_dry_run.json" \
  --confirm TRAIN_LOCALPILOT_ADAPTER_V1 \
  "$@"
