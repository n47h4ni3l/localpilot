#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/LLM_HOME/src/localpilot
source /home/natha/.venvs/localpilot-training/bin/activate
source /etc/profile.d/rocdxg-amd-smi-lib.sh

# exec preserves this PID so the Windows memory guard can interrupt the exact
# training process without matching unrelated Python or WSL workloads.
printf '%s\n' "$$" > training/reports/adapter_v1_eager_22g_20260922.pid
exec python training/scripts/train_adapter.py \
  --config training/configs/qlora_v1.yaml \
  --train \
  --dry-run-report training/reports/adapter_v1_eager_22g_20260922_dry_run.json \
  --confirm TRAIN_LOCALPILOT_ADAPTER_V1 \
  "$@"
