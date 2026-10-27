#!/usr/bin/env bash
set -euo pipefail
# This is a dedicated mutable rollout service, never the frozen 180B teacher.
: "${STUDENT_MODEL_DIR:?Set the initial Qwen3.5-9B or Qwen3.6-35B-A3B model directory}"
: "${STUDENT_GPU_IDS:?Set explicitly allocated comma-separated GPU IDs}"
: "${OPD_RUN_ID:?Set a unique run ID}"
: "${STUDENT_PORT:?Set an available loopback port, for example 18082}"
sync_dir=${OPD_SYNC_DIR:-/data/zengjiajie/projects/verl/add4-opd/student-weight-sync}
image=${OPD_SGLANG_IMAGE:-local/qwen38-add-teacher:sglang0.5.21-cu130-tp8-20261017}
model_type=$(python3 - "$STUDENT_MODEL_DIR/config.json" <<'PY'
import json, sys
kind = json.load(open(sys.argv[1]))['model_type']
if kind not in ('qwen3_5', 'qwen3_5_moe'):
    raise SystemExit('Expected Qwen3.5 dense or Qwen3.6 MoE; refusing teacher/other model')
print(kind)
PY
)
model_args=()
if [[ "$model_type" == qwen3_5_moe ]]; then
  model_args+=(--ep-size 1 --moe-runner-backend triton)
fi
mkdir -p "$sync_dir"
IFS=',' read -ra selected_gpus <<< "$STUDENT_GPU_IDS"
for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do unset "$key"; done
docker run -d --pull never --name "add4-opd-student-$OPD_RUN_ID" \
  --gpus "\"device=$STUDENT_GPU_IDS\"" --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "127.0.0.1:$STUDENT_PORT:18080" \
  -v "$STUDENT_MODEL_DIR:$STUDENT_MODEL_DIR:ro" -v "$sync_dir:$sync_dir:ro" \
  --entrypoint python3 "$image" -m sglang.launch_server \
  --model-path "$STUDENT_MODEL_DIR" --tokenizer-path "$STUDENT_MODEL_DIR" \
  --served-model-name add4-opd-student --dtype bfloat16 \
  --weight-version "$OPD_RUN_ID:initial" --mamba-ssm-dtype float32 \
  --tp-size "${#selected_gpus[@]}" --context-length 262144 \
  "${model_args[@]}" \
  --max-running-requests 4 --chunked-prefill-size 2048 --mem-fraction-static 0.85 \
  --enable-custom-logit-processor --host 0.0.0.0 --port 18080
