#!/usr/bin/env bash
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=${VERL_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}
run_owner=$(id -un)
ADD_EXPERIMENT=${ADD_EXPERIMENT:-/data/$run_owner/projects/codemem/swe-add-pairing-20261026/add7-noresults-sft-8gpu-20261029}
ADD_DELIVERY=${ADD_DELIVERY:-/mnt/users/zengjiajie/add7-no-tool-results-generation-20261029}
# Existing shared base model, mounted read-only; no download or model server needed.
MODEL_DIR=${MODEL_DIR:-/data/memory-verl-sft/models/base/Qwen3.5-9B}
IMAGE=${IMAGE:-verl-sft:qwen3.5-9b-cu130-driver535}
mode=${1:-check}
if (($#)); then shift; fi
case "$mode" in check|prepare-train|prepare-val|preflight|smoke|train) ;; *) echo 'Use check|prepare-train|prepare-val|preflight|smoke|train' >&2; exit 2;; esac
mkdir -p "$ADD_EXPERIMENT/reports"
entry="$VERL_ROOT/examples/sft/add7"
for required in "$VERL_ROOT/verl" "$ADD_DELIVERY" "$MODEL_DIR"; do
  [[ -d "$required" && -r "$required" && -x "$required" ]] || { echo "Missing/unreadable: $required" >&2; exit 1; }
done
common=(--rm --user "$(id -u):$(id -g)" --shm-size 16g
  -v "$VERL_ROOT:$VERL_ROOT:ro" -v "$ADD_EXPERIMENT:$ADD_EXPERIMENT"
  -v "$ADD_DELIVERY:$ADD_DELIVERY:ro" -v "$MODEL_DIR:/model:ro"
  -e "VERL_ROOT=$VERL_ROOT" -e "ADD_EXPERIMENT=$ADD_EXPERIMENT"
  -e "PYTHONPATH=$VERL_ROOT:$entry" -e HOME=/tmp -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false
  -e OMP_NUM_THREADS=4 -e HYDRA_FULL_ERROR=1 -e ADD_SP_SIZE=4
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint /usr/bin/env)
# Preserve NAS dockerteam access for the non-root training user.
for supplementary_group in $(id -G); do common+=(--group-add "$supplementary_group"); done
clean=( -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy )
# CPU modes never request GPUs and do not touch currently running containers.
if [[ "$mode" == check ]]; then
  docker run --network none "${common[@]}" -e WANDB_MODE=disabled \
    -e ADD_RUN_ID=config-check -e "ADD_RUN_DIR=$ADD_EXPERIMENT/reports" \
    "$IMAGE" "${clean[@]}" python "$entry/train.py" --config-path "$entry" --config-name qwen35.8gpu --cfg job --resolve "$@"
  exit
fi
if [[ "$mode" == prepare-train || "$mode" == prepare-val ]]; then
  split=${mode#prepare-}
  if [[ "$split" == train ]]; then source="$ADD_DELIVERY/delivery/messages.jsonl"; count=12008
  else source="$ADD_DELIVERY/val/messages.jsonl"; count=398; fi
  test -s "$source" || { echo "Not delivered yet: $source" >&2; exit 1; }
  docker run --network none "${common[@]}" "$IMAGE" "${clean[@]}" python "$entry/prepare_data.py" \
    --source "$source" --output "$ADD_EXPERIMENT/prepared/$split" --tokenizer /model \
    --split "$split" --expected "$count" "$@"
  exit
fi
if [[ "$mode" == preflight ]]; then
  docker run --network none "${common[@]}" "$IMAGE" "${clean[@]}" python "$entry/preflight.py" \
    --root "$ADD_EXPERIMENT" --report "$ADD_EXPERIMENT/reports/preflight.json"
  exit
fi
# Explicit train/smoke only. Check complete caches before GPU allocation.
"$0" preflight
# Refuse occupied GPUs; never kill the Add generation service or another user's job.
if [[ -n "$(docker ps --filter 'name=zjj-add7-noresults-' --format '{{.Names}}')" ]]; then
  echo 'An Add-7 SFT container is already active.' >&2; exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU compute jobs are active; leaving them running.' >&2; exit 1
fi
mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ ${#gpu_memory[@]} == 8 ]] || { echo 'Expected exactly 8 GPUs' >&2; exit 1; }
for used in "${gpu_memory[@]}"; do
  ((used < 1000)) || { echo 'GPUs occupied; leave existing jobs running.' >&2; exit 1; }
done
run_id="${mode}-$(date +%Y%m%dT%H%M%S)"
run="$ADD_EXPERIMENT/runs/$run_id"
mkdir -p "$run/wandb"
git -C "$VERL_ROOT" rev-parse HEAD > "$run/source-head.txt"
git -C "$VERL_ROOT" status --short > "$run/source-status.txt"
git -C "$VERL_ROOT" diff > "$run/source-tracked.diff"
cp "$entry/"*.py "$entry/"*.yaml "$entry/"*.sh "$run/"
docker image inspect "$IMAGE" --format '{{.Id}}' > "$run/image-id.txt"
args=(--config-path "$entry" --config-name qwen35.8gpu "hydra.run.dir=$run/hydra")
WANDB_MODE=${WANDB_MODE:-online}
case "$WANDB_MODE" in
  online)
    network=()
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      wandb_env=(-e WANDB_API_KEY)
    elif [[ -r "${HOME}/.netrc" ]]; then
      wandb_env=(-v "${HOME}/.netrc:/tmp/.netrc:ro")
    else
      echo 'WANDB_MODE=online requires a protected WANDB_API_KEY or an existing host wandb login (~/.netrc); never paste a key into the script.' >&2
      exit 1
    fi
    ;;
  offline)
    network=(--network none)
    wandb_env=()
    ;;
  *) echo 'WANDB_MODE must be online or offline for smoke/train.' >&2; exit 2;;
esac
wandb_env+=(-e "WANDB_MODE=$WANDB_MODE" -e "WANDB_DIR=$run/wandb")
[[ -z "${WANDB_ENTITY:-}" ]] || wandb_env+=(-e WANDB_ENTITY)
if [[ "$mode" == smoke ]]; then
  # Three longest-batch64 updates in an isolated run; keep the full-run LR schedule.
  # Stop after three updates; never use its checkpoint for the formal run.
  common+=(-e ADD_SMOKE_STEPS=3)
fi
printf '%s\n' "$@" > "$run/overrides.txt"
printf 'Run directory: %s\n' "$run"
docker run -d --name "zjj-add7-noresults-$run_id" --gpus all "${network[@]}" "${common[@]}" "${wandb_env[@]}" \
  -e "ADD_RUN_ID=$run_id" -e "ADD_RUN_DIR=$run" "$IMAGE" "${clean[@]}" \
  bash -c 'exec torchrun --standalone --nnodes=1 --nproc_per_node=8 "$1/train.py" "${@:2}" > "$ADD_RUN_DIR/training.log" 2>&1' \
  bash "$entry" "${args[@]}" "$@"
