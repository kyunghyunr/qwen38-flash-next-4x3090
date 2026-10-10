#!/usr/bin/env bash
# Container entry point. Run it as:
#   docker run ... --entrypoint /model/runtime/serve-int4-ple.sh <image> /model
#
# Derived from serve-container.sh in DominikBucko/qwen38-flash-next-2x3090 (Apache-2.0).
# The upstream file sets the profile/env contract and the vLLM flags; that part is unchanged.
# Changes I made, in order of how much they matter:
#   1. INT4 PLE sidecar. When VLLM_PLE_QUANT_DIR is set, patch_int4_mrv1.py rewrites the stock
#      worker.py and ple_layer.py inside the image before vLLM starts, and the quantized table
#      is served from disk. With the variable unset this file behaves like the plain FP8-PLE path.
#   2. --limit-mm-per-prompt: 1 image upstream, 20 here (I tested two at once).
#   3. MTP checkpoint defaults to runtime/mtp-int4-g32.
#   4. Optional: PATCH_SAMPLING_DEFAULTS=1 forces temp/top-p/top-k server defaults for clients
#      that send none. Personal preference, off by default.
#
# Do NOT set VLLM_LOGGING_LEVEL=DEBUG here. See notes/cudagraph-capture-vs-debug-logging.md.
set -euo pipefail

profile=${PROFILE:-/opt/qwen38/configs/2x3090-128gb.env}
[[ -f "$profile" ]] || { echo "missing runtime profile: $profile" >&2; exit 2; }
# shellcheck source=/dev/null
source "$profile"

model=${1:-/model}
mtp_model=${2:-"$model/runtime/mtp-int4-g32"}
rankings=/workspace/static_hot_cache_rankings.json

[[ -f "$model/model.safetensors.index.json" ]] || {
  echo "model checkpoint not found at $model" >&2
  exit 2
}
[[ -f "$rankings" ]] || { echo "missing hot-cache rankings: $rankings" >&2; exit 2; }
[[ -f "$mtp_model/model.safetensors.index.json" ]] || {
  echo "compact MTP checkpoint not found at $mtp_model" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_WNA16_DYNAMIC_LRU=1
export VLLM_WNA16_STATIC_HOT_CACHE_FILE=$rankings
export VLLM_WNA16_MIXED_VMM_HOT_CACHE=1
export VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING=1

# (4) optional server-wide sampling defaults. The sed below is idempotent.
if [ "${PATCH_SAMPLING_DEFAULTS:-0}" = "1" ]; then
  sed -i 's/"top_p": 1.0,/"top_p": 0.95,/; s/"top_k": 0,/"top_k": 20,/' \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/protocol.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/protocol.py
fi

# (1) INT4 PLE sidecar. Keeps the stock connector, protocol, capture path and MTP3 in place and
# swaps only the table serving: FP8-in-RAM becomes INT4-on-disk with a mmap reader.
if [ -n "${VLLM_PLE_QUANT_DIR:-}" ]; then
  OVERLAY=/model/runtime/ple-overlay
  python3 "$OVERLAY/patch_int4_mrv1.py"
  export PYTHONFAULTHANDLER=1
  export PYTHONUNBUFFERED=1
  echo "INT4 PLE sidecar enabled: $VLLM_PLE_QUANT_DIR"
fi

exec vllm serve "$model" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --moe-backend humming \
  --dtype bfloat16 \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --offload-backend uva \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --cpu-offload-params experts \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype auto \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --no-async-scheduling \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 20}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config \
  "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DEPTH,\"use_local_argmax_reduction\":true,\"model\":\"$mtp_model\"}"
