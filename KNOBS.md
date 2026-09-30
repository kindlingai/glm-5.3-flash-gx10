# Knobs

Environment variables read by `entrypoint.sh` and `discover.sh`, extracted mechanically.
A default of _(none)_ means the variable is referenced without one --
check the entrypoint for whether it is required or merely optional.
Meaning and rationale stay in the entrypoint and compose comments.

| variable | default |
|---|---|
| `ALLOW_STOCK` | _(empty)_ |
| `API_PORT` | `8002` |
| `AUTOTUNE_CACHE` | _(empty)_ |
| `BLOCK_SIZE` | `2304` |
| `CACHE_ROOT` | `/root/.cache` |
| `CACHE_TAG` | `${SHARED_TAG}-${_opthash}` _(derived)_ |
| `CHAT_TEMPLATE` | `/usr/local/share/glm53-chat-template.jinja` |
| `CLUSTER_SUBNET` | `${FABRIC_SUBNETS%% *}` _(derived)_ |
| `CONTAINER_NAME` | `glm53` |
| `CUDAGRAPH_CAPTURE_SIZES` | `8 16 32 64 96 128 192 256` |
| `CUDAGRAPH_MAX` | `MAX_NUM_SEQS` x (1 + `SPEC_TOKENS`), at most 512 _(derived)_; unset, vLLM picks it when the draft width varies with batch size |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` |
| `CUDA_GRAPHS` | `1` |
| `DECODE_RESERVE_TOKENS` | _(empty)_ |
| `DFLASH_MODEL` | `/models/glm-5.3-flash-dflash2` |
| `EXTRA_ARGS` | _(empty)_ |
| `FABRIC_CHECK` | `1` |
| `FABRIC_CHECK_GBPS_PER_DEVICE` | `95` |
| `FABRIC_CHECK_PORT` | `29511` |
| `FABRIC_CHECK_TIMEOUT_S` | `120` |
| `FABRIC_SUBNETS` | `$CLUSTER_SUBNET`, else one prefix per `rdma`-tagged address from mentatd _(derived)_ |
| `FABRIC_WAIT_S` | `0` |
| `GLOO_SOCKET_IFNAME` | _(empty)_ |
| `GPU_MEM_UTIL` | `0.88` |
| `HEAD_HOST` | elected when `ROLE` is also unset _(derived)_ |
| `ITERATION_DETAILS` | _(empty)_ |
| `KV_CACHE_DTYPE` | `fp8_e4m3` |
| `KV_CACHE_MEMORY` | `27917287424` |
| `LIMIT_MM` | `{\"image\":16,\"video\":0\}` |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `2304` |
| `MAX_JOBS` | `4` |
| `MAX_MODEL_LEN` | `524288` |
| `MAX_NUM_BATCHED_TOKENS` | `16384` |
| `MAX_NUM_SEQS` | `32` |
| `MCP_LOG_DIR` | `/logs` |
| `MENTAT_GROUP` | `${SERVICE_NAME:-glm53}` _(derived)_ |
| `MENTAT_MCP_API` | `${STATUS_PORT:-8082}/mcp` _(derived)_ |
| `MENTAT_MODEL_PROVIDER` | `vllm` |
| `MENTAT_NODE_IP` | mentatd's `node_ip` when electing _(derived)_ |
| `MENTAT_OPENAI_API` | `${API_PORT:-8002}/v1` _(derived)_ |
| `MODEL_DIR` | `/models/glm-5.3-flash-nvfp4` |
| `MOE_BACKEND` | `flashinfer_cutlass` |
| `MTP` | `1` |
| `NCCL_DEBUG` | `INFO` |
| `NCCL_IB_GID_INDEX` | _(empty)_ |
| `NCCL_IB_HCA` | _(empty)_ |
| `NCCL_MAX_NCHANNELS` | `8` |
| `NCCL_SOCKET_IFNAME` | `$GLOO_SOCKET_IFNAME` _(derived)_ |
| `PREFLIGHT` | `1` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |
| `RAY_ADDRESS` | `127.0.0.1:6379` when electing, else `${HEAD_HOST:?set HEAD_HOST to the head node address}:6379` _(derived)_ |
| `RAY_MEMORY_MONITOR_REFRESH_MS` | `0` |
| `RAY_OBJECT_STORE_MEMORY` | `4294967296` |
| `ROCE_SETTLE_S` | `60` |
| `ROLE` | elected when `HEAD_HOST` is also unset, else `head` |
| `SAFETENSORS_LOAD_STRATEGY` | `eager` |
| `SELF_TEST` | `1` |
| `SERVED_NAME` | `glm53` |
| `SERVICE_NAME` | `glm53` |
| `SHARED_TAG` | `glm53-${_arch}-${_kv}` _(derived)_ |
| `SKIP_MM_PROFILING` | `0` |
| `SPEC_METHOD` | `dflash` |
| `SPEC_TOKENS` | `7` |
| `STAGE_FILE` | `/tmp/glm53-stage` |
| `STATUS_PORT` | `8082` |
| `TILELANG_CACHE_DIR` | `${CACHE_ROOT}/${SHARED_TAG}/tilelang` _(derived)_ |
| `TOOL_PARSER` | `glm47` |
| `TOPK_BACKEND` | `per_row` |
| `TORCH_MEM_FRACTION` | `0.92` |
| `TP` | `4` (or `2`, `3`, `RING4`) |
| `TRITON_CACHE_DIR` | `${CACHE_ROOT}/${SHARED_TAG}/triton` _(derived)_ |
| `VLLM_ARX_TWO_SHOT_MAX_KB` | `2048` (with arx.yaml; `0` sends all-reduces over 256 KB to NCCL; NCCL is faster at 4 MB) |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600` |
| `VLLM_GLM5NEXT_RECOVERSSM` | `0` (1 with recoverssm.yaml) |
| `VLLM_HOST_IP` | the `lan`-tagged address from mentatd, else its `node_ip` _(derived)_ |
| `VLLM_WEIGHT_SNAPSHOT_DIR` | _(empty)_ |
| `WORKER_WAIT_S` | `0` |
