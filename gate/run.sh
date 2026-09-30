#!/bin/bash
# The PR gate: quality battery and speed suite against a running stack.
#
#   gate/run.sh full LABEL     everything a PR needs (about 2.5 hours)
#   gate/run.sh quick LABEL    the reduced gate (about 1 hour)
#
# Run it on the head node from the repo root, with the stack up and nothing else
# using it. GATE_URL (http://127.0.0.1:8002), GATE_MODEL (glm53), GATE_IMAGE
# (build.sh's default tag, runs the HumanEval programs with no network), QUALITY_DIR
# (GSM8K and HumanEval data, see experimental/quality/quality.py) and
# RIGMARK_DIR (a RigMark checkout; RigMark is skipped without it).
set -u
MODE=${1:?full or quick}; LABEL=${2:?label}
export GATE_URL=${GATE_URL:-http://127.0.0.1:8002} GATE_MODEL=${GATE_MODEL:-glm53}
export QUALITY_URL=$GATE_URL QUALITY_DIR=${QUALITY_DIR:-quality-data}
IMAGE=${GATE_IMAGE:-spark-glm53:v8}
full() { [ "$MODE" = full ]; }
# The HumanEval programs run in a container; without docker access this fell
# through to "permission denied" and HumanEval went unscored.
DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"

echo "--- smoketest"
bash smoketest/run.sh "$GATE_URL" 2>&1 | tail -1
echo "--- NLL against bf16 dense"
python3 gate/nll.py compare | cut -c1-100
echo "--- GSM8K$(full && echo ' and HumanEval')"
python3 experimental/quality/quality.py "$LABEL" 250 2>&1 | grep -E "GSM8K|Error"
if full; then
  $DOCKER run --rm --network none -v "$(realpath "$QUALITY_DIR/$LABEL")":/q:ro \
    -v "$PWD/experimental/quality/run_he.py":/run_he.py:ro --entrypoint python3 "$IMAGE" /run_he.py /q 2>&1 | tail -1
  echo "--- count to 200"
  python3 gate/count.py 5 | tail -1
fi
echo "--- tool call at 42k context"
python3 dev/repro/toolcall_corruption.py --url "$GATE_URL" --model "$GATE_MODEL" \
  --runs "$(full && echo 40 || echo 10)" --concurrency 4 2>&1 | grep diverge | tail -1
echo "--- agent tools"
python3 experimental/quality/agent_tools.py "$GATE_URL" --model "$GATE_MODEL" 2>&1 | tail -1
echo "--- needle"
if full; then LENGTHS="32000 128000 256000 480000"; else LENGTHS="128000 480000"; fi
python3 dev/repro/needle.py --url "$GATE_URL" --model "$GATE_MODEL" --lengths $LENGTHS 2>&1 | tail -1

echo "--- prefill 32k, 128k (cold)"
for i in $(seq 1 "$(full && echo 4 || echo 2)"); do
  python3 gate/prefill.py --base "$GATE_URL/v1" --model "$GATE_MODEL" --sizes 32k,128k --skip-thinking \
    --seed $RANDOM 2>&1 | grep -E "^ +[0-9]{4,}" | awk '{print $3}' | tr "\n" " "; echo
done
echo "--- decode"
python3 gate/decode.py | tail -3
python3 gate/decode_stream.py | tail -3
echo "--- concurrent streams (mixed prompts)"
python3 gate/conc.py 1 2 4 8 16 | tail -5
curl -s "$GATE_URL/metrics" | grep -E "^vllm:num_preemptions_total"
if full && [ -n "${RIGMARK_DIR:-}" ]; then
  echo "--- RigMark"
  mkdir -p gate-results
  (cd "$RIGMARK_DIR" && timeout 1800 python3 bench.py --base-url "$GATE_URL" --model "$GATE_MODEL" \
    --label "$LABEL" --comparison-id ringside-redhat-rowsplit-20260926 --seed 20260905 --runs 2 \
    --metadata "$OLDPWD/gate/rigmark-metadata.json" \
    --decode-tokens 4096 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
    --skip-concurrency --prefill-depths 8192,32768,65536 --prefill-runs 1 \
    --output "$OLDPWD/gate-results/$LABEL.json" 2>&1 | grep -E "GATES")
  python3 gate/rigsum.py "gate-results/$LABEL.json"
fi
echo "--- done"
