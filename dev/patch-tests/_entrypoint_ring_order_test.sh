#!/usr/bin/env bash
# Runs the entrypoint's TP parse (TP=RING4 / RING3 and the rank order that goes
# with them) for each TP value. The block is cut out of image/entrypoint.sh, so
# this tests the shipped code. Needs only bash. Exit status is the number of
# failures.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ep="$here/../../image/entrypoint.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

sed -n '/^FABRIC_LAYOUT=mesh$/,/^  export VLLM_RAY_BUNDLE_INDICES$/p' "$ep" > "$tmp/block.sh"
echo fi >> "$tmp/block.sh"
if ! grep -q 'VLLM_RAY_BUNDLE_INDICES=\$(seq 0' "$tmp/block.sh"; then
  echo "FAIL could not cut the TP block out of $ep"; exit 1
fi

check() {
  if [[ "$2" == "$3" ]]; then echo "  ok   $1"
  else echo "  FAIL $1: got '$2', want '$3'"; fails=$(( fails + 1 )); fi
}

# Prints TP|FABRIC_LAYOUT|VLLM_RAY_BUNDLE_INDICES after the block.
run() {
  env -i PATH="$PATH" BLOCK="$tmp/block.sh" "$@" bash -c '
    set -euo pipefail
    source "$BLOCK"
    echo "$TP|$FABRIC_LAYOUT|${VLLM_RAY_BUNDLE_INDICES:-}"'
}

check "TP=4 mesh leaves the order to vLLM" "$(run TP=4)" "4|mesh|"
check "TP=RING4 keeps mentat's cable order" "$(run TP=RING4)" "4|ring|0,1,2,3"
check "TP=RING3 keeps mentat's cable order" "$(run TP=RING3)" "3|ring|0,1,2"
check "an explicit order wins" "$(run TP=RING4 VLLM_RAY_BUNDLE_INDICES=0,1,3,2)" "4|ring|0,1,3,2"
exit "$fails"
