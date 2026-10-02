#!/usr/bin/env bash
# Runs the entrypoint's fabric-port derivation (the block that sets NCCL_IB_HCA
# and NCCL_IB_GID_INDEX) against scripted RoCE GID tables, with fabric_port
# answering from the script and sleep advancing a fake clock. The block is cut
# out of image/entrypoint.sh by its first and last lines, so this tests the
# shipped code, not a copy. Needs only bash. Exit status is the number of
# failures.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ep="$here/../../image/entrypoint.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

sed -n '/^if \[\[ -z "${NCCL_IB_HCA:-}" || -z "${NCCL_IB_GID_INDEX:-}" \]\]; then$/,/^export NCCL_IB_HCA NCCL_IB_GID_INDEX$/p' \
  "$ep" > "$tmp/block.sh"
if ! grep -q '^export NCCL_IB_HCA' "$tmp/block.sh"; then
  echo "FAIL could not cut the fabric block out of $ep"; exit 1
fi

check() {
  if [[ "$2" == "$3" ]]; then echo "  ok   $1"
  else echo "  FAIL $1: got '$2', want '$3'"; fails=$(( fails + 1 )); fi
}

# One box's pass through the block. PORTS lists "prefix=device:gid@seconds",
# the time at which that port's RoCE v2 GID appears; a prefix left out never
# comes up. The rest are VAR=value settings. Prints
# NCCL_IB_HCA|NCCL_IB_GID_INDEX|seconds waited|exit status
# on stdout, and the block's own messages in $tmp/err.
run() {
  env -i PATH="$PATH" BLOCK="$tmp/block.sh" FABRIC_LAYOUT=mesh "$@" bash -c '
    set -euo pipefail
    NOW=0
    fabric_port() {
      local e
      for e in ${PORTS:-}; do
        [[ "${e%%=*}" == "$1" ]] || continue
        e="${e#*=}"
        (( NOW >= ${e##*@} )) || return 1
        e="${e%@*}"; echo "${e%%:*} ${e##*:}"; return 0
      done
      return 1
    }
    sleep() { NOW=$(( NOW + $1 )); }
    ip() { :; }
    ls() { :; }
    # the block exits on FATAL, so report from an EXIT trap on fd 3
    ( trap '"'"'rc=$?; echo "${NCCL_IB_HCA:-}|${NCCL_IB_GID_INDEX:-}|$NOW|$rc" >&3'"'"' EXIT
      . "$BLOCK" >/dev/null ) 3>&1 || true
  ' 2> "$tmp/err"
}

A=198.18.0. B=198.19.0.

echo "== both roots up at boot"
check "two devices, no wait" "$(run FABRIC_SUBNETS="$A $B" PORTS="$A=mlx5_0:3@0 $B=mlx5_1:3@0")" "mlx5_0,mlx5_1|3|0|0"

echo "== second root's GID comes up 20 s late"
check "waits for it, then both" "$(run FABRIC_SUBNETS="$A $B" PORTS="$A=mlx5_0:3@0 $B=mlx5_1:3@20")" "mlx5_0,mlx5_1|3|20|0"

echo "== second root never comes up"
check "fatal, not one device" "$(run FABRIC_SUBNETS="$A $B" PORTS="$A=mlx5_0:3@0")" "||60|1"
check "names the missing subnet" "$(grep -c "no RoCE v2 GID for $B after 60s" "$tmp/err")" "1"

echo "== ROCE_SETTLE_S bounds the wait"
check "fatal at 15 s" "$(run FABRIC_SUBNETS="$A $B" ROCE_SETTLE_S=15 PORTS="$A=mlx5_0:3@0")" "||15|1"

echo "== no fabric at all"
check "fatal" "$(run FABRIC_SUBNETS="$A $B")" "||60|1"
check "says any of" "$(grep -c "no RoCE v2 GID for any of" "$tmp/err")" "1"

echo "== one subnet configured"
check "one device" "$(run FABRIC_SUBNETS="$A" PORTS="$A=mlx5_0:3@0")" "mlx5_0|3|0|0"

echo "== ring: any one subnet is enough, as before"
check "starts with the root it has" \
  "$(run FABRIC_LAYOUT=ring FABRIC_SUBNETS="$A $B" PORTS="$A=mlx5_0:3@0")" "mlx5_0|3|0|0"

echo "== roots at different GID indexes"
check "still fatal" "$(run FABRIC_SUBNETS="$A $B" PORTS="$A=mlx5_0:3@0 $B=mlx5_1:5@0")" "||0|1"
check "says disagree" "$(grep -c "disagree on GID index" "$tmp/err")" "1"

echo "== both pinned in .env"
check "derivation skipped" \
  "$(run NCCL_IB_HCA=mlx5_9 NCCL_IB_GID_INDEX=7 FABRIC_SUBNETS="$A $B")" "mlx5_9|7|0|0"

echo "$fails failure(s)"
exit "$fails"
