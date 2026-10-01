#!/usr/bin/env bash
# Tests the entrypoint's speculative config (image/entrypoint.sh, from `SPEC=()` to its "speculative:" line) without a GPU:
# runs those exact lines under each setting and checks the one --speculative-config vLLM gets. The reference for the default
# stack is what vLLM used before SPEC_EXTRA: the second --speculative-config adaptive-k.yaml put in EXTRA_ARGS (vLLM keeps
# the last one). Usage: dev/entrypoint-spec-test.sh   (prints PASS or the failing case; exit status = failures)
cd "$(dirname "$0")/.."
block=$(sed -n '/^SPEC=()$/,/^\[\[ \${#SPEC\[@\]} -gt 0 \]\] && echo "speculative:/p' image/entrypoint.sh)
extra=$(sed -n 's/^ *- SPEC_EXTRA=//p' experimental/compose/adaptive-k.yaml)
d1=$(mktemp -d); d2=$(mktemp -d); touch "$d1/config.json" "$d2/config.json"; trap 'rm -rf "$d1" "$d2"' EXIT
fails=0
run() {  # run <env assignments...>: prints the final SPEC[1] (or nothing), exit status of the block
  env -i PATH="$PATH" BLOCK="$BLOCK" "$@" bash -c 'eval "$BLOCK" >/dev/null; printf %s "${SPEC[1]:-}"' 2>/dev/null
}
export BLOCK="$block"
check() {  # check <name> <python condition on c (the parsed config, or None) and rc>
  local name=$1 cond=$2; shift 2
  out=$(run "$@"); rc=$?
  python3 -c 'import json, sys; out, rc = sys.argv[1], int(sys.argv[2]); c = json.loads(out) if out else None
d1, d2 = sys.argv[4], sys.argv[5]; sys.exit(0 if eval(sys.argv[3]) else 1)' "$out" "$rc" "$cond" "$d1" "$d2" \
    || { echo "FAIL $name: rc=$rc out=$out"; fails=$((fails + 1)); }
}
ref='{"method": "dflash", "disable_eagle_block_drop": True, "model": d1, "num_speculative_tokens": 7, "attention_backend": "TRITON_ATTN", "num_speculative_tokens_per_batch_size": [[1,1,7],[2,2,5],[3,3,4],[4,4,3],[5,5,2],[6,32,7]]}'
check "default stack = the old overlay's config" "rc == 0 and c == $ref" SPEC_METHOD=dflash DFLASH_MODEL="$d1" SPEC_EXTRA="$extra"
for f in tp3 tp6; do  # these restate EXTRA_ARGS after adaptive-k.yaml and keep its SPEC_EXTRA (#66)
  ! grep -q '^ *- SPEC_EXTRA=' "experimental/compose/$f.yaml" || { echo "FAIL $f.yaml sets its own SPEC_EXTRA"; fails=$((fails + 1)); }
  check "$f.yaml = the default stack's config" "rc == 0 and c == $ref" SPEC_METHOD=dflash DFLASH_MODEL="$d1" \
    SPEC_EXTRA="$extra" EXTRA_ARGS="$(sed -n 's/^ *- EXTRA_ARGS=//p' experimental/compose/$f.yaml)"
done
check "stock (no overlay) unchanged" "rc == 0 and out == json.dumps({'method': 'dflash', 'model': d1, 'num_speculative_tokens': 7}, separators=(',', ':'))" SPEC_METHOD=dflash DFLASH_MODEL="$d1"
check "SPEC_TOKENS takes effect, widths capped" "rc == 0 and c['num_speculative_tokens'] == 5 and [r[2] for r in c['num_speculative_tokens_per_batch_size']] == [5,5,4,3,2,5]" SPEC_METHOD=dflash SPEC_TOKENS=5 DFLASH_MODEL="$d1" SPEC_EXTRA="$extra"
check "DFLASH_MODEL takes effect" "rc == 0 and c['model'] == d2" SPEC_METHOD=dflash DFLASH_MODEL="$d2" SPEC_EXTRA="$extra"
check "mtp with the DFlash overlay is refused" "rc != 0" SPEC_METHOD=mtp SPEC_EXTRA="$extra"
check "none with the DFlash overlay is refused" "rc != 0" SPEC_METHOD=none SPEC_EXTRA="$extra"
check "mtp without the overlay still works" "rc == 0 and c == {'method': 'mtp', 'num_speculative_tokens': 4}" SPEC_METHOD=mtp
check "a second --speculative-config in EXTRA_ARGS is refused" "rc != 0" SPEC_METHOD=dflash DFLASH_MODEL="$d1" EXTRA_ARGS='--speculative-config {"method":"dflash"}'
check "SPEC_EXTRA cannot set the method, model or width" "rc != 0" SPEC_METHOD=dflash DFLASH_MODEL="$d1" SPEC_EXTRA='{"model":"/x"}'
check "SPEC_EXTRA must be a JSON object" "rc != 0" SPEC_METHOD=dflash DFLASH_MODEL="$d1" SPEC_EXTRA='[1]'
grep -q '"${SPEC\[\*\]:-}" == \*num_speculative_tokens_per_batch_size\*' image/entrypoint.sh \
  || { echo "FAIL CUDA graph ceiling still reads the draft table from EXTRA_ARGS"; fails=$((fails + 1)); }
[ $fails = 0 ] && echo PASS
exit $fails
