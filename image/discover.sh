# Sourced by entrypoint.sh. Fills in the networking and the head/worker role
# from mentat when compose/.env leaves them out. Every function returns without
# asking mentat when its values are already set.
#
# dev/patch-tests/_entrypoint_discovery_test.sh sources this file with
# mentat_query and sleep stubbed.

# The one place that asks mentat anything, so the test can answer from
# fixtures. `local` asks this box's daemon over HTTP. The control port would
# relay the question to mentat's head, and the head answers about its own box. `group` is the head's view of MENTAT_GROUP, where the agents are.
mentat_query() {
  case "$1" in
    local) curl -sf --max-time 5 http://127.0.0.1:6380/status ;;
    group) ray status --address="$RAY_ADDRESS" --group "$MENTAT_GROUP" --json 2>/dev/null ;;
  esac
}

# The local daemon's view as "node_ip|lan address|fabric prefixes". The lan
# address falls back to node_ip when nothing is tagged lan, since mentat names
# the box by node_ip either way. A prefix is "a.b.c.", the form of CLUSTER_SUBNET
# and FABRIC_SUBNETS, one per rdma-tagged address and sorted, so every box
# lists its PCIe roots in the same order.
mentat_local_view() {
  mentat_query local | python3 -c '
import ipaddress, json, sys
try:
    s = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
tags = s.get("addr_tags") or {}
node = s.get("node_ip") or ""
lan = [a for a in s.get("addrs") or [] if "lan" in tags.get(a, [])]
fabric = {a.rsplit(".", 1)[0] + "." for a, t in tags.items() if "rdma" in t and ":" not in a}
fabric = sorted(fabric, key=lambda p: ipaddress.ip_address(p + "0"))
print(node + "|" + (lan[0] if lan else node) + "|" + " ".join(fabric))
'
}

# Sets whichever of VLLM_HOST_IP, CLUSTER_SUBNET and FABRIC_SUBNETS are unset,
# and MENTAT_NODE_IP when the boxes will elect a head. Waits for the daemon,
# and for its rdma addresses when the fabric is needed: they are also absent
# while the switch reboots, and exiting then turns a two-minute outage into a
# restart loop racing it. FABRIC_WAIT_S bounds the wait; 0 waits forever.
discover_network() {
  local want_ip="" want_fabric="" want_node="" view node lan fabric waited=0
  [[ -z "${VLLM_HOST_IP:-}" ]] && want_ip=1
  [[ -z "${CLUSTER_SUBNET:-}" && -z "${FABRIC_SUBNETS:-}" && "${TP:-4}" != 1 ]] && want_fabric=1
  # Every box registers with its own daemon, and an agent that reaches its
  # daemon over loopback leaves its address out of its agent id. The relay to
  # the head fills in the address but not the id, so four boxes would register
  # as one agent and replace each other in a loop. Naming the box keeps the
  # ids apart. It must be the daemon's own node_ip, or mentat refuses it.
  [[ -z "${ROLE:-}" && -z "${HEAD_HOST:-}" && -z "${MENTAT_NODE_IP:-}" ]] && want_node=1
  [[ -n "$want_ip$want_fabric$want_node" ]] || return 0
  while :; do
    view=$(mentat_local_view) || view=""
    IFS='|' read -r node lan fabric <<< "$view"
    [[ -n "$node" && ( -z "$want_fabric" || -n "$fabric" ) ]] && break
    if (( ${FABRIC_WAIT_S:-0} > 0 && waited >= ${FABRIC_WAIT_S:-0} )); then
      if [[ -z "$node" ]]; then
        echo "FATAL: no answer from mentatd at 127.0.0.1:6380 after ${waited}s." >&2
        echo "Start mentatd on this box, or set VLLM_HOST_IP, CLUSTER_SUBNET, ROLE and HEAD_HOST in .env." >&2
      else
        echo "FATAL: mentatd lists no address tagged rdma after ${waited}s." >&2
        echo "Tag the ConnectX interfaces in MENTAT_ANNOUNCE_IFACES, or set FABRIC_SUBNETS in .env." >&2
      fi
      exit 1
    fi
    if (( waited % 60 == 0 )); then
      if [[ -z "$node" ]]; then echo "waiting for mentatd at 127.0.0.1:6380 (${waited}s)"
      else echo "waiting for mentatd to list an rdma address (${waited}s)"; fi
    fi
    sleep 10
    waited=$(( waited + 10 ))
  done
  [[ -n "$want_ip" ]] && VLLM_HOST_IP="$lan"
  [[ -n "$want_fabric" ]] && FABRIC_SUBNETS="$fabric"
  [[ -n "$want_node" ]] && export MENTAT_NODE_IP="$node"
  echo "mentat: node $node, VLLM_HOST_IP=$VLLM_HOST_IP, fabric ${FABRIC_SUBNETS:-${CLUSTER_SUBNET:-}}"
}

# The group's live GPU agents as "count|addresses", addresses sorted
# numerically. An agent with an empty GPU list is a `python -m ray.register`
# announcement, so the count skips it. A box counts once even if an
# old registration of it lingers.
mentat_group_view() {
  mentat_query group | python3 -c '
import ipaddress, json, sys
try:
    s = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
agents = ((s.get("groups") or {}).get(sys.argv[1]) or {}).get("agents") or {}
ips = {a.get("node_ip") for a in agents.values()
       if a.get("alive") and (a.get("machine") or {}).get("gpus") and a.get("node_ip")}
ips = sorted(ips, key=lambda a: (ipaddress.ip_address(a).version, ipaddress.ip_address(a)))
print(str(len(ips)) + "|" + " ".join(ips))
' "$MENTAT_GROUP"
}

# Waits until MENTAT_GROUP has TP live GPU agents, then makes the lowest
# address the head. Every box runs this against the same list and reaches the
# same answer without talking to the others. Sets and exports HEAD_HOST and
# ROLE. WORKER_WAIT_S bounds the wait. 0 waits forever, so a switch reboot
# is a pause.
elect_head() {
  local view n ips waited=0
  while :; do
    view=$(mentat_group_view) || view=""
    IFS='|' read -r n ips <<< "$view"
    (( ${n:-0} >= TP )) && break
    if (( ${WORKER_WAIT_S:-0} > 0 && waited >= ${WORKER_WAIT_S:-0} )); then
      echo "FATAL: only ${n:-0} of $TP agents in group $MENTAT_GROUP after ${waited}s (${ips:-none})" >&2
      exit 1
    fi
    (( waited % 60 )) || echo "electing: waiting for $TP agents in group $MENTAT_GROUP, have ${n:-0} (${waited}s)"
    sleep 5
    waited=$(( waited + 5 ))
  done
  HEAD_HOST="${ips%% *}"
  if [[ "$HEAD_HOST" == "$MENTAT_NODE_IP" ]]; then ROLE=head; else ROLE=worker; fi
  export HEAD_HOST ROLE
  echo "election: candidates $ips; head $HEAD_HOST; this box ($MENTAT_NODE_IP) is $ROLE"
}

# With HEAD_HOST set and ROLE not, this box is the head if it holds HEAD_HOST.
# Leaves the head's resolved address in _head_ip for the entrypoint's checks.
fixed_role() {
  _head_ip=$(getent ahostsv4 "$HEAD_HOST" 2>/dev/null | awk 'NR==1 {print $1}' || true)
  _head_ip=${_head_ip:-$HEAD_HOST}
  if [[ -z "${ROLE:-}" ]]; then
    if [[ "$VLLM_HOST_IP" == "$_head_ip" ]]; then ROLE=head; else ROLE=worker; fi
  fi
  export ROLE
}
