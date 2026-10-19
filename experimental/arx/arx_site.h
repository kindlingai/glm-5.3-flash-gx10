// arx_site.h: the site-path contract shared by arx_vllm.cu and any producer kernel that publishes
// its own output (producer=2). A site is one all-reduce position in a step; each has a 64 KB pinned
// send slot. A producer writes bf16 rows into its slot and calls arx_site_publish for the bytes it
// has finished, in any order; the CPU proxy RDMA-writes them to every peer's receive slot for that
// site. Arrival is lamport: receive slots hold 0xFFFF in every bf16 lane until data lands, so a
// producer must never emit the bf16 pattern 0xFFFF (a NaN); the all-reduce kernel canonicalises
// its own copy-mode input to 0xFFC0 (also NaN), a producer writing directly cannot, and a 0xFFFF
// it emits leaves a sentinel in place and traps the receivers after ARX_SPIN_TIMEOUT_S.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace arx_site {
constexpr int kSites = 128;
constexpr size_t kSiteBytes = 64 << 10;
constexpr int kRing = 4096;
constexpr unsigned kSentinel = 0xFFFFu, kSentinelFix = 0xFFC0u;

struct Work { uint64_t site, lo, hi; volatile uint64_t tag; };  // bytes [lo, hi) of send slot `site`

// Publish bytes [lo, hi) of site `site`: the producer's stores must be fenced (__threadfence_system)
// before this call. ring/next come from arx's site_handles().
__device__ __forceinline__ void publish(Work* ring, unsigned long long* next, uint64_t site, uint64_t lo, uint64_t hi) {
  const unsigned long long idx = atomicAdd(next, 1ull);
  Work& w = ring[idx % kRing];
  w.site = site; w.lo = lo; w.hi = hi;
  __threadfence_system();
  w.tag = idx + 1;
}
__device__ __forceinline__ bool has_sentinel(uint4 v) {
  return (v.x & 0xFFFFu) == kSentinel || (v.x >> 16) == kSentinel || (v.y & 0xFFFFu) == kSentinel ||
         (v.y >> 16) == kSentinel || (v.z & 0xFFFFu) == kSentinel || (v.z >> 16) == kSentinel ||
         (v.w & 0xFFFFu) == kSentinel || (v.w >> 16) == kSentinel;
}
__device__ __forceinline__ unsigned canon_w(unsigned w) {
  if ((w & 0xFFFFu) == kSentinel) w = (w & 0xFFFF0000u) | kSentinelFix;
  if ((w >> 16) == kSentinel) w = (w & 0xFFFFu) | (kSentinelFix << 16);
  return w;
}
__device__ __forceinline__ uint4 canon(uint4 v) {
  v.x = canon_w(v.x); v.y = canon_w(v.y); v.z = canon_w(v.z); v.w = canon_w(v.w);
  return v;
}
}  // namespace arx_site
