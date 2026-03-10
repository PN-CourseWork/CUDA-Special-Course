# Exercises for week 3 - CUDA reduction with Warp shuffles & Shared memory

## Ex 1 — CUDA baseline kernel (`ex1.cu`)
Baseline CUDA kernel for sum reduction that uses `atomicAdd` as described in the slides.
Every thread calls `atomicAdd` directly on the global result — causes extreme serialization.

## Ex 2 — Warp shuffles (`ex2.cu`)
Modify the baseline kernel to use warp shuffles (`__shfl_down_sync`).
After 5 shuffle steps lane 0 of each warp holds the warp partial sum, and only
lane 0 calls `atomicAdd` — 32× fewer atomics than the baseline.
Calculate the speed-up from this modification.

## Ex 3 — Warp shuffles + shared memory (`ex3.cu`)
Modify the warp-shuffle kernel to use shared memory to accumulate the per-warp
partial sums within a block (slide 11):

1. **Phase 1** — warp shuffle in registers (same as ex2)
2. **Phase 2** — lane 0 of each warp writes its partial sum to `__shared__` memory;
   `__syncthreads()` to ensure all writes are visible; the first warp then reduces
   the `BLOCK_SIZE/32` shared-memory values with another shuffle pass
3. **Phase 3** — thread 0 of the block calls `atomicAdd` — 1 atomic per block
   (vs 8 per block in ex2 with BLOCK_SIZE=256)

Calculate the speed-up vs both ex1 and ex2.

## Ex 4 — Three-way benchmark (`ex4.cu`)
Runs all three kernels back-to-back and prints a speed-up table:

```
baseline  : sum = 16777216  time = ...  BW = ...
warp shfl : sum = 16777216  time = ...  BW = ...
warp+smem : sum = 16777216  time = ...  BW = ...

speed-up warp vs baseline  : X.Xx
speed-up smem vs baseline  : X.Xx
speed-up smem vs warp-only : X.Xx
```

## Notes

- All kernels use `N = 1 << 24` (16M doubles) and `BLOCK_SIZE = 256` (8 warps/block)
- The three-phase reduction hierarchy (slide 10):
  - Phase 1: registers (warp shuffle)
  - Phase 2: shared memory (block reduction)
  - Phase 3: global memory (one `atomicAdd` per block)
- `__syncthreads()` is required between the smem write (Phase 2) and the smem read (Phase 2b)
- Use `-1` (`0xffffffff`) as the full-warp mask for `__shfl_down_sync`
