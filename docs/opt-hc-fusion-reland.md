# hc-fusion reland: repeat+mul+add in one pass

The hyper-connection combine (repeat(b) * w + residual) runs as one
REPEAT_MUL_ADD kernel: the broadcast operand is read once and the
[ne0, n_hc, nt] intermediates are never written by separate dispatches.
Bit-exact: the multiply and the add keep their separate roundings, no fma
contraction (the shader marks both `precise`).

## Why it was parked, and what changed

The first attempt (wip/hc-fusion, 35aeaf67b) was reverted for a GPU memory
footprint jump, 75 -> 101 GiB, at `-c 262144 -ub 2048`. The reland keeps the
same kernel and gates; the footprint at production settings is unchanged
(GTT peak 78.2 GiB at d16384 / 81.1 GiB at d65536, identical to the base arm,
`-c` up to 81920, `-ub 512`). The 262144/2048 combination is outside the
production envelope and was not re-measured; if that regime is ever needed,
re-check the GTT peak before trusting the fusion.

## Gates (branch pr/hc-fusion-reland @ 1a5e528df, base 64d011225)

- `test-backend-ops -o FLASH_ATTN_EXT`: 13369/13369 (Vulkan0).
- `test-backend-ops -o REPEAT`: 18/18; `-o REPEAT_MUL_ADD`: 7/7 (includes the
  must-not-fuse variants: repeat reused elsewhere, w not broadcast).
- `llama-ab-run` (micro, raw ids + FNV-1a logits): fingerprints bit-identical
  to the base binary, plain and with the `-rm 32 8` cache mutation.
- Needle probe (full model, passphrase at 10%/60%): PASS at d=32768 (dense)
  and d=98304 (union engaged, stats confirmed).
- Real-corpus pp (full model, 2 reps per depth, `-ub 512`, MTP on):
  d4096 +3.7%, d16384 +3.1%, d65536 +2.0% vs base; GTT peak equal.
- Micro pp (micro-mem-compare, ctx 4096): +5.3/+6.7% (two reps), tg parity.

The fusion is declined when the repeat result or the multiply result has any
other consumer (use-count check), when the shapes are outside the broadcast
pattern, or when types are not f32/contiguous - those sites keep the three
separate kernels.
