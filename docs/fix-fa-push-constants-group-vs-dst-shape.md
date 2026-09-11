
# Recipe: push constants carry the destination tensor's shape, never the dispatch tile's

## Bug

The Vulkan grouped union prefill path (`ggml_vk_flash_attn_union_groups`) processes the
batch in groups of 64 query rows. It sized three things by the *group* while the shaders
expect the *destination tensor's* shape:

1. `vk_flash_attn_push_constants.ne1` was set to the group's row count `N`. `ne1` is the
   head-to-head stride of the output, and `dst` is `[HSV, n_head_q, n_batch, ns]`:

   ```glsl
   // flash_attn.comp / flash_attn_cm1.comp / flash_attn_cm2.comp:
   uint32_t o_offset = (gqa_iq1*p.ne1*HSV + iq3*p.ne2*p.ne1*HSV) / 4;
   data_ov4[o_offset + (iq2*HSV + (i*Br + row)*p.ne1*HSV)/4 + ...] = ...;
   ```

   It must be `q->ne[2]` (the head count). Passing the row count transposes the output
   for every head but the first.

2. The group's slice of `dst` advanced by `dst->nb[1]`. One batch row is `n_head_q*HSV`
   wide, so the batch-row stride is `nb[2]`.

3. The split-K reduce was dispatched as `{N, HSV, neq2}` with
   `ne1 = N, ne2 = neq2` — the opposite of its own convention (`x` enumerates heads,
   `z` the rows of the split buffer, `ne1` is the head stride). Compare the dense call
   site in `ggml_vk_flash_attn`, which is the reference.

## Symptom

`ERR ≈ 1.0` (not a small numeric drift) for `nh != nb`, passing for `nh == nb`; at
`nb=128` a `DEVICE_LOST` from the out-of-range writes. Split-K shapes (`nb=8`, `nb=16`
at depth, where `split_k > 1`) failed *independently* of the `ne1` error, so fixing only
the first one leaves a shape family still broken.

## Fix

- `ne1 = q->ne[2]`, `ne2 = rows` (the group's own row count in the split buffer, matching
  the allocation and the reduce's `ne2`).
- `dst_buf.offset += batch_off * dst->nb[2]`.
- reduce: `{HSV, neq2, N, N, 1, split_k, false}` dispatched over `{neq2, HSV, N}`.

## Debugging technique: bisect the *shape*, and read the rule off the result

The error looked like a shader bug (garbage output), but the decisive observation was
the *pattern* of pass/fail across shapes, not any single failure: `nh=64` passed while
`nh=16/24/32/48` failed at fixed `nb=64`. That is a symmetry, and a symmetry says the
host is passing a quantity that coincides with the right one only in the symmetric case.
Sweeping `nb` separately from `nh` then inverts it (`nb=24, nh=32` fails while
`nb=24, nh=24` passes), which pins the bug to the *pair* rather than either dimension.

Two traps cost most of the time here:

- **A verdict of "PASS" is only meaningful if the path was actually taken.** A dense
  fallback reports PASS. The union gate is a *measurement* (the previous call's overlap
  count), so on the first call it always declines, and `test-backend-ops` computes each
  case once: an entire sweep "passed" while measuring dense. Confirm engagement from a
  counter (`GGML_VK_FA_UNION_STATS=1`), not from a verdict or a timing.
- **A regex `-p` filter over the case's `vars()` string silently matches several cases**
  (or none), so "N/N tests passed" may belong to a case you did not mean to run. Assert
  on the exact expected count (`1/1`), never on the presence of a `tests passed` line
  (which also matches `0/1 tests passed`).

## General rule

A compute shader's push constants describe the **tensors and their logical indexing**;
the dispatch geometry (tile coordinates, workgroup counts) is separate. Anything used
inside an offset expression must be a tensor invariant. When a path tiles a tensor and
reuses the same push-constant struct, the tiled quantity belongs only in the *dispatch*
fields, never in the tensor-shape fields.
