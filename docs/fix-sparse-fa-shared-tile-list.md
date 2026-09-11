# One index list per dispatch tile: the sparse-FA tiling contract

## Symptom

The sparse mask-compaction Flash Attention path returned plausible-looking but
wrong answers on prefill shapes: a full-model recall probe over a 36k-token
prompt answered "NO" to a question whose answer ("YES") was in the file, while
the same model with `GGML_VK_FA_SPARSE_DISABLE=1` answered "YES". Decode and
small-batch shapes were unaffected.

## Root cause

`flash_attn_base.glsl` resolves **one** sparse index list and **one** mask row
for the whole dispatch tile, not per row:

```glsl
uint32_t qrow = (p.gqa_ratio > 1) ? gqa_iq1 : (i * Br);
sparse_base = (((iq3 % p.nem3) * p.nem2 + (iq2 % p.nem2)) * p.nem1 + qrow) * p.split_kv;
```

with `m_stride == 0` under GQA (every row of the tile shares one mask row, and
`m_offset = gqa_iq1 * m_row_len`). The consumer shaders assume the same thing,
and `flash_attn_cm1.comp` states it outright:

```glsl
// sparse is gqa-gated (m_stride == 0): all four rows share the value
FLOAT_TYPE mv = FLOAT_TYPE(data_m[m_offset + kcol]);
```

That is correct only when the tile's rows are the GQA heads of a **single**
query. The host folds GQA only when `N <= 8`, so any shape with `N > 8` runs
with `gqa_ratio == 1`, where a `Br`-row tile (16, or 64 for cm2) spans 16
*different* tokens: every row then attended the tile's first token's selection
and read its mask row. Silent wrong answers, no crash.

## How it was localized

A full-model A/B is far too slow for this. It reproduces in seconds in
`test-backend-ops` once the test actually sets the sparse hint (the pre-existing
top-k test left `op_params[5]` at 0, so every "sparse" case was silently
measuring dense):

| picks per query row | ERR vs CPU | reading |
| --- | --- | --- |
| different per token (the real case) | 1.85 | broken |
| identical within a 16-row tile | 0.035 | share-the-tile hypothesis |
| identical across the whole batch | 0.00055 (pass) | only row 0 is right |

The monotone fall of the error with the amount of sharing is the signature of a
shared-tile read. Confirming it directly: forcing a single-row tile
(`block_rows = 1`) makes the per-token case pass at nb = 64/128/512 — it is
correct, but ~4x slower than dense (147 vs 607 GFLOPS at nb=512, depth 32768),
so it is not a fix, it is a measurement that pins the cause.

## Fix

Gate the sparse path on the property the tiling actually needs, encoded in the
host as `gqa_ratio > 1` rather than `N >= 64`, and write the contract down next
to the gate. Prefill then declines to dense (correct, slower); decode keeps the
sparse path.

Note the direction of the tradeoff: with the shared tile the prefill numbers
look *good* (it is the same work with 1/Br of the gathers) and the answers are
wrong. Prefer the gate that is provably correct; a per-tile **union** consumer
(one list per tile = the union of its rows' selections, since every row of the
tile then legitimately shares it) is how prefill gets the speed back correctly.

## Reusable checklist

- A shared assumption in a tiled kernel is a correctness condition, not an
  optimization detail. When a tile resolves one index/row/list for all its rows,
  find the shape invariant that makes those rows interchangeable, and gate on
  *that*, not on a proxy (`N`, batch size, depth).
- Before trusting any sparse/perf measurement, verify the path was taken at all:
  env-gated one-line `fprintf` is enough, and it must be removed before commit.
  In this episode the test never set the sparse hint and an upstream comparison
  was measuring its *dense fallback* — a wrong conclusion that survived several
  rounds of reasoning.
- Sweep the amount of sharing (per-token -> per-tile -> per-batch). If the error
  collapses monotonically to zero as sharing grows, the bug is a shared read, not
  a bad index or a stride error.
- Check the generated artifacts get rebuilt: `ggml-vulkan-shaders.hpp` did not
  list the `.comp` files in its `DEPENDS`, so shader edits were not compiled and
  measurements silently described old code. `touch`/remove the generated header
  after editing any shader.
