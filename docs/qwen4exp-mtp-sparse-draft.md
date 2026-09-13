# qwen4exp: sparse QSA for the MTP draft block

## Symptom

`llama-server --spec-type draft-mtp` on qwen4exp loses pp512 with context depth far below
the no-draft server: 6936 / 4494 / 2235 / 790 t/s at d0 / 8192 / 32768 / 131072 on the
micro model against 10631 / 6445 / 3943 / 1983 without the draft. `GGML_VK_PERF_LOGGER`
shows why: every prefill ubatch runs a SECOND graph (the draft, `graph_mtp`), and its one
attention layer does a DENSE full-context FA over the whole ubatch (508 rows at ub 512).
That FA grows linearly with depth: ~37.6 ms of a ~49 ms draft window at d32768, ~390 ms
per prefill at d131072.

## Why the draft ran dense, and why that was wrong

`graph_mtp` hardcoded `mctx_hyb = nullptr`, so `build_layer_attn`'s QSA gate
(`mctx_hyb != nullptr && get_idx() != nullptr && dsv4_compress_ratios[il] > 0`) was false,
and the MTP context carried a plain `llama_kv_cache` (the deepseek32 MTP pattern).

The MTP block of this architecture is a full-attention QSA layer:

1. The real sidecar (`Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf`, `blk.48`) ships its own
   indexer tensors (`indexer.{q_proj,k_proj,q_norm,k_norm}`) with values distinct from
   every trunk indexer layer.
2. The indexer norms sit far from their zero initialization, so they were trained: the
   block's attention used the indexer during training.
3. The reference config says `mtp.layer_types = ["full_attention"]`; in this architecture
   every full-attention layer is a QSA layer.

The sidecar's `compress_ratios[48] == 0` is a padding artifact: `llama-model-saver.cpp`
writes the array out to `n_layer_all` from the trunk's array, whose tail is zero. It does
NOT mean the draft runs dense.

## What the fix does

1. **Loader** (`load_arch_hparams`): when a nextn layer has a zero ratio but carries
   `blk.<il>.indexer.q_proj.weight`, restore its ratio from the trunk's QSA layers. A
   genuinely dense MTP block (no indexer tensors) stays dense.
2. **Memory** (`create_memory`): the MTP context gets `llama_memory_hybrid_idx` with the
   trunk's filters inverted - attention + indexer over `il >= n_layer()`, and a recurrent
   filter that is always false. The old comment ("a hybrid memory with an empty recurrent
   layer set fails its buffer allocation") is wrong: an all-false filter leaves the
   recurrent cache without tensors and without buffers, and `find_slot` still succeeds on
   its single spare cell.
3. **Graph** (`graph_mtp`): build `build_inp_mem_hybrid()` and pass the hybrid-idx
   context into `build_layer_attn`, like the mainline graph does.
4. **Prefill-only gate**: `graph_mtp` enables QSA only for `n_tokens >= 16`. A decode or
   verify batch of 1-4 rows pays more for the indexer pipeline (k_proj, pooling, top-k,
   plus the O(n_kv) host scan in `set_input_qsa`) than sparse FA saves: measured on the
   micro model, ~+2-3 ms per draft decode window against ~1.6-2 ms of dense FA, which is
   a ~25% tg loss. Prefill batches amortize the pipeline and win big at depth.
5. **Indexer keys stay written**: when the gate declines, `build_layer_attn` still runs
   one `index_k_proj` matmul + `cpy_k` into the index cache. The pooled-key cache
   recomputes every block above its watermark from the stored keys, so a hole in the key
   cache would poison a later sparse read. The watermark only advances inside
   `set_input_qsa`, so not touching the pool on dense ubatches is safe by construction.

Also in this change: `llm_graph_input_mem_hybrid{,_k,_iswa}::set_input` skip an `s_copy`
input that no node reads (ggml-alloc assigns no buffer to it). The MTP graph reads no
recurrent state, and the old unconditional dereference aborted on it.

## Micro-model measurements (llama-server, pp512 / tg128, same session)

| depth | dense draft pp | QSA draft pp | dense draft tg | QSA draft tg |
| --- | --- | --- | --- | --- |
| 0 | 6258 | 6157 | 124 | 124 |
| 8192 | 4118 | 4579 | 114 | 112 |
| 32768 | 2120 | 2350 | 94 | 87 |
| 131072 | 734 | 1019 | 53 | 49 |

Prefill is at or above the dense draft at every depth and +39% at d131072; the draft's
per-ubatch FA is depth-independent once the union takes (the sparse draft attends the
compact union, not the whole cache). Generation time stays within a few percent of the
dense draft.

## Open question for the real model

The micro model's weights are random, so the draft's indexer produces top-k selections
with only 19-39% overlap between adjacent tokens; the union gate then declines at
shallow/mid depth and the sparse path pays the indexer without the FA win there. The
trunk's random indexer reaches 93% overlap and the real trunk measures ov=86, so trained
weights should compact much tighter. Before calling this done on the real model:

1. A/B acceptance rate and tg of the sparse draft against `GGML_VK_FA_SPARSE_DISABLE=1`
   on the same server (the draft only proposes; the target verifies, so the text is
   identical either way - only the acceptance rate moves).
2. Check the union gate engages for the draft from shallow depth
   (`GGML_VK_FA_UNION_STATS`), which is where the remaining prefill win lives.
