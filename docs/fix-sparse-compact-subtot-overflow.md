
# Recipe: shared-memory per-subgroup tallies must be sized for the whole workgroup

## Bug

`flash_attn_sparse_compact.comp` (sparse FA mask compaction) kept a
per-subgroup tally in `shared uint sub_tot[8]`, but the pipeline is
created with `compact_wg = min(1024, ...)` threads = **16 subgroups**
on wave-64 GPUs (RADV). Waves 8..15 wrote past the array.

## Symptom

Not a crash: each mask row's compacted index list silently shrank from
~2051 entries to ~930, keeping ascending order (the lost waves' counts
zero out, so slots shift and later writes collide/overwrite). The FA
then attends over a pseudo-random half of the selected cells. The model
stays locally coherent but loses global structure — long-context
analysis hallucinates breaks that do not exist. Bit-diff tests at small
context pass because the sparse path never fires below KV >= 4096 with
the ratio gate.

## Fix

`shared uint sub_tot[32];` (max subgroups = 1024 threads / 32 lanes).
General rule: any shared array indexed by `gl_SubgroupID` must be sized
`max workgroup threads / min subgroup size` (32 on most GPUs, 16 on
wave-64 AMD if the workgroup is <= 512), not by the value you see on
your dev machine.

## Debugging technique

Synchronous or even queued-async host readbacks inside the dispatch
path race the recording (commands are not submitted yet). Snapshot the
debug state at dispatch (`g_sparse_dbg`), then read after the graph's
fence in `ggml_backend_vk_graph_compute` (`GGML_VK_SPARSE_VALIDATE`).
Validation criteria: list length == mask finite count, strictly
ascending, all in-bounds.
