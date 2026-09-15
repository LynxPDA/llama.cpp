# Flash-Next PLE table tools

`per_layer_token_embd.weight` is the n-gram engram table (f16, ~102 GB in the stock GGUF, shard 2 of the split).
Two byte-level transforms make it small enough to sit on the GPU, where the Vulkan backend gathers it in-graph:

1. `quant_ple.py <shard2-f16.gguf> <shard2-q4_0.gguf> [bytes/s]` requantises the table rows to Q4_0
   (rows are 160 halves: k-quants need 256-multiples, Q4_0 is legal). ~28.8 GB out (4.5 bpw of 102 GB f16), one pass, throttled writes.
2. `split_ple_heads.py <shard2-q4_0.gguf> <shard2-q4_0-perhead.gguf> <shard1.gguf> [bytes/s]` splits the single
   tensor into one tensor per indexer head (`per_layer_token_embd.h{h}.weight`) using the head offsets in shard 1,
   so every buffer stays under the 4 GiB Vulkan allocation cap. Byte-range copy, no requantisation.

Pair the new shard 2 with the original shard 1 (same split metadata): copy or symlink shard 1 next to it under the
same `-0000N-of-00002.gguf` naming, and run with `-ot per_layer_token_embd=Vulkan0 LLAMA_PLE_HOST_GATHER=0`.
The tools need the repo's `gguf-py` (resolved relative to this directory) and numpy.
