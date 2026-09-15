#!/usr/bin/env python3
"""Split the single per_layer_token_embd tensor of a shard into one tensor per indexer head
(per_layer_token_embd.h{h}.weight, rows [offset_h, offset_h+vocab_h)), so each buffer stays under the
Vulkan per-allocation cap (4 GiB on RADV gfx1151) and the table can be GPU-resident. Byte-range copy,
no requantisation. Trailing padding rows past the last head are dropped. Throttled writes."""
import os
import sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf
from gguf import GGUFReader, GGUFWriter
src, dst, main = sys.argv[1], sys.argv[2], sys.argv[3]
rate = float(sys.argv[4]) if len(sys.argv) > 4 else 150e6
r = GGUFReader(src); m = GGUFReader(main)
def ints(k): f = m.fields[k]; return [int(f.parts[i][0]) for i in f.data]
off, voc = ints('qwen4exp.ple.head_offsets'), ints('qwen4exp.ple.head_vocab_sizes')
t = [x for x in r.tensors if x.name == 'per_layer_token_embd.weight'][0]
ne0, nrows = int(t.shape[0]), int(t.shape[1]); qt = t.tensor_type
row_bytes = t.n_bytes // nrows
assert off[-1] + voc[-1] <= nrows, (off[-1] + voc[-1], nrows)
print(f"{t.name} {qt.name} [{ne0} x {nrows}] row {row_bytes} B -> {len(off)} head tensors", flush=True)
w = GGUFWriter(dst, arch='qwen4exp', endianess=gguf.GGUFEndian.LITTLE); w.kv_data = [{}]
for k, f in r.fields.items():
    if k.startswith('GGUF.'): continue
    v = f.contents()
    if f.types[0] == gguf.GGUFValueType.UINT16: w.add_uint16(k, v)
    elif f.types[0] == gguf.GGUFValueType.INT32: w.add_int32(k, v)
    elif f.types[0] == gguf.GGUFValueType.UINT32: w.add_uint32(k, v)
    elif f.types[0] == gguf.GGUFValueType.STRING: w.add_string(k, v)
for h in range(len(off)):
    w.add_tensor_info(f'per_layer_token_embd.h{h}.weight', [voc[h], ne0], np.dtype(np.float16), voc[h]*row_bytes, raw_dtype=qt)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
w.fout[0].flush(); w.write_padding(w.fout[0], w.fout[0].tell()); pos = w.fout[0].tell(); w.fout[0].close()
raw = np.memmap(src, dtype=np.uint8, mode='r')
base = int(t.data_offset)
t0 = time.time(); written = 0; CH = 256 << 20
with open(dst, 'r+b') as f:
    f.seek(pos)
    for h in range(len(off)):
        s0 = base + off[h]*row_bytes; n = voc[h]*row_bytes
        for c in range(0, n, CH):
            b = raw[s0 + c : s0 + min(c+CH, n)].tobytes(); f.write(b); written += len(b)
            el = time.time()-t0
            if written/el > rate: time.sleep(written/rate - el)
        # each tensor's data must start 32-byte aligned: the writer's ti offsets assume padding between tensors
        pad = (-f.tell()) % 32
        if pad: f.write(b'\0'*pad)
        print(f"  head {h}: {n/2**30:.2f} GiB  total {written/2**30:.1f} GiB  {written/(time.time()-t0)/1e6:.0f} MB/s", flush=True)
print(f"done {written/2**30:.2f} GiB in {time.time()-t0:.0f}s -> {dst}")
