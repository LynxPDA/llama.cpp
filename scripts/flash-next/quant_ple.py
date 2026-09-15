#!/usr/bin/env python3
"""Quantise the per_layer_token_embd shard of a split Flash-Next GGUF to Q4_0 (rows of 160 halves: k-quants
need 256-multiples, Q4_0/IQ4_NL need 32). Writes a new shard-2 with the same split metadata, so it pairs with
any shard-1 of the same split. Chunked numpy quantisation, throttled writes (the NVMe rule: no sustained writes)."""
import os
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize

src, dst = sys.argv[1], sys.argv[2]
rate = float(sys.argv[3]) if len(sys.argv) > 3 else 150e6   # bytes/s write throttle
r = GGUFReader(src)
t = [x for x in r.tensors if x.name == 'per_layer_token_embd.weight'][0]
ne0, nrows = int(t.shape[0]), int(t.shape[1])
assert t.tensor_type == GGMLQuantizationType.F16 and ne0 % 32 == 0, (t.tensor_type, ne0)
qtype = GGMLQuantizationType.Q4_0
row_bytes = ne0 // 32 * 18
print(f"{t.name}: F16 [{ne0} x {nrows}] -> Q4_0, {nrows*row_bytes/2**30:.1f} GiB", flush=True)

w = GGUFWriter(dst, arch='qwen4exp', endianess=gguf.GGUFEndian.LITTLE)
w.kv_data = [{}] if hasattr(w,'kv_data') else w.kv_data  # drop the arch key the constructor adds: shard files carry only split.* keys
# split metadata as-is (shard files carry only split.* keys)
for k, f in r.fields.items():
    if k.startswith('GGUF.'): continue
    v = f.contents()
    if f.types[0] == gguf.GGUFValueType.UINT16: w.add_uint16(k, v)
    elif f.types[0] == gguf.GGUFValueType.INT32: w.add_int32(k, v)
    elif f.types[0] == gguf.GGUFValueType.UINT32: w.add_uint32(k, v)
    elif f.types[0] == gguf.GGUFValueType.STRING: w.add_string(k, v)
    else: raise SystemExit(f"unhandled kv {k} {f.types}")
w.add_tensor_info(t.name, [nrows, ne0], np.dtype(np.float16), nrows*row_bytes, raw_dtype=qtype)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
w.fout[0].flush(); w.write_padding(w.fout[0], w.fout[0].tell())
off0 = w.fout[0].tell(); w.fout[0].close()

data = t.data  # memmap [nrows, ne0] f16
chunk = 4_000_000
t0 = time.time(); written = 0
with open(dst, 'r+b') as f:
    f.seek(off0)
    for i in range(0, nrows, chunk):
        blk = np.ascontiguousarray(data[i:i+chunk])
        q = quantize(blk.astype(np.float32), qtype)
        b = q.tobytes(); f.write(b); written += len(b)
        el = time.time() - t0
        if written/el > rate: time.sleep(written/rate - el)
        if (i // chunk) % 8 == 0: print(f"  {i/nrows*100:5.1f}%  {written/2**30:.1f} GiB  {written/(time.time()-t0)/1e6:.0f} MB/s", flush=True)
print(f"done: {written/2**30:.2f} GiB in {time.time()-t0:.0f}s -> {dst}")
