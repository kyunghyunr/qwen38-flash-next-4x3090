#!/usr/bin/env python3
"""Convert a bf16 PLE table to FP8 E4M3, in the layout the loader expects.

The PLE table is the 320M x 160 n-gram embedding that Qwen3.8-Flash-Next keeps per layer. In bf16
it is 102 GB, which is the reason this repo exists. FP8 halves it; INT4 quarters it.

Encoding, matching what the checkpoints in the wild use:
  - per-tensor scale = max|w| / 448.0, stored as a bf16 tensor of shape [1] with the key
    `<tensor>.weight_scale` next to the quantized tensor
  - MTP and any non-PLE tensors are copied through unchanged
  - output is sharded into files of roughly --chunk-gb, plus manifest.json mapping tensor -> file

If your loader wants a different scale shape (per-row, per-shard), change it here and check the
loader's code path: shape decides which branch runs.

Usage:
  python3 ple-to-fp8.py --source /path/to/model-mtp-merged.safetensors --out /path/to/plefp8
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="safetensors file holding the bf16 PLE tensors")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--chunk-gb", type=float, default=1.5, help="target shard size in GB")
    ap.add_argument("--key-filter", default=".ple.", help="substring that marks PLE tensors")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chunk_bytes = int(args.chunk_gb * 1_000_000_000)
    print("source:", args.source, flush=True)

    with safe_open(args.source, framework="pt") as f:
        keys = list(f.keys())
        ple = [k for k in keys if args.key_filter in k]
        print(f"tensors: {len(keys)} total, {len(ple)} PLE", flush=True)

        buf, buf_bytes, shard, mapping = {}, 0, 0, {}

        def flush():
            nonlocal buf, buf_bytes, shard
            if not buf:
                return
            name = os.path.join(args.out, f"plefp8-{shard:05d}.safetensors")
            save_file(buf, name, metadata={"format": "pt"})
            for k in buf:
                mapping[k] = os.path.basename(name)
            print(f"  wrote {name} ({len(buf)} tensors, {buf_bytes / 2**30:.2f} GiB)", flush=True)
            buf, buf_bytes, shard = {}, 0, shard + 1

        for i, k in enumerate(keys, 1):
            t = f.get_tensor(k)
            if args.key_filter in k and k.endswith(".weight") and t.dtype == torch.bfloat16:
                w = t.to(torch.float32)
                scale = (w.abs().amax() / FP8_MAX).clamp(min=1e-12)
                q = (w / scale).to(torch.float8_e4m3fn)
                buf[k] = q.contiguous()
                buf[k + "_scale"] = scale.to(torch.bfloat16).reshape(1)
                buf_bytes += q.numel() + 2
                del w, q
            else:
                buf[k] = t.contiguous()
                buf_bytes += t.numel() * t.element_size()
            if buf_bytes >= chunk_bytes:
                flush()
            if i % 20 == 0:
                print(f"  ..{i}/{len(keys)}", flush=True)
        flush()

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(mapping, fh)
    total = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)
        if f.endswith(".safetensors")
    )
    print(f"done: {len(mapping)} tensors, {total / 2**30:.1f} GiB, {shard} files", flush=True)


if __name__ == "__main__":
    main()
