#!/usr/bin/env python3
"""Tidy up an assembled PLE sidecar directory, then check it matches the index.

Two things went wrong when I assembled a table by hand, and both fail late and confusingly:

1. MTP tensors ended up in files named `plefp8-*.safetensors`. The PLE loader globs that pattern,
   so it picked up an MTP shard and complained about shapes. The files are renamed to the model's
   own `model-mtp-*` pattern here and the index is remapped to match.
2. Empty placeholder shards (16 bytes each in my assembly) sat in the directory. Nothing referenced
   them, but they were in glob range. They are deleted only after confirming the index does not
   reference them.

Then it validates: every file the index points at exists, all PLE shards 0..N-1 are present, and
each quantized tensor has the dtype, shape and scale the loader expects.

Dry run by default; pass --apply to change anything. The index is backed up before it is rewritten.

Usage:
  python3 finalize-ple-layout.py --model-dir /path/to/checkpoint [--apply]
  python3 finalize-ple-layout.py --model-dir ... --nshards 128 --rows 2500012 --cols 160
"""
import argparse
import collections
import json
import os
import shutil
import struct
import sys

BASE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
PLACEHOLDER_BYTES = 1000  # anything smaller than this is a placeholder, not a table


def hdr_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--nshards", type=int, default=128)
    ap.add_argument("--rows", type=int, default=2500012)
    ap.add_argument("--cols", type=int, default=160)
    ap.add_argument("--dtype", default="F8_E4M3")
    ap.add_argument("--apply", action="store_true", help="actually rename/delete/rewrite")
    args = ap.parse_args()

    d = args.model_dir.rstrip("/") + "/"
    idx_path = d + "model.safetensors.index.json"
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]

    # 1) MTP shards must not sit in PLE glob range
    changed = 0
    for old_base in list({v for v in wm.values() if v.startswith("plefp8-")}):
        if not old_base.startswith(("plefp8-00032", "plefp8-00033", "plefp8-00034")):
            continue
        new_base = old_base.replace("plefp8-", "model-mtp-")
        if not os.path.exists(d + old_base):
            print(f"  skip {old_base} (absent)")
            continue
        n = sum(1 for v in wm.values() if v == old_base)
        if args.apply:
            os.replace(d + old_base, d + new_base)
        for k, v in list(wm.items()):
            if v == old_base:
                wm[k] = new_base
        print(f"  {old_base} -> {new_base} ({n} tensors)")
        changed += n

    if args.apply and changed:
        if not os.path.exists(idx_path + ".bak-premtp"):
            shutil.copy2(idx_path, idx_path + ".bak-premtp")
        tmp = idx_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(idx, f, separators=(",", ":"))
        os.replace(tmp, idx_path)
        print("  index rewritten")

    # 2) drop unreferenced placeholder shards
    for i in range(32):
        p = d + f"plefp8-{i:05d}.safetensors"
        if os.path.exists(p) and os.path.getsize(p) < PLACEHOLDER_BYTES:
            ref = [k for k, v in wm.items() if v == os.path.basename(p)]
            if ref:
                print(f"  !! {p} is referenced ({len(ref)} tensors), not deleting")
                continue
            if args.apply:
                os.remove(p)
            print(f"  drop {os.path.basename(p)}")

    # 3) validate
    files = sorted(set(wm.values()))
    missing = [f for f in files if not os.path.exists(d + f)]
    print(f"index points at {len(files)} files, missing {len(missing)}", missing[:3])

    shards = {}
    for k, v in wm.items():
        if k.startswith(BASE + ".shard_"):
            shards[int(k[len(BASE) + 7:].split(".")[0])] = v
    print(
        f"PLE shards {len(shards)}, range {min(shards)}..{max(shards)}, "
        f"missing {sorted(set(range(args.nshards)) - set(shards))[:5]}"
    )

    bad, cache = [], {}
    for s in sorted(shards):
        f = shards[s]
        if f not in cache:
            cache[f] = hdr_keys(d + f)
        t = cache[f].get(f"{BASE}.shard_{s}.weight")
        if not t or t["dtype"] != args.dtype or t["shape"] != [args.rows, args.cols]:
            bad.append((s, f, t))
    print("shape/dtype mismatches:", bad[:3], f"total {len(bad)}")

    sc = wm.get(BASE + ".weight_scale")
    if sc:
        print("weight_scale:", sc, (cache.get(sc) or hdr_keys(d + sc)).get(BASE + ".weight_scale"))
    print("shards per file:", collections.Counter(shards.values()).most_common())
    return 0


if __name__ == "__main__":
    sys.exit(main())
