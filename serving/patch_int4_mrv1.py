"""Serve an INT4 PLE table on the stock (unpatched connector) path.

Why this exists: primitive-ai's PLE-quant overlay ships full replacement files for
`vllm/v1/ple_offload/worker.py`, `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` and
`vllm/v1/ple_offload/connector.py`. On 2x3090 the replaced connector path did not survive
(the CPU worker gets a CUDA context and OOMs a 24 GB card), and the third file is only needed
because the overlay expects a newer connector protocol. This script instead edits the *stock*
files in place, so the connector, protocol, CUDA graph capture and MTP3 stay exactly as the
recipe image ships them, and only table serving changes.

It edits two files inside the container:
  1) vllm/v1/ple_offload/worker.py
     - splices the quant table class and disk readers out of the overlay's worker_image_quant.py
       (function `_ple_disk_shard_of` through `class PleOffloadRunner`)
     - when VLLM_PLE_QUANT_DIR is set: attaches the disk table per offloaded layer, drops the
       checkpoint's PLE tensors before they land in the meta stub, and excludes those parameters
       from the missing-parameter check
     - replaces gather_into with a row-cache-aware version (see the cache comment below)
  2) vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
     - builds the embedding under a meta device when VLLM_PLE_QUANT_DIR is set, so no 51 GB of
       scratch is allocated for tensors that are about to be discarded
     - routes the forward pass through the quantized gather when the table is attached

Requires, in /model/runtime/ple-overlay (mounted from the host, not redistributed here):
  worker_image_quant.py   from primitive-ai/Qwen3.8-Flash-Next-PLE-quant (Apache-2.0)
and this file, patch_int4_mrv1.py (MIT).

Both edits are idempotent: the script detects its own markers and skips work already done.
Set PATCH_INT4_NO_META=1 to skip the meta stub and load the table on CPU instead. That was a
diagnostic switch while chasing capture failures, not a supported configuration.
"""
import os
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
OVERLAY = "/model/runtime/ple-overlay"

changed = False

# ============ 1) worker.py ============
worker_p = f"{SP}/vllm/v1/ple_offload/worker.py"
w = open(worker_p).read()

if "_PleQuantTable" not in w:
    ov = open(f"{OVERLAY}/worker_image_quant.py").read()
    start = ov.index("def _ple_disk_shard_of")
    end = ov.index("class PleOffloadRunner:")
    block = ov[start:end].rstrip() + "\n\n\n"
    anchor = "class PleOffloadRunner:"
    assert anchor in w, "worker.py: PleOffloadRunner anchor not found"
    w = w.replace(anchor, block + anchor, 1)

    # attach the disk table right after offload_prefixes is built
    old = '''        offload_prefixes = tuple(f"{name}." for name in offload_layers)
'''
    new = '''        offload_prefixes = tuple(f"{name}." for name in offload_layers)

        quant_dir = _ple_quant_dir()
        disk_complete_tables: tuple[str, ...] = ()
        disk_complete_params: set[str] = set()
        if quant_dir is not None:
            table_prefixes = []
            for layer_name, layer in offload_layers.items():
                pname = _ple_quant_attach(layer_name, layer, quant_dir)
                if pname is None:
                    continue
                full = f"{layer_name}.{pname}"
                disk_complete_params.add(full)
                table_prefixes.append(full.rsplit(".", 1)[0])
            disk_complete_tables = tuple(table_prefixes)
'''
    assert old in w, "worker.py: offload_prefixes anchor not found"
    w = w.replace(old, new, 1)

    # drop table tensors coming from the checkpoint
    old = '''                if mapped_name is not None and mapped_name.startswith(offload_prefixes):
                    matched_checkpoint_tensors += 1
                    yield weight_name, tensor
'''
    new = '''                if mapped_name is not None and mapped_name.startswith(offload_prefixes):
                    matched_checkpoint_tensors += 1
                    if disk_complete_tables:
                        table = _ple_disk_shard_of(mapped_name)
                        if table is None and mapped_name.endswith(
                            (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
                        ):
                            # rows owned by the sidecar: drop before the loader writes a
                            # 51.2B-element tensor into the stub
                            table = mapped_name.rsplit(".", 1)[0]
                        if table is not None and table.startswith(disk_complete_tables):
                            continue
                    yield weight_name, tensor
'''
    assert old in w, "worker.py: offload_only_iter anchor not found"
    w = w.replace(old, new, 1)

    # the sidecar owns these parameters, so they are not missing
    old = '''            missing_offload_params = sorted(
                expected_offload_params.difference(loaded_expected_params)
            )
'''
    new = '''            missing_offload_params = sorted(
                expected_offload_params.difference(loaded_expected_params)
                - disk_complete_params
            )
'''
    assert old in w, "worker.py: missing-param anchor not found"
    w = w.replace(old, new, 1)

    open(worker_p, "w").write(w)
    print("worker.py: INT4 sidecar patch applied")
    changed = True
else:
    print("worker.py: already patched")

# ============ 1b) dequant row cache ============
if "_cache_cap" not in w:
    # cache fields, right after self._lut = None
    old = '''        self.width = width
        self._lut = None
'''
    new = '''        self.width = width
        self._lut = None
        # Dequantized row cache. n-gram rows repeat heavily within a prompt and across
        # requests, so caching a dequantized row removes the dequant and the mmap page
        # fault for every repeat lookup.
        self._cache: dict[int, torch.Tensor] = {}
        self._cache_cap = 131072  # 128K rows x 160 x 2B ~= 40 MB
'''
    assert old in w, "worker.py: _lut anchor not found"
    w = w.replace(old, new, 1)

    # cache-aware gather_into
    old = '''    def gather_into(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        ids = ids.long()
        shard = ids // self.ROWS_PER_SHARD
        local = ids - shard * self.ROWS_PER_SHARD
        order = torch.argsort(shard)
        s_sorted, l_sorted = shard[order], local[order]
        uniq, counts = torch.unique_consecutive(s_sorted, return_counts=True)
        pos = 0
        for s, c in zip(uniq.tolist(), counts.tolist()):
            sel = l_sorted[pos:pos + c]
            rows = self._dequant(s, sel)
            out[order[pos:pos + c]] = rows.to(out.dtype)
            pos += c
'''
    new = '''    def gather_into(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        ids = ids.long()
        shard = ids // self.ROWS_PER_SHARD
        local = ids - shard * self.ROWS_PER_SHARD
        key = shard * self.ROWS_PER_SHARD + local
        cache = self._cache
        hit = torch.zeros(ids.numel(), dtype=torch.bool)
        for i in range(ids.numel()):
            k = int(key[i])
            row = cache.get(k)
            if row is not None:
                out[i] = row
                hit[i] = True
        miss_idx = (~hit).nonzero().flatten()
        if miss_idx.numel() == 0:
            return
        miss_ids = ids[miss_idx]
        mshard = shard[miss_idx]
        mlocal = local[miss_idx]
        mkey = key[miss_idx]
        order = torch.argsort(mshard)
        s_sorted, l_sorted = mshard[order], mlocal[order]
        uniq, counts = torch.unique_consecutive(s_sorted, return_counts=True)
        pos = 0
        for s, c in zip(uniq.tolist(), counts.tolist()):
            sel = l_sorted[pos:pos + c]
            rows = self._dequant(s, sel).to(out.dtype)
            out_idx = miss_idx[order[pos:pos + c]]
            out[out_idx] = rows
            for j in range(c):
                rk = int(mkey[order[pos + j]])
                if len(cache) >= self._cache_cap:
                    # FIFO eviction, O(1). Evicting in batches (clearing half the cache)
                    # is an O(n) pass inside a per-chunk loop and stalled prefill badly
                    # enough to look like a hang. Evict one row.
                    cache.pop(next(iter(cache)))
                cache[rk] = rows[j].clone()
            pos += c
'''
    assert old in w, "worker.py: gather_into anchor not found"
    w = w.replace(old, new, 1)
    open(worker_p, "w").write(w)
    print("worker.py: dequant row cache patch applied")
    changed = True
else:
    print("worker.py: row cache already patched")

# ============ 2) ple_layer.py ============
ple_p = f"{SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
s = open(ple_p).read()

if "_ple_quant" not in s:
    # imports
    if "\nimport os\n" not in s:
        s = s.replace("import math\n", "import math\nimport os\n", 1)
    if "nullcontext" not in s:
        s = s.replace(
            "from collections.abc import Iterable, Sequence",
            "from collections.abc import Iterable, Sequence\nfrom contextlib import nullcontext",
            1,
        )

    # meta stub. PATCH_INT4_NO_META=1 loads the table on CPU instead (diagnostic only)
    if os.environ.get("PATCH_INT4_NO_META") == "1":
        pass
    else:
        old = '''        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config,
                f"{prefix}.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            ),
        )
'''
        new = '''        # With the INT4 sidecar (VLLM_PLE_QUANT_DIR) the bf16/FP8 table is stubbed out on
        # the meta device: the checkpoint tensors are dropped, not loaded and freed.
        table_device = (
            torch.device("meta") if os.environ.get("VLLM_PLE_QUANT_DIR") else nullcontext()
        )
        with table_device:
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim,
                params_dtype=params_dtype,
                padding_size=divisor,
                prefix=f"{prefix}.ngram_embedding",
                quant_method=_get_ple_embedding_quant_method(
                    quant_config, f"{prefix}.ngram_embedding"
                ),
            )
'''
        assert old in s, "ple_layer.py: embedding constructor anchor not found"
        s = s.replace(old, new, 1)

    # forward: quantized gather when attached
    old = '''        ngram_ids = torch.cat(id_blocks, dim=-1)
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)
'''
    new = '''        ngram_ids = torch.cat(id_blocks, dim=-1)
        quant = getattr(self.ngram_embedding, "_ple_quant", None)
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            if quant is not None:
                quant.gather_into(
                    ngram_ids.reshape(-1), output.reshape(-1, self.head_dim)
                )
            else:
                torch.index_select(
                    self.ngram_embedding.weight,
                    0,
                    ngram_ids.reshape(-1),
                    out=output.reshape(-1, self.head_dim),
                )
            return output
        if quant is not None:
            flat = torch.empty(
                ngram_ids.numel(),
                self.head_dim,
                dtype=torch.bfloat16,
                device=ngram_ids.device,
            )
            quant.gather_into(ngram_ids.reshape(-1), flat)
            return flat.view(*ngram_ids.shape, self.head_dim).flatten(-2)
        return self.ngram_embedding(ngram_ids).flatten(-2)
'''
    assert old in s, "ple_layer.py: forward_impl anchor not found"
    s = s.replace(old, new, 1)

    open(ple_p, "w").write(s)
    print("ple_layer.py: INT4 sidecar patch applied")
    changed = True
else:
    print("ple_layer.py: already patched")

sys.exit(0 if changed else 0)
