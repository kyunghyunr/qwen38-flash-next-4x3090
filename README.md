# Qwen3.8-Flash-Next on two 3090s: serving the PLE table from disk

Notes from a box I actually run. Qwen3.8-Flash-Next is a 180B-parameter MoE with a 51.2B-parameter
per-layer embedding (PLE) table bolted on. The community calls the language model "125B" and the
number is right if you exclude the table: 180B total minus 51.2B of PLE is 128.8B, and the PLE rows
are looked up, not multiplied. So the model is a 125B MoE plus a 102 GB lookup table in bf16.

That table is the whole problem. Two 24 GB cards can hold the MoE with expert offload; nothing on
this machine can hold 102 GB of bf16 embeddings in RAM alongside everything else, and quantizing it
to FP8 only gets you to 51 GB, which still eats a 96 GB box whole. What follows is how I got the
table out of RAM entirely and served it from disk, and what that costs.

Numbers from this box:

| what | value |
|---|---|
| decode, INT4 PLE sidecar, MTP depth 3, WNA16 | 36-45 tok/s |
| decode, base config (no MTP, no WNA16) | ~26 tok/s |
| context | 262,144 tokens |
| PLE table, bf16 / FP8 / INT4 | 102 GB / 51 GB / 25.6 GB |
| PLE table RAM residency with the sidecar | ~40 MB (row cache only) |
| MTP acceptance | normal (18 drafts accepted in the run I logged) |
| cards | 2x 24 GB, tensor parallel 2 with expert parallel |

The server is up and serving a 256K context at those speeds, but the decode number is a range
because it depends on how many n-gram rows miss the row cache. Prefill is not the interesting part
of this setup and I did not benchmark it carefully, so I'm not going to quote a number for it.

## Hardware

- 2x 24 GB cards (RTX 3090 class), `CUDA_VISIBLE_DEVICES=0,1`, TP=2 with `--enable-expert-parallel`
- 96 GB RAM container on a Threadripper PRO 3975WX host, Proxmox LXC, no P2P between the cards
- The model lives on the container's own disk. Nothing else significant runs on those cards.

The RAM figure is the one that matters. The same checkpoint in this recipe is usually documented
against a 128 GB machine; the point of the changes below is that 96 GB is enough once the PLE table
stops being resident.

## Software

- The recipe image `qwen38-flash-next-2x3090:locked`, from
  [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090)
  (Apache-2.0). That repo is where the working launch layout, the vLLM fork flags (`--moe-backend
  humming`, `--offload-backend uva`, `--cpu-offload-params experts`, the WNA16 hot cache) and the
  `docker_serve.sh` wrapper came from. I did not write those and I'm not redistributing them.
- Checkpoint: [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE),
  revision `ef554143`, plus a compact MTP checkpoint mounted at `runtime/mtp-int4-g32`.
- PLE overlay: [primitive-ai/Qwen3.8-Flash-Next-PLE-quant](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant).
  Their overlay replaces three files in the image (`worker.py`, `ple_layer.py`, `connector.py`).
  The quantized table machinery in `serving/patch_int4_mrv1.py` is lifted from their
  `worker_image_quant.py`; see Licenses below.
- Vision is on, `--limit-mm-per-prompt '{"image": 20}'`. Two images at once is what I tested.

## Why the PLE table decides everything

A PLE is a lookup, which normally makes offloading it cheap: you pay one gather per token per layer
and nothing else. That is not what happens here. The table is indexed by n-gram ids, and a chat
prompt contains a lot of repeated n-grams, so the lookups are extremely concentrated, but the table
itself is 320M rows by 160 dimensions, which is 102 GB in bf16. You cannot page that in and out per
request.

So there are three ways to hold it:

1. bf16 in RAM: 102 GB. Not possible here.
2. FP8 in RAM: 51 GB of the 96 GB box, resident, forever. This works and it is what I ran first.
   It is also why the machine could not do anything else, and why a smaller box can't run it at all.
3. Quantized on disk, memory-mapped, dequantized per lookup, with a small row cache. This is the
   sidecar in this repo.

Option 3 is the one that makes the difference between a 128 GB machine and a 96 GB machine, and in
principle between 96 GB and 64 GB.

## What the sidecar does

`serving/patch_int4_mrv1.py` patches the *stock* path rather than replacing it. The overlay from
primitive-ai replaces the connector and protocol wholesale and that route did not work on 2x3090
(see Dead ends). This patch keeps the stock connector, the stock protocol, the capture path and
MTP3 exactly as they are, and swaps only the table serving:

- It splices the quant table class and the disk readers from the overlay's `worker_image_quant.py`
  into the stock `worker.py`, and attaches the table when `VLLM_PLE_QUANT_DIR` is set.
- It drops the checkpoint's PLE tensors before they are written into what is now a meta-device stub,
  and removes them from the missing-parameter check. Without the meta stub you would allocate 51 GB
  of scratch to hold tensors you are about to throw away.
- It puts a dequantizing `gather_into` on the embedding's forward path, so the layer sees rows of
  bf16 exactly as before.
- It adds a row cache on top of that `gather_into`. Repeated n-grams are the common case, so one
  dequant per distinct row instead of one per lookup. The cache is capped at 128K rows (~40 MB) with
  FIFO eviction.

That last part has a wart worth recording: my first version cleared half the cache when it filled
up. That is an O(n) pass over 65K entries inside a loop that runs per chunk, which cost enough
operations to look like a hang on prefill. FIFO eviction is O(1) and the problem went away. If you
write your own cache here, evict one row at a time.

`tools/ple-to-fp8.py` is what I used to build the table in the layout the loader expects: per-tensor
scale `max|w| / 448` stored as a shape-`[1]` bf16 tensor next to the FP8 shard. `tools/finalize-ple-layout.py`
cleans up two things that will bite you if you assemble the table yourself (below).

## Gotchas when you build your own PLE table

- The loader globs PLE files by name. If you write MTP tensors into a file called `plefp8-*.safetensors`,
  the loader may pick it up and fail. Rename those shards to the model's `model-mtp-*` pattern, and
  delete the 16-byte placeholder shards rather than leaving them in the directory.
- A merged safetensors file I built had a header that declared only the last shard, so splitting by
  `data_offsets` from the header gave the wrong byte ranges. Splitting by fixed byte range
  (`rows * cols` per shard) was correct and I verified the file size matched to the byte before
  writing anything.
- `weight_scale` has to keep the shape the checkpoint uses (here `[1]`, per-tensor). A different
  shape takes a different code path in the loader.
- One global scale per table is not free. Shards with a small amax spend fewer exponent levels, so
  effective precision varies across the table. `tools/` includes the scripts I used to look at that
  distribution; run them on your own table before choosing global vs per-shard scales.

## What broke

**The debug log invalidated CUDA graph capture.** Speculative prefill and decode graph capture kept
failing with `cudaErrorStreamCaptureInvalidated`, and the obvious suspects (the connector, the
speculator, non-exact QSA) were all innocent. `CUDA_LAUNCH_BLOCKING=1` finally named the op. The
cause was `VLLM_LOGGING_LEVEL=DEBUG`, which I had set myself while diagnosing something else: a DEBUG
log line rendered an IR op's tensor argument with `str(tensor)`, that evaluates `torch.masked_select`,
and launching a kernel mid-capture invalidates the capture. Production runs at INFO, so this only
ever happened while debugging. If you are chasing capture failures, turn your own debug logging off
before you believe anything else. The full trail is in `notes/cudagraph-capture-vs-debug-logging.md`.

**`--cudagraph_capture_sizes:[1]` silently skips capture** in this fork, and eager MTP collapsed
decode to 5-7 tok/s. That looked like a speculator bug and was a config bug.

**MTP3 with non-exact QSA blocked on warmup.** The persistent top-k dispatch spun on TP0 while TP1
and the CPU worker sat idle. `VLLM_QSA_EXACT_TOPK=1` uses the non-spinning kernel and warmup
completed. Worth knowing before you bisect the speculator.

## Dead ends

- **Replacing the connector (the overlay's "mrv2" route)**: the CPU worker gets a CUDA context and
  OOMs a 24 GB card, and eager MTP fell apart. Patching only the stock worker and `ple_layer.py`
  was the route that survived. Two of the three symptoms I originally blamed on this route turned
  out to be the debug logging above, so treat my first write-up of it with suspicion.
- **The mixed-NVFP4 community checkpoint**: the NVFP4-to-marlin conversion OOMs on 2x3090. It wants
  roughly 96 GB of VRAM, not system RAM, so it is out of reach here.
- **Serving the FP8 table instead of INT4**: this works, it is just 51 GB resident. If your box has
  the RAM, do that and skip this repo entirely.

## Quality checks

PLE quantization is not obviously free. primitive-ai measured their own FP8 / INT4 / e2m1 tables as
equivalent within noise (knowledge 92.2, tool calling 79.2 / 77.7 / 78.2 / 78.7, ±1.5), which is why
I was willing to run INT4. I checked output sanity on this box: Korean 받침 (final-consonant)
questions, which the smaller model in the same family gets wrong occasionally, came back correct,
and arithmetic and tool calls looked normal. MTP acceptance stayed in the usual range, which is a
decent canary: a broken table degrades drafts first.

Separately, I ran a program-graded comparison between the 125B (this box) and the W4A16 27B of the
same family on a different machine, because I wanted to know whether the big model earns its keep
for daily work. The harness has 21 tasks; 19 of them ran before I stopped it. `bench/RESULTS.md` has
the table. The short version:

- At `effort=medium` both models scored 100% on everything that completed (67/67 and 56/56; the
  totals differ because I stopped the run early, not because of failures).
- `effort=xhigh` was *worse*, not better: on the three hard tasks both models produced no usable code
  in 20K tokens of budget, while the same tasks passed at medium. Overthinking is a real failure mode.
- The 27B used 1.4-2.1x the tokens and 1.3-2.0x the wall time for the same answers.
- The task set is too easy to separate the models. That is a limitation of my harness, not a finding
  about the models.

## Licenses, and what is not in this repo

I ship no model weights and no third-party source files. Specifically not included: the checkpoint
(W4A16 or its PLE table in any precision), the recipe's container image, primitive-ai's overlay
files, any GGUF, and any speculative-decoding drafter whose license is not permissive. One drafter I
looked at is CC BY-NC-ND, which forbids redistribution and derivatives, so it is not here and I am
not going to re-host it.

- Code in `serving/` and `tools/` that I wrote: MIT, see `LICENSE`.
- `serving/serve-int4-ple.sh` is derived from the Apache-2.0 `serve-container.sh` in
  DominikBucko/qwen38-flash-next-2x3090. The header lists what I changed. Keep the attribution if
  you pass it on.
- The table-serving machinery in `serving/patch_int4_mrv1.py` is spliced from primitive-ai's
  Qwen3.8-Flash-Next-PLE-quant overlay (Apache-2.0 as tagged on Hugging Face). The port to the stock
  path, the meta stub, the drop-before-load logic and the row cache are mine.
- Model weights are under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
  It is permissive for use, modification and redistribution, with two conditions worth reading
  before you build anything commercial on it: keep the notice, and if you run the model as a service
  or ship an AI coding/office assistant product, you need a separate license from Qwen. Internal use
  is exempt. I am not a lawyer and this is not advice.

## Credits

The parts that worked are mostly other people's work, stacked: turboderp's vLLM fork features and
the recipe image, DominikBucko's 2x3090 launch layout, the W4A16 checkpoint and its FP8 PLE from
albucino, the quantized-PLE idea and overlay from primitive-ai, and the Qwen team for the model.
What is here is the port of that overlay onto the stock path, the row cache, and the measurements.
