# Third-party components

No model weights and no third-party source files are redistributed in this repository. This file
records what the code here depends on, what it was derived from, and under which terms, so that
anyone copying it can get the attribution right.

This repository covers two phases of one machine: the current 4-card production setup and the earlier
2-card, RAM-constrained setup. The production stack belongs to halt95; see the table below.

## Code and writing in this repository

| path | author | license |
|---|---|---|
| `serving/patch_int4_mrv1.py` | this repo | MIT |
| `serving/serve-int4-ple.sh` | derived from DominikBucko/qwen38-flash-next-2x3090 | Apache-2.0 |
| `serving/README.md` | this repo | MIT |
| `tools/ple-to-fp8.py` | this repo | MIT |
| `tools/finalize-ple-layout.py` | this repo | MIT |
| `notes/*.md`, `bench/RESULTS.md`, `README.md` | this repo | MIT (my own measurements and prose) |

`serving/serve-int4-ple.sh` is a modified copy of `serve-container.sh` from
[DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090),
which is Apache-2.0. Per section 4(b) of that license, the changes are stated in the file header: the
optional INT4 sidecar hook, the image-per-prompt limit, the MTP checkpoint default, and the optional
sampling-defaults patch. The container entry point hook, the env contract and the vLLM flags are
unchanged from upstream.

The table-serving machinery inside `serving/patch_int4_mrv1.py` (the `_PleQuantTable` class, the disk
shard readers, `_ple_quant_attach`) is lifted at run time from `worker_image_quant.py` in
[primitive-ai/Qwen3.8-Flash-Next-PLE-quant](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant),
tagged Apache-2.0 on Hugging Face. The port onto the unpatched worker path, the meta-device stub, the
drop-before-load logic and the dequant row cache are from this repo. That overlay file is not
redistributed here; you mount your own copy.

## Components referenced but not distributed

| component | license | how it is used |
|---|---|---|
| Qwen3.8-Flash-Next weights | Qwen Community License 1.0 | user downloads |
| [halt95/qwen38-flash-next-3090s](https://github.com/halt95/qwen38-flash-next-3090s) v2 | Apache-2.0 | **the 4-card production stack.** Built locally and run from their tree; their image, scripts and container files are not redistributed here |
| Merlin W4A16 checkpoint | derivative of the model license | downloaded and served; not redistributed |
| aikitoria P2P kernel modules (`610.57.04-p2p-v3`) | see upstream | built on the host to enable PCIe P2P; not redistributed |
| [albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) | derivative of the model license | used in the 2-card phase (rev `ef554143` in my run) |
| [primitive-ai/Qwen3.8-Flash-Next-PLE-quant](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) | Apache-2.0 tag; the weights within it remain under the model license | overlay files mounted from the host in the 2-card phase |
| [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) | Apache-2.0 | 2-card recipe and container image, not redistributed |
| vLLM and the forks used by both stacks | Apache-2.0 | the engine; the 2-card patch edits it in place at container start |

Numbers in this repository are my own measurements unless the text labels them as someone else's
published result.

## Model license note

The [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
permits use, modification, redistribution and derivative works, with two conditions: keep the
copyright notice, and if you run the model as a service or ship an AI coding or office-productivity
assistant product, obtain a separate license from Qwen. Internal use that does not expose the model to
third parties is exempt from the second condition. Nothing here grants rights to the weights, and
nothing here is legal advice.

## Deliberately not included

- Any quantized checkpoint, PLE table (bf16, FP8 or INT4), GGUF, or merged safetensors file.
- The container images and setup scripts of the upstream recipes, including halt95's v2 tree.
- Any speculative-decoding drafter whose license forbids redistribution or derivatives. One drafter I
  evaluated is CC BY-NC-ND and is therefore absent, unnamed, and not configured here.
- A derived checkpoint from a safety-modified variant of the model. It is not part of this work and is
  not described here.
