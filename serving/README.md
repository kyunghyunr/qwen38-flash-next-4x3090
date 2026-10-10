# This directory is the 2-card phase

Everything in `serving/` and `tools/` belongs to the period when this model ran on two 24GB cards,
and the binding constraint was system RAM rather than VRAM. On two cards the recipe needed a 51GB FP8
PLE table resident in a 96GB box, which worked but left no room for anything else. The patch here
serves a 25.6GB INT4 copy of that table from disk instead:

- memory-mapped, dequantized per lookup, with a 40MB row cache for repeated n-grams
- the stock connector, protocol, CUDA graph capture and MTP path are left alone, which is the part
  that took the work: the upstream overlay replaces all three and that route OOMs a 24GB card
- the checkpoint's PLE tensors are dropped into a meta-device stub instead of being loaded into 51GB
  of scratch and then discarded

It reached 36-45 tok/s. That is not the fastest this box ever went on two cards: with the FP8 table
resident in RAM, the same setup decoded at 70-87 tok/s. The sidecar trades roughly half the decode
speed for 51GB of RAM, which is the entire point when RAM is what you are short of. The current
production setup is a different stack on four cards at 163-192 tok/s, described in the top-level
README, and it does not use anything in this directory.

I am keeping it because the technique is the difference between "this model needs 128GB of system
RAM" and "this model needs a disk", and because it is the reason a single 128GB machine can consider
this model at all. If you are RAM-constrained rather than VRAM-constrained, this is the useful part
of the repo.

Files:

| file | what it is |
|---|---|
| `patch_int4_mrv1.py` | rewrites the stock `worker.py` and `ple_layer.py` in place: splices in the quant table from primitive-ai's overlay, attaches the disk table, drops the checkpoint tensors, adds the dequant row cache |
| `serve-int4-ple.sh` | container entry point, derived from DominikBucko's `serve-container.sh` (Apache-2.0). Header lists the changes. |

`serve-int4-ple.sh` expects primitive-ai's `worker_image_quant.py` in `/model/runtime/ple-overlay` and
a quantized table directory mounted at `$VLLM_PLE_QUANT_DIR`; with that variable unset the script
behaves like the ordinary in-RAM FP8 path. See `../NOTICE.md` for what is and is not redistributed.
