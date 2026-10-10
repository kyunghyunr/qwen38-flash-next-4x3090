# Power cap and speculation depth: what I measured, and two ways I fooled myself

## Power: 250W per card

Ten-step sweep, all four cards capped together, 100K prefill plus decode at each step:

| cap | prefill 100K | tok/s | wall | clock | efficiency |
|---|---|---|---|---|---|
| 150W | 40.04 s | 2,497 | 589W | 844MHz | 79% |
| 180W | 26.59 s | 3,760 | 708W | 1193MHz | 99% |
| 200W | 21.81 s | 4,585 | 782W | 1398MHz | 109% |
| 220W | 18.43 s | 5,425 | 858W | 1553MHz | 118% |
| **250W** | **15.36 s** | **6,508** | 972W | 1774MHz | **125%** |
| 280W | 14.68 s | 6,809 | 1085W | 1865MHz | 117% |
| 300W | 14.37 s | 6,958 | 1158W | 1917MHz | 112% |
| 330W | 14.08 s | 7,101 | 1269W | 1978MHz | 104% |
| 360W | 13.90 s | 7,192 | 1348W | 2007MHz | 100% |
| 390W | 13.93 s | 7,179 | 1339W | 2005MHz | 100% |

Prefill is power-bound. Sampling all four cards every 5 seconds during a 261K prefill shows them
pinned at 249W with clocks between 1635 and 1890MHz the whole way through, then dropping to ~110-135W
the moment prefill ends and decode starts. That is also why an earlier belief of mine ("prefill is
not resource-bound, it is a pipeline bottleneck") was wrong: I had sampled the cards after the
prefill finished, because the commands ran sequentially.

The 220W row explains a reference discrepancy. A public rig reported 4,944 tok/s prefill where we
measured 6,716 on the same recipe; they were capped at 220W and we were not.

**The cap does not persist on its own.** With persistence mode disabled, `nvidia-smi -pl` is reverted
to the card defaults when the last client holding the GPU exits. Mine reverted during an unrelated
container restart and I measured a phantom regression for an evening. `nvidia-smi` prints a warning
about this when you set the limit; it is easy to skim past. If your numbers move on their own, check
the cap before you check the code.

## Speculation: k=3

Eight real tasks, thinking on, max 512 tokens, c1/c4/c8, two repetitions, acceptance from
`vllm:spec_decode_*` counters:

| k | c1 | c4 | c8 | acceptance | per-position accept | tokens/step | KV pool |
|---|---|---|---|---|---|---|---|
| 1 | 115.0 | 364.3 | 587.7 | 84.0% | 84.0 / - / - / - | 1.840 | 870,209 |
| 2 | 130.9 | **381.9** | 603.9 | 71.7% | 81.1 / 62.4 / - / - | 2.435 | - |
| **3** | **167.7** | 368.8 | **611.7** | 63.5% | 80.4 / 62.6 / 47.7 / - | 2.906 | 806,792 |
| 4 | 144.6 | 381.3 | 449.6 | 55.7% | 81.0 / 61.8 / 46.1 / **34.1** | 3.229 | 774,182 |
| k=3 remeasured | 163.6 | 381.1 | 598.7 | 63.3% | 80.7 / 62.1 / 47.0 / - | 2.898 | 806,792 |

- Single stream (my actual use): k=3 by 27% over k=2 and 44% over k=1. Repeat runs put noise at ±2-3%,
  so this is outside it.
- At c=4 and c=8, k=2 and k=3 are equivalent within noise; k=4 loses badly at c=8 because the fourth
  draft is accepted only 34% of the time.
- k=1 has the largest KV pool (870,209 tokens) because fewer draft tokens are reserved for
  verification, but it is 44% slower, so k=3 is the right trade.
- A report that k=1 wins on NVFP4 hardware did not transfer here. Do not generalize across quant
  formats; measure on your own setup.

## Two measurement bugs that dwarfed the result

Both inflated or deflated throughput by roughly 3x, and both are easy to repeat:

1. **Counting chunks instead of tokens.** MTP emits 2.4-3.5 tokens per chunk. Counting streamed chunks
   as tokens under-reports by that factor. Use `stream_options: {"include_usage": true}` and read the
   usage block.
2. **Missing the thinking stream.** With thinking enabled, reasoning arrives in `delta.reasoning`, not
   `delta.reasoning_content`. Miss it and the whole reasoning phase is excluded, which inflates decode
   by the same factor. My first pass reported 526 tok/s before I caught it.

An independent cross-check is worth building in: the corrected benchmark's c1 (167.7) and TTFT
(0.156 s) matched numbers measured a different way (176-192 and 0.16 s), which is the only reason I
trust either.
