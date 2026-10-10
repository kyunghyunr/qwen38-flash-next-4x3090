# Qwen3.8-Flash-Next on 4x 3090: the numbers, and what actually unblocked it

Working notes from the box I run daily. Qwen3.8-Flash-Next is a 180B MoE with a 51.2B-parameter
per-layer embedding table, served here on four 24GB cards at 262,144 context with MTP speculation.

Current production numbers:

| what | value |
|---|---|
| decode, reasoning / repeat / code | 192 / 186 / 163 tok/s |
| decode, mixed tasks at c=1 (wall, incl. TTFT) | 167.7 tok/s |
| prefill, 13K / 100K / 261K | 2.01 s (6,456 tok/s) / 6,508 tok/s / 38.5 s |
| TTFT | 0.16-0.18 s |
| KV pool | 806,792 tokens |
| context and vision | 262,144 tokens, up to 100 images, video enabled |
| MTP | k=3, 63.5% acceptance, 2.906 tokens emitted per step |
| restart | 210 s warm, 670 s cold |

The stack that produces these numbers is not mine. It is
[halt95/qwen38-flash-next-3090s](https://github.com/halt95/qwen38-flash-next-3090s) v2 (Apache-2.0)
running the Merlin W4A16 checkpoint, and I feed it a 4x3090 box. What is mine is the host and
container work needed to make it run at all on this machine, the power and speculation tuning, and
the measurements. Two of those findings are the reason this repo exists, because both cost me days:

1. **NCCL would not use P2P until the plain 3090's 256MB BAR1 was fixed.** Raw `cudaMemcpy` P2P
   worked at 23.9 GB/s while `ncclCommInitRank` died with `unhandled cuda error`. A ReBAR-capable
   VBIOS took BAR1 to 32GB on all four cards and NCCL P2P came up. Prefill doubled.
2. **The power cap silently reverts when the last GPU client exits unless persistence mode is on.**
   `nvidia-smi -pl` warns about this and I did not read the warning carefully enough.

Both are written up in `notes/`.

## Hardware

- 4x 24GB: one RTX 3090 (slot 01) and three RTX 3090 Ti, water cooled, all capped at 250W
- Host: Threadripper PRO 3975WX, Proxmox, driver 610.57.04 with the aikitoria P2P DKMS patch
- P2P verified: `nvidia-smi topo -p2p rw` reports OK for every pair, raw copy 23.9 GB/s
- The model runs in a privileged LXC (32 cores, 96GB RAM), not a VM. That matters, see below.
- All four cards report 32GB BAR1 after the VBIOS work. Before it, the plain 3090 had 256MB.

## The stack

- [halt95](https://github.com/halt95/qwen38-flash-next-3090s) v2 image, built locally from their
  tree. Their scripts and image are not redistributed here.
- Checkpoint: the Merlin W4A16 build of the model, 124GB, with its calibrated FP8 KV scale sidecar.
- Parallelism: TP2 x PP2 with expert parallelism, MTP k=3, `--max-model-len 262144`, chunk 1024,
  FP8 KV, PLE offload.
- Served through vLLM's OpenAI API, answering to five model aliases so that clients did not need
  reconfiguring when the backend changed underneath them.

## How the numbers got here

Same box, same checkpoint family, same task set, in the order I ran it. This is the honest version of
"it got faster": every row is a configuration I actually ran, not a projection.

| stage | prefill 13K | decode (reason / repeat / code) | KV pool |
|---|---|---|---|
| DominikBucko recipe, 2 cards, INT4 PLE sidecar | — | 36-45 | — |
| same recipe stretched to 4 cards (TP4, MTP k=4) | 8.53 s (1,523) | 139 / 106 / 107 | 281,030 |
| halt95 v2 in a VM, no P2P | 4.87 s (2,668) | 118 / 113 / 66 | 806,792 |
| halt95 v2 in the LXC, still no P2P | 4.50 s (2,887) | 176 / 195 / 171 | 806,792 |
| + P2P, BAR1 fixed | **2.01 s (6,456)** | **192 / 186 / 163** | 806,792 |

Two things fall out of this. Moving from a VM to a privileged LXC was worth a lot on its own (decode
118 to 176) even with P2P still broken, and fixing BAR1 roughly doubled prefill again. The KV pool
going from 281K to 806K tokens when I changed stacks changed what I could do with the model more than
the speed did.

## Power: 250W, measured

I swept the per-card power limit in ten steps with a 100K prefill plus decode at each step, all four
cards capped together. Prefill is a power-bound workload: during a 261K prefill all four cards sit
pinned at 249W and the clocks never drop.

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

250W is where tokens per watt peaks. Above it you buy prefill with heat almost linearly, and at 390W
it stops buying anything. The 220W row also answered a puzzle I had for a while: a reference rig
published 4,944 tok/s prefill where we measured 6,716, and the gap was the power cap, not the silicon
or the recipe.

On this hardware the setting does not stick by itself. With persistence mode off, the limit reverts to
the card defaults as soon as the last container holding the GPUs exits. That happened to me during an
unrelated restart, and I spent an evening measuring a "regression" that was really a power cap.

## Speculation: k=3 is the right depth here

I re-checked this after reading that k=1 wins on NVFP4 hardware. It does not transfer. Wall-clock
tok/s over eight real tasks, thinking on, two repetitions, acceptance counted from
`vllm:spec_decode_*` deltas:

| k | c1 | c4 | c8 | acceptance | tokens/step | KV pool |
|---|---|---|---|---|---|---|
| 1 | 115.0 | 364.3 | 587.7 | 84.0% | 1.840 | 870,209 |
| 2 | 130.9 | **381.9** | 603.9 | 71.7% | 2.435 | — |
| **3** | **167.7** | 368.8 | **611.7** | 63.5% | 2.906 | 806,792 |
| 4 | 144.6 | 381.3 | 449.6 | 55.7% | 3.229 | 774,182 |

k=3 wins single-stream by 27% over k=2 and 44% over k=1, and at c=8 k=4 collapses to 449.6 because
the fourth draft is accepted only 34% of the time. Two measurement bugs were worth more than the
result: MTP emits several tokens per chunk, so counting chunks under-reports throughput by about 3x,
and the thinking stream arrives in `delta.reasoning`, not `delta.reasoning_content`, so missing it
inflates decode by the same factor. I found both because a first pass reported 526 tok/s.

## Vision

`--limit-mm-per-prompt '{"image": 100, "video": 1}'`. Images were verified up to 8 in a single request,
and a 3-second 6-frame video returned `multimodal: {'video': 384}` and read the sequence correctly.
Token cost per image is roughly `ceil(H/28) * ceil(W/28)`, capped at 8,000 tokens per image, which is
what really bounds how many you can send at full resolution.

## What did not work

- **`cudagraph_mode: FULL`**: our fork rejects it for this attention backend
  (`CUDAGraphMode.FULL is not supported with Qwen3_8FlashNextQSAFlashAttentionBackend`), and forcing
  it produced `shm_broadcast: No available shared memory broadcast block found in 60 seconds` with the
  API never coming up. The advice to use FULL comes from a different patch tree; `FULL_DECODE_ONLY`
  is what works here.
- **The FP8 KV patch**: not adopted. Its author reverted it in his own tree (-5.7% and -5.5% at 3.8K
  and 30K, +73.8% only past 65K), and the module paths differ from ours, which would have meant
  re-deriving 23 patch anchors plus MTP-aware calibration.
- **NVFP4 checkpoints**: the conversion step wants about 96GB of VRAM, so they are out of reach on
  24GB cards no matter how much system RAM is available.
- **P2P inside a VM**: `nvidia-smi topo -p2p` reports `NS` under KVM with GeForce cards, and the
  community P2P patch gates on the host chipset path. A privileged LXC with the driver on the host is
  what made P2P possible here.
- **`--disable-custom-all-reduce`**: kept on. The custom all-reduce kernel fails with
  `mapping of buffer object failed` on this box, so only the risky kernel is off while NCCL transport
  stays on.

## The 2-card phase, and why it is still in this repo

Before the fourth card, the model ran on two cards with the
[DominikBucko](https://github.com/DominikBucko/qwen38-flash-next-2x3090) recipe, and the binding
constraint was RAM rather than VRAM: the PLE table is 102GB in bf16 and 51GB in FP8, and a 96GB box
cannot afford either alongside everything else. `serving/` and `tools/` are that phase: a patch that
serves a 25.6GB INT4 copy of the table from disk with a memory-mapped reader and a 40MB dequant row
cache, leaving the stock connector, protocol and capture path untouched. It reached 36-45 tok/s,
which is slow, and it is superseded by the 4-card production stack. I keep it because the technique
is the difference between "this model needs 128GB of RAM" and "this model needs a disk", and because
two of its traps (allocating a 51GB table just to discard it, and O(n) cache eviction inside a
per-chunk loop) cost me real time. Read `serving/README.md` before copying anything from there.

`notes/cudagraph-capture-vs-debug-logging.md` is from the same phase: a DEBUG log line rendered a
tensor argument with `str(tensor)`, that launched a `masked_select` kernel inside a capture region,
and every speculative graph capture failed until I turned my own diagnostic logging off.

## Quality check

I compared the 125B against the 27B of the same family on a program-graded task set, to decide whether
the big model earns its keep. Both scored 100% at `effort=medium`, and `xhigh` was actively worse: on
the three hard tasks both models burned 20K tokens and produced no usable code, while the same tasks
passed at medium. The 27B used 1.4-2.1x the tokens for the same answers. My task set is too easy to
separate two models, which is a limitation of the harness, not a finding about them. Table in
`bench/RESULTS.md`.

## Licenses and what is not here

No model weights, no container images, no third-party scripts. Specifically absent: the Merlin
checkpoint, halt95's image and scripts, the DominikBucko recipe files, the PLE table in any precision,
any GGUF, and any drafter whose license forbids redistribution (one I evaluated is CC BY-NC-ND, so it
is not shipped and not named).

- Code I wrote in `serving/` and `tools/`: MIT, see `LICENSE`.
- `serving/serve-int4-ple.sh` derives from DominikBucko's Apache-2.0 `serve-container.sh`; the changes
  are listed in its header.
- The quantized-PLE machinery in `serving/patch_int4_mrv1.py` is spliced at runtime from primitive-ai's
  overlay (Apache-2.0 as tagged). The port onto the stock path, the meta stub and the row cache are mine.
- halt95's v2 stack (Apache-2.0) is the 4-card production path; aikitoria's P2P kernel patch is theirs
  and is not redistributed.
- Model weights are under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE):
  permissive for use and modification, with two conditions worth reading before building anything
  commercial on it. I am not a lawyer.

## Credits

halt95's v2 stack does the 4-card work, DominikBucko's recipe got the model running on two cards,
primitive-ai built the quantized PLE overlay, albucino published the W4A16 checkpoint with an FP8 PLE,
aikitoria's patch is what makes P2P possible on consumer cards, and the Qwen team made the model.
Everything here is either the host work needed to run that stack on this particular machine, the
tuning, or the measurements.
