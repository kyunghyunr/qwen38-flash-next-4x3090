# P2P on GeForce 3090s: the 256MB BAR1 was the blocker

This is the longest-running problem I had with this box, and the fix was a hardware-identity issue
rather than a software setting.

## The symptom

Prefill sat about 55% below the reference numbers for the same recipe, and nothing in the vLLM flags
moved it. Meanwhile `nvidia-smi topo -p2p rw` said `NS` (not supported) for every pair, and NCCL fell
back to host transport.

## Stage 1: P2P is impossible in a VM

The model was running inside a KVM guest with the GPUs passed through. Under KVM with GeForce cards,
peer-to-peer over PCIe is not exposed at all: `nvidia-smi topo -p2p rw` returns `NS`. The community
P2P patch gates on the host chipset path and is intended for bare metal or LXC. Moving the workload
into a privileged LXC with the driver on the host is what changed this, specifically:

- host driver userspace 610.57.04 with the aikitoria P2P kernel modules built from source for the
  running kernel
- the VM that owned the GPUs set to `onboot: 0`, because it re-grabbed the devices through vfio on
  every boot and the driver could not load
- vfio binding commented out and the GPUs unbound so `nvidia-smi -L` sees all four

After that `nvidia-smi topo -p2p rw` reported OK for all six pairs and a raw `cudaMemcpy` between
peers ran at 23.9 GB/s.

## Stage 2: P2P worked, NCCL still refused

Raw copies worked and NCCL still died, which is the part that wasted the most time:

```
ncclCommInitRank: NCCL error: unhandled cuda error
custom all-reduce: CUDA error: mapping of buffer object failed
```

With P2P disabled NCCL worked fine (over the host), so this was not a version mismatch or a missing
symbol. The fallback (`NCCL_P2P_DISABLE=1`) let the server run, and that is how it ran for a while,
at 2,887 tok/s prefill.

## The actual cause

BAR1 size. `nvidia-smi -q` showed the three 3090 Ti cards with a 32GB BAR1 and the plain RTX 3090
(slot 01) with **256MB**. BAR1 is the aperture the driver uses to map peer memory, and 256MB cannot
describe a 24GB peer's buffers, so NCCL's P2P path failed while a direct copy that did not need the
full mapping still succeeded.

The 3090's VBIOS did not support Resizable BAR. Flashing a ReBAR-capable VBIOS took it to 32GB, and:

```
NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels
NCCL INFO Symmetric VA size=24GB
```

## What it bought

| metric | P2P off | P2P on |
|---|---|---|
| prefill 10K | ~2,973 | **6,716** |
| prefill 13K | 3,063 (4.24 s) | **6,456 (2.01 s)** |
| prefill 100K | — | **7,130** |
| prefill 261K | — | **6,782 (38.49 s)** |
| decode (reason / repeat / code) | 176 / 195 / 171 | 192 / 186 / 163 |
| KV pool | 806,792 | 806,792 |

Decode barely moves; P2P pays off in prefill, where the all-reduce volume is. The KV pool is
unchanged, so this is pure speed.

## Cautions

- VBIOS flashing can brick a card. I backed up the original ROM first, and the process did hang the
  whole system once before it succeeded, which needed a recovery cycle. Do not start this without a
  backup and without accepting that the machine may be down for a while.
- `--disable-custom-all-reduce` stays on even with P2P working. The custom all-reduce kernel failed
  with `mapping of buffer object failed` and only the risky kernel needs to be off; NCCL transport is
  what carries the gain.
- `NCCL_DEBUG=INFO` was used for diagnosis and left the recipe pinned config. Remove it once things
  work: it does not cost performance but it is not part of the reference either.
- None of this is specific to this model. Any multi-GPU workload on mixed 3090/3090 Ti hardware can
  hit the same BAR1 asymmetry, and it is worth checking `nvidia-smi -q | grep -i bar1` before
  blaming NCCL.
