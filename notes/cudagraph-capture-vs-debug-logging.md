# A DEBUG log line killed CUDA graph capture

Symptom: speculative prefill and decode graph capture failed with `cudaErrorStreamCaptureInvalidated`.
Warmup never finished, so MTP was effectively unusable and the engine looked broken.

I spent a while on the wrong suspects before finding it. Writing this down because the actual cause
was a setting I had turned on myself for unrelated reasons, and because the way I found it is
repeatable.

## What it was not

- The connector. I had already replaced it once (the full overlay route) and blamed it, wrongly.
- Non-exact QSA. That was a separate, real problem (see below), but it caused a different failure:
  a spin in the persistent top-k dispatch during warmup, with one rank busy and the rest idle.
- The speculator. Eager MTP ran, just at 5-7 tok/s, which turned out to be a capture *config* problem
  (`--cudagraph_capture_sizes:[1]` makes this fork skip capture entirely) rather than a speculator bug.

## How I found it

`CUDA_LAUNCH_BLOCKING=1` made the failing launch synchronous and named the operation: a
`torch.masked_select` kernel, launched from inside the capture region. Nothing in the model's capture
path calls `masked_select`, which is what made it click: a log statement did.

`VLLM_LOGGING_LEVEL=DEBUG` was set. At DEBUG, rendering an IR op's arguments calls `str(tensor)`,
and for some tensor wrappers that is not a cheap repr: it evaluates a `masked_select` on the GPU. A
kernel launch while a stream is capturing invalidates the capture. Production runs at INFO, so this
only ever happened on the machine where I was debugging.

Removing `VLLM_LOGGING_LEVEL=DEBUG` made speculative prefill and decode graph capture succeed on the
first attempt, in about 14 seconds of warmup.

## Takeaways

- If capture fails with `cudaErrorStreamCaptureInvalidated` and the trace points at an op your model
  never calls, look for a log statement that touches a tensor. Turn verbose logging off before
  bisecting anything else.
- The reverse also bit me: the DEBUG logging was masking the real QSA problem behind a more
  spectacular failure. Fix the loudest problem first, then re-measure before believing your earlier
  diagnosis. Two of the three symptoms I originally attributed to the connector route were this.
- Keep diagnostic flags out of the production entrypoint. In this repo `serving/serve-int4-ple.sh`
  never sets `VLLM_LOGGING_LEVEL`.
