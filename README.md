# FSDP Memory Estimator

Browser-based per-GPU memory estimator for PyTorch **FSDP** training.

**▶ Live demo: https://popsoda2002.github.io/FSDP_memory_estimator/**

Pick a model preset (Llama / Qwen / Mistral / Gemma / Phi) or punch in a custom config,
choose your sharding strategy, precision and activation-checkpointing, and see the per-GPU
breakdown plus a sweep over `world_size`.

## Run locally

```bash
git clone https://github.com/PopSoda2002/FSDP_memory_estimator.git
cd FSDP_memory_estimator && python3 -m http.server 8000
# open http://localhost:8000
```

Pure static site — no build step, no backend.

## What it covers

- Sharding: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `HYBRID_SHARD`, `NO_SHARD` (DDP)
- Precision: bf16 / fp16 / fp32 + fp32 master copy
- Optimizers: AdamW, Adam, SGD(+momentum), 8-bit Adam
- Activation checkpointing: none / selective / full, ± FlashAttention
- GPU presets (H100/H200/A100/L40S/4090/3090) with a fit indicator

## Caveat

It's an estimator, not a profiler — real peak depends on FSDP version, allocator
fragmentation, custom kernels, NCCL/CUDA buffers. Treat numbers as a budgeting tool
(typically within ±15% of measured peak).

Inspired by [ISEEKYAN/megatron_memory_estimator](https://huggingface.co/spaces/ISEEKYAN/megatron_memory_estimator).
