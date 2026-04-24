# FSDP Memory Estimator

A visual, browser-based memory estimator for **PyTorch FSDP** training. Drop in your model
config (or pick a preset), choose your training/parallelism settings, and instantly see a
per-GPU memory breakdown — parameters, gradients, optimizer states, activations, comms
buffers — plus how that memory scales as you add GPUs.

Inspired by [ISEEKYAN/megatron_memory_estimator](https://huggingface.co/spaces/ISEEKYAN/megatron_memory_estimator),
but specialised for FSDP / FSDP2 / HSDP training (no TP/PP/CP needed).

## Features

- **Per-GPU memory breakdown**: parameters, gradients, optimizer master + states, activations, all-gather/reduce-scatter buffers, framework overhead
- **Sharding strategies**: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), `HYBRID_SHARD`, plain `DDP`
- **Mixed precision**: bf16 / fp16 / fp32 compute + fp32 master copy
- **Activation checkpointing**: none / selective / full
- **Optimizers**: AdamW (8 bytes/param state), Adam, SGD-momentum, SGD, 8-bit Adam
- **Sweep view**: how peak memory scales with `world_size`
- **Presets**: Llama-3, Qwen-3, Mistral, DeepSeek, plus a "Custom" mode
- **No backend**: 100% client-side. Open `index.html` directly, host on GitHub Pages, or HF Spaces (static).

## Quick start

```bash
git clone https://github.com/PopSoda2002/FSDP_memory_estimator.git
cd FSDP_memory_estimator
# Just open index.html in a browser — no build step.
python3 -m http.server 8000
# then visit http://localhost:8000
```

## What's modeled

Per GPU, with sharding factor `S` (= `world_size` for `FULL_SHARD`, 1 for `DDP`):

| Component | Size |
|---|---|
| Sharded params (low-precision) | `2 · P / S` |
| Sharded grads (low-precision)  | `2 · P / S` |
| Master params (fp32)           | `4 · P / S` |
| Adam m (fp32)                  | `4 · P / S` |
| Adam v (fp32)                  | `4 · P / S` |
| All-gather buffer (peak layer) | `2 · P_layer` |
| Activations                    | `≈ s·b·h · (10 + 24 + 5·a·s/h) · L_eff` |

`P_layer` is the largest unit being all-gathered (one transformer block by default).
`L_eff` depends on activation checkpointing.

See `script.js` for the exact formulas.

## Layout

```
.
├── index.html       UI
├── style.css        styling
├── script.js        memory math + Chart.js rendering
├── presets.js       model presets (Llama, Qwen, etc.)
└── README.md
```

## Caveats

This is an *estimator*, not a profiler. Real memory varies with:
- PyTorch / FSDP version (FSDP2 vs FSDP1, sharded-state-dict, etc.)
- CUDA caching allocator fragmentation
- Custom kernels, fused ops, FlashAttention version
- CPU-offload settings, NCCL buffers

Treat the numbers as a budgeting tool: usually within ±15% of measured peak.
