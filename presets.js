// Model presets — values pulled from each model's official HF config.json.
// Fields:
//   family             — used to group entries in the dropdown (<optgroup>)
//   hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads
//   head_dim           — optional; defaults to hidden_size / num_heads
//   vocab_size, tie_embeddings (bool)
//
// All entries use SwiGLU MLPs (3 weight matrices: gate + up + down).
// Models with non-gated MLPs (StarCoder2, Falcon, …) need extra schema work
// and are intentionally omitted to keep the estimator honest.

window.MODEL_PRESETS = {

  // ─────────── Llama ───────────
  "TinyLlama-1.1B":  { family: "Llama",  hidden_size: 2048,  intermediate_size: 5632,  num_layers: 22,  num_heads: 32,  num_kv_heads: 4,   vocab_size: 32000,  tie_embeddings: false },
  "Llama-3.2-1B":    { family: "Llama",  hidden_size: 2048,  intermediate_size: 8192,  num_layers: 16,  num_heads: 32,  num_kv_heads: 8,   vocab_size: 128256, tie_embeddings: true  },
  "Llama-3.2-3B":    { family: "Llama",  hidden_size: 3072,  intermediate_size: 8192,  num_layers: 28,  num_heads: 24,  num_kv_heads: 8,   vocab_size: 128256, tie_embeddings: true  },
  "Llama-2-7B":      { family: "Llama",  hidden_size: 4096,  intermediate_size: 11008, num_layers: 32,  num_heads: 32,  num_kv_heads: 32,  vocab_size: 32000,  tie_embeddings: false },
  "Llama-2-13B":     { family: "Llama",  hidden_size: 5120,  intermediate_size: 13824, num_layers: 40,  num_heads: 40,  num_kv_heads: 40,  vocab_size: 32000,  tie_embeddings: false },
  "Llama-3.1-8B":    { family: "Llama",  hidden_size: 4096,  intermediate_size: 14336, num_layers: 32,  num_heads: 32,  num_kv_heads: 8,   vocab_size: 128256, tie_embeddings: false },
  "Llama-3.1-70B":   { family: "Llama",  hidden_size: 8192,  intermediate_size: 28672, num_layers: 80,  num_heads: 64,  num_kv_heads: 8,   vocab_size: 128256, tie_embeddings: false },
  "Llama-3.1-405B":  { family: "Llama",  hidden_size: 16384, intermediate_size: 53248, num_layers: 126, num_heads: 128, num_kv_heads: 8,   vocab_size: 128256, tie_embeddings: false },

  // ─────────── Qwen 2.5 ───────────
  "Qwen2.5-0.5B":    { family: "Qwen",   hidden_size: 896,   intermediate_size: 4864,  num_layers: 24,  num_heads: 14,  num_kv_heads: 2,   vocab_size: 151936, tie_embeddings: true  },
  "Qwen2.5-1.5B":    { family: "Qwen",   hidden_size: 1536,  intermediate_size: 8960,  num_layers: 28,  num_heads: 12,  num_kv_heads: 2,   vocab_size: 151936, tie_embeddings: true  },
  "Qwen2.5-3B":      { family: "Qwen",   hidden_size: 2048,  intermediate_size: 11008, num_layers: 36,  num_heads: 16,  num_kv_heads: 2,   vocab_size: 151936, tie_embeddings: true  },
  "Qwen2.5-7B":      { family: "Qwen",   hidden_size: 3584,  intermediate_size: 18944, num_layers: 28,  num_heads: 28,  num_kv_heads: 4,   vocab_size: 152064, tie_embeddings: false },
  "Qwen2.5-14B":     { family: "Qwen",   hidden_size: 5120,  intermediate_size: 13824, num_layers: 48,  num_heads: 40,  num_kv_heads: 8,   vocab_size: 152064, tie_embeddings: false },
  "Qwen2.5-32B":     { family: "Qwen",   hidden_size: 5120,  intermediate_size: 27648, num_layers: 64,  num_heads: 40,  num_kv_heads: 8,   vocab_size: 152064, tie_embeddings: false },
  "Qwen2.5-72B":     { family: "Qwen",   hidden_size: 8192,  intermediate_size: 29568, num_layers: 80,  num_heads: 64,  num_kv_heads: 8,   vocab_size: 152064, tie_embeddings: false },

  // ─────────── Qwen 3 ───────────
  // Qwen 3 pins head_dim to 128 across the entire family, regardless of hidden/num_heads.
  "Qwen3-0.6B":      { family: "Qwen",   hidden_size: 1024,  intermediate_size: 3072,  num_layers: 28,  num_heads: 16,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: true  },
  "Qwen3-1.7B":      { family: "Qwen",   hidden_size: 2048,  intermediate_size: 6144,  num_layers: 28,  num_heads: 16,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: true  },
  "Qwen3-4B":        { family: "Qwen",   hidden_size: 2560,  intermediate_size: 9728,  num_layers: 36,  num_heads: 32,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: true  },
  "Qwen3-8B":        { family: "Qwen",   hidden_size: 4096,  intermediate_size: 12288, num_layers: 36,  num_heads: 32,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: false },
  "Qwen3-14B":       { family: "Qwen",   hidden_size: 5120,  intermediate_size: 17408, num_layers: 40,  num_heads: 40,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: false },
  "Qwen3-32B":       { family: "Qwen",   hidden_size: 5120,  intermediate_size: 25600, num_layers: 64,  num_heads: 64,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: false },

  // ─────────── Mistral ───────────
  "Mistral-7B":          { family: "Mistral", hidden_size: 4096, intermediate_size: 14336, num_layers: 32, num_heads: 32, num_kv_heads: 8, vocab_size: 32000,  tie_embeddings: false },
  "Mistral-Nemo-12B":    { family: "Mistral", hidden_size: 5120, intermediate_size: 14336, num_layers: 40, num_heads: 32, num_kv_heads: 8, head_dim: 128, vocab_size: 131072, tie_embeddings: false },
  "Codestral-22B":       { family: "Mistral", hidden_size: 6144, intermediate_size: 16384, num_layers: 56, num_heads: 48, num_kv_heads: 8, head_dim: 128, vocab_size: 32768,  tie_embeddings: false },
  "Mistral-Small-3-24B": { family: "Mistral", hidden_size: 5120, intermediate_size: 32768, num_layers: 40, num_heads: 32, num_kv_heads: 8, head_dim: 128, vocab_size: 131072, tie_embeddings: false },

  // ─────────── Gemma 2 ───────────
  "Gemma-2-2B":      { family: "Gemma",  hidden_size: 2304, intermediate_size: 9216,  num_layers: 26, num_heads: 8,  num_kv_heads: 4,  head_dim: 256, vocab_size: 256000, tie_embeddings: true },
  "Gemma-2-9B":      { family: "Gemma",  hidden_size: 3584, intermediate_size: 14336, num_layers: 42, num_heads: 16, num_kv_heads: 8,                vocab_size: 256000, tie_embeddings: true },
  "Gemma-2-27B":     { family: "Gemma",  hidden_size: 4608, intermediate_size: 36864, num_layers: 46, num_heads: 32, num_kv_heads: 16,               vocab_size: 256000, tie_embeddings: true },

  // ─────────── Phi ───────────
  "Phi-3-mini-4k":   { family: "Phi",    hidden_size: 3072, intermediate_size: 8192,  num_layers: 32, num_heads: 32, num_kv_heads: 32, vocab_size: 32064,  tie_embeddings: false },
  "Phi-4-14B":       { family: "Phi",    hidden_size: 5120, intermediate_size: 17920, num_layers: 40, num_heads: 40, num_kv_heads: 10, vocab_size: 100352, tie_embeddings: false },

  // ─────────── 01.AI · Yi ───────────
  "Yi-1.5-9B":       { family: "Yi",     hidden_size: 4096, intermediate_size: 11008, num_layers: 48, num_heads: 32, num_kv_heads: 4,  vocab_size: 64000, tie_embeddings: false },
  "Yi-1.5-34B":      { family: "Yi",     hidden_size: 7168, intermediate_size: 20480, num_layers: 60, num_heads: 56, num_kv_heads: 8,  vocab_size: 64000, tie_embeddings: false },

  // ─────────── AllenAI · OLMo 2 ───────────
  "OLMo-2-7B":       { family: "OLMo",   hidden_size: 4096, intermediate_size: 11008, num_layers: 32, num_heads: 32, num_kv_heads: 32, vocab_size: 100352, tie_embeddings: false },
  "OLMo-2-13B":      { family: "OLMo",   hidden_size: 5120, intermediate_size: 13824, num_layers: 40, num_heads: 40, num_kv_heads: 40, vocab_size: 100352, tie_embeddings: false },

  // ─────────── Other ───────────
  "SmolLM2-1.7B":    { family: "Other",  hidden_size: 2048, intermediate_size: 8192,  num_layers: 24, num_heads: 32, num_kv_heads: 32, vocab_size: 49152,  tie_embeddings: true },
  "Granite-3-8B":    { family: "Other",  hidden_size: 4096, intermediate_size: 12800, num_layers: 40, num_heads: 32, num_kv_heads: 8,  head_dim: 128, vocab_size: 49155,  tie_embeddings: false },
  "GLM-4-9B":        { family: "Other",  hidden_size: 4096, intermediate_size: 13696, num_layers: 40, num_heads: 32, num_kv_heads: 2,  head_dim: 128, vocab_size: 151552, tie_embeddings: false },
};

// Order in which model families appear in the dropdown.
window.MODEL_FAMILY_ORDER = [
  "Llama", "Qwen", "Mistral", "Gemma", "Phi", "Yi", "OLMo", "Other",
];

window.GPU_PRESETS = {
  "H100 80GB":  { name: "H100 80GB",  mem_gb: 80  },
  "H200 141GB": { name: "H200 141GB", mem_gb: 141 },
  "A100 80GB":  { name: "A100 80GB",  mem_gb: 80  },
  "A100 40GB":  { name: "A100 40GB",  mem_gb: 40  },
  "L40S 48GB":  { name: "L40S 48GB",  mem_gb: 48  },
  "RTX 4090":   { name: "RTX 4090",   mem_gb: 24  },
  "RTX 3090":   { name: "RTX 3090",   mem_gb: 24  },
};
