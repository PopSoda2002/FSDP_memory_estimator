// Model presets — values pulled from each model's official HF config.json.
// Fields:
//   family             — used to group entries in the dropdown (<optgroup>)
//   hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads
//   head_dim           — optional; defaults to hidden_size / num_heads
//   vocab_size, tie_embeddings (bool)
//
// All entries use SwiGLU MLPs (3 weight matrices: gate + up + down).

window.MODEL_PRESETS = {

  // ─────────── Qwen 3 ───────────
  // Qwen 3 pins head_dim to 128 across the entire family, regardless of hidden/num_heads.
  "Qwen3-0.6B":      { family: "Qwen3", hidden_size: 1024,  intermediate_size: 3072,  num_layers: 28,  num_heads: 16,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: true  },
  "Qwen3-1.7B":      { family: "Qwen3", hidden_size: 2048,  intermediate_size: 6144,  num_layers: 28,  num_heads: 16,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: true  },
  "Qwen3-4B":        { family: "Qwen3", hidden_size: 2560,  intermediate_size: 9728,  num_layers: 36,  num_heads: 32,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: true  },
  "Qwen3-8B":        { family: "Qwen3", hidden_size: 4096,  intermediate_size: 12288, num_layers: 36,  num_heads: 32,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: false },
  "Qwen3-14B":       { family: "Qwen3", hidden_size: 5120,  intermediate_size: 17408, num_layers: 40,  num_heads: 40,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: false },
  "Qwen3-32B":       { family: "Qwen3", hidden_size: 5120,  intermediate_size: 25600, num_layers: 64,  num_heads: 64,  num_kv_heads: 8, head_dim: 128, vocab_size: 151936, tie_embeddings: false },
};

window.MODEL_FAMILY_ORDER = ["Qwen3"];

window.GPU_PRESETS = {
  "H100 80GB":  { name: "H100 80GB",  mem_gb: 80  },
  "H200 141GB": { name: "H200 141GB", mem_gb: 141 },
  "A100 80GB":  { name: "A100 80GB",  mem_gb: 80  },
  "A100 40GB":  { name: "A100 40GB",  mem_gb: 40  },
  "L40S 48GB":  { name: "L40S 48GB",  mem_gb: 48  },
  "RTX 4090":   { name: "RTX 4090",   mem_gb: 24  },
  "RTX 3090":   { name: "RTX 3090",   mem_gb: 24  },
};
