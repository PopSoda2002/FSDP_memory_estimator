// Model presets. All values come from the official HF config.json of each model.
// Fields:
//   hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads,
//   head_dim (optional, defaults to hidden_size / num_heads),
//   vocab_size, tie_embeddings (bool), is_moe (bool), num_experts, num_experts_per_token

window.MODEL_PRESETS = {
  "Llama-3.1-8B": {
    hidden_size: 4096, intermediate_size: 14336, num_layers: 32,
    num_heads: 32, num_kv_heads: 8, vocab_size: 128256, tie_embeddings: false,
  },
  "Llama-3.1-70B": {
    hidden_size: 8192, intermediate_size: 28672, num_layers: 80,
    num_heads: 64, num_kv_heads: 8, vocab_size: 128256, tie_embeddings: false,
  },
  "Llama-3.2-1B": {
    hidden_size: 2048, intermediate_size: 8192, num_layers: 16,
    num_heads: 32, num_kv_heads: 8, vocab_size: 128256, tie_embeddings: true,
  },
  "Llama-3.2-3B": {
    hidden_size: 3072, intermediate_size: 8192, num_layers: 28,
    num_heads: 24, num_kv_heads: 8, vocab_size: 128256, tie_embeddings: true,
  },
  "Qwen2.5-7B": {
    hidden_size: 3584, intermediate_size: 18944, num_layers: 28,
    num_heads: 28, num_kv_heads: 4, vocab_size: 152064, tie_embeddings: false,
  },
  "Qwen2.5-14B": {
    hidden_size: 5120, intermediate_size: 13824, num_layers: 48,
    num_heads: 40, num_kv_heads: 8, vocab_size: 152064, tie_embeddings: false,
  },
  "Qwen2.5-32B": {
    hidden_size: 5120, intermediate_size: 27648, num_layers: 64,
    num_heads: 40, num_kv_heads: 8, vocab_size: 152064, tie_embeddings: false,
  },
  "Qwen2.5-72B": {
    hidden_size: 8192, intermediate_size: 29568, num_layers: 80,
    num_heads: 64, num_kv_heads: 8, vocab_size: 152064, tie_embeddings: false,
  },
  "Qwen3-8B": {
    hidden_size: 4096, intermediate_size: 12288, num_layers: 36,
    num_heads: 32, num_kv_heads: 8, vocab_size: 151936, tie_embeddings: false,
  },
  "Qwen3-14B": {
    hidden_size: 5120, intermediate_size: 17408, num_layers: 40,
    num_heads: 40, num_kv_heads: 8, vocab_size: 151936, tie_embeddings: false,
  },
  "Qwen3-32B": {
    // Qwen3-32B uses an explicit head_dim of 128 instead of hidden/num_heads (= 80).
    hidden_size: 5120, intermediate_size: 25600, num_layers: 64,
    num_heads: 64, num_kv_heads: 8, head_dim: 128,
    vocab_size: 151936, tie_embeddings: false,
  },
  "Mistral-7B": {
    hidden_size: 4096, intermediate_size: 14336, num_layers: 32,
    num_heads: 32, num_kv_heads: 8, vocab_size: 32000, tie_embeddings: false,
  },
  "Gemma-2-9B": {
    hidden_size: 3584, intermediate_size: 14336, num_layers: 42,
    num_heads: 16, num_kv_heads: 8, vocab_size: 256000, tie_embeddings: true,
  },
  "Gemma-2-27B": {
    hidden_size: 4608, intermediate_size: 36864, num_layers: 46,
    num_heads: 32, num_kv_heads: 16, vocab_size: 256000, tie_embeddings: true,
  },
  "Phi-3-mini-4k": {
    hidden_size: 3072, intermediate_size: 8192, num_layers: 32,
    num_heads: 32, num_kv_heads: 32, vocab_size: 32064, tie_embeddings: false,
  },
};

window.GPU_PRESETS = {
  "H100 80GB":   { name: "H100 80GB",   mem_gb: 80 },
  "H200 141GB":  { name: "H200 141GB",  mem_gb: 141 },
  "A100 80GB":   { name: "A100 80GB",   mem_gb: 80 },
  "A100 40GB":   { name: "A100 40GB",   mem_gb: 40 },
  "L40S 48GB":   { name: "L40S 48GB",   mem_gb: 48 },
  "RTX 4090":    { name: "RTX 4090",    mem_gb: 24 },
  "RTX 3090":    { name: "RTX 3090",    mem_gb: 24 },
};
