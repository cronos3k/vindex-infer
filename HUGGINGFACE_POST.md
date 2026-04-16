# HuggingFace Community Post

---

**Title: vindex-infer — Run LLMs without CUDA, without PyTorch, from flat binary files**

I built a standalone LLM inference engine in Rust that loads decomposed transformer weights from LarQL's vindex format and produces exact HuggingFace output — without any ML framework.

## What it does

Takes a vindex (flat binary files containing a model's weights organized by function) and runs full transformer inference: multi-head GQA attention with RoPE, dense FFN with GeluTanh, RMSNorm — the complete Gemma 3 4B architecture, 34 layers.

Output matches `transformers` library to 5 significant figures:

| Prompt | #1 Prediction | Score | Matches HF? |
|--------|--------------|-------|-------------|
| The capital of France is | Paris | +21.24 | [YES] Exact |
| The largest planet is | Jupiter | +20.71 | [YES] Exact |
| The color of the sky is | blue | +23.23 | [YES] Exact |
| Einstein was born in | Ulm | +30.26 | [YES] Exact |

## How to try it

A pre-extracted Gemma 3 4B vindex is available here: **[cronos3k/gemma-3-4b-it-vindex](https://huggingface.co/cronos3k/gemma-3-4b-it-vindex)** (7.29 GB)

```bash
# Download the vindex
huggingface-cli download cronos3k/gemma-3-4b-it-vindex --local-dir gemma3-4b.vindex

# Build the inference engine (Rust required)
cargo install --git https://github.com/cronos3k/vindex-infer

# Run — CPU only, no GPU needed
vindex-infer --vindex gemma3-4b.vindex --token-ids "818,5279,529,7001,563"
# → Paris (#1, +21.24)
```

## Why it exists

Every LLM inference path today requires a vendor-specific stack. PyTorch needs CUDA. MLX needs Apple Silicon. TensorRT needs NVIDIA. Even llama.cpp needs GGUF conversion per quantization level.

The vindex format stores weights as flat binary files organized by function (attention, FFN, norms, embeddings). They can be memory-mapped and processed by any program on any platform. The inference engine is ~600 lines of Rust with zero ML dependencies.

The Vulkan GPU backend runs on any GPU vendor (NVIDIA, AMD, Intel, Qualcomm) — not just CUDA-capable hardware.

## Performance

| Backend | Time (Gemma 3 4B, 6 tokens) |
|---------|---------------------------|
| CPU (64-core AMD EPYC) | ~6 seconds |
| V100 GPU (Vulkan) | ~4.6 seconds |

Performance is currently unoptimized (per-operation GPU dispatch, no batching, no quantization). The focus was correctness first — every output digit matches HuggingFace.

## Technical details

The implementation handles all Gemma 3 specifics:
- GQA attention (8 Q heads, 4 KV heads, head_dim=256)
- QK normalization with +1 offset
- RoPE with half-split pairing (not adjacent pairs)
- Per-layer RoPE base (10K for sliding window, 1M/8 for global attention)
- GeluTanh activation
- 4 RMSNorm layers per transformer layer (all with +1 offset)
- down_proj stored as [hidden, intermediate] (transposed relative to gate/up)

Each of these details was verified by step-by-step comparison against the HuggingFace implementation.

## Credits

- **LarQL** by Chris Hayuk ([@chrishayuk](https://github.com/chrishayuk)) — the model decomposition engine, vindex format, and LQL query language. The core innovation that makes this possible.
- **Gemma 3** by Google — the model whose weights we're running.

## Links

- Inference engine: [github.com/cronos3k/vindex-infer](https://github.com/cronos3k/vindex-infer)
- Pre-extracted model: [cronos3k/gemma-3-4b-it-vindex](https://huggingface.co/cronos3k/gemma-3-4b-it-vindex)
- LarQL: [github.com/chrishayuk/larql](https://github.com/chrishayuk/larql)
- LarQL CUDA fork: [github.com/cronos3k/larql](https://github.com/cronos3k/larql)

---
