# vindex-infer

**Vendor-free LLM inference from decomposed weights. CPU or Vulkan GPU. No CUDA required.**

**Paper**: [Framework-Free Transformer Inference from Decomposed Weight Files via Vulkan Compute](https://zenodo.org/records/19609604) (Zenodo)

Run any open-source LLM on any hardware — NVIDIA, AMD, Intel, Apple, or just CPU. No framework dependencies, no vendor lock-in. One Rust binary, flat weight files, exact output.

```bash
# Download pre-extracted model from HuggingFace (7.3 GB)
pip install huggingface-hub
huggingface-cli download cronos3k/gemma-3-4b-it-vindex --local-dir gemma3-4b.vindex

# Build
cargo build --release

# Run inference — CPU (works everywhere, no GPU needed)
./target/release/vindex-infer --vindex gemma3-4b.vindex --token-ids "818,5279,529,7001,563"
#  1. Paris     (+21.24)
#  2. a         (+17.69)
#  3. the       (+17.51)
```

## Pre-Extracted Model

A pre-extracted Gemma 3 4B vindex is available on HuggingFace for immediate testing — no LarQL installation, no model extraction, no gated access approval needed:

**[cronos3k/gemma-3-4b-it-vindex](https://huggingface.co/cronos3k/gemma-3-4b-it-vindex)** (7.29 GB, f16, all extraction levels)

Or extract your own from any supported model via [LarQL](https://github.com/chrishayuk/larql):
```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level all --f16
```

## Why This Exists

Every LLM inference solution locks you into a vendor:

| Tool | Requires |
|------|----------|
| PyTorch | Python + CUDA (NVIDIA only) |
| llama.cpp | GGUF format, CUDA for GPU |
| MLX | Apple Silicon only |
| TensorRT | NVIDIA only |
| ONNX Runtime | Microsoft ecosystem |

**vindex-infer requires nothing.** One Rust binary. Flat binary weight files. CPU runs everywhere. GPU runs on Vulkan — which means every discrete GPU, every integrated GPU, every mobile GPU made in the last decade.

## Features

- **Two backends**: CPU (rayon multi-core) and Vulkan GPU (any vendor)
- **Exact output**: Matches HuggingFace Transformers to 5 significant figures
- **Zero framework dependencies**: No Python, no CUDA toolkit, no ML libraries
- **Flat file input**: Vindex format — mmap'd binary files, zero-copy loading
- **Small binary**: ~20 MB compiled, no runtime bloat
- **Verified**: Paris, Jupiter, blue, Ulm, Pound — factual knowledge retrieval confirmed

## Verified Results

Gemma 3 4B inference (34 layers, 2560 hidden, 262K vocab):

| Prompt | Prediction | Correct? |
|--------|-----------|----------|
| The capital of France is | **Paris** | [YES] |
| The largest planet is | **Jupiter** | [YES] |
| The color of the sky is | **blue** | [YES] |
| Einstein was born in | **Ulm** | [YES] |
| The currency of the UK is | **Pound** (#2) | [YES] |

All predictions match HuggingFace ground truth exactly.

## Performance

| Backend | Time (Gemma 4B) | Hardware |
|---------|----------------|----------|
| CPU (rayon) | ~11s | 64-core AMD EPYC |
| Vulkan GPU | ~4.6s | NVIDIA V100 |

Performance is currently limited by per-operation dispatch overhead (1400+ individual GPU submissions). Batch dispatch optimization is planned.

## Building

```bash
# CPU only (no Vulkan needed)
cargo build --release --no-default-features

# CPU + Vulkan GPU
cargo build --release
```

Requires Rust 1.75+. Vulkan GPU backend requires a Vulkan driver (no SDK needed at runtime).

## Test Prompts (Gemma 3 4B Token IDs)

Since vindex-infer uses token IDs directly (no tokenizer built in yet), here are some pre-tokenized prompts for testing:

```bash
# "The capital of France is" → Paris
--token-ids "818,5279,529,7001,563"

# "The largest planet is" → Jupiter
--token-ids "818,7488,13401,563"

# "The color of the sky is" → blue
--token-ids "818,2258,529,506,7217,563"

# "Einstein was born in" → Ulm
--token-ids "147505,691,8132,528"

# "The currency of the UK is" → Pound (#2)
--token-ids "818,15130,529,506,6322,563"
```

BOS token (2) is prepended automatically.

## How It Works

1. **Vindex loading**: Weight matrices (gate, up, down, attention, embeddings, norms) are memory-mapped from flat binary files. No deserialization — just pointer arithmetic on mmap'd data.

2. **Attention**: Multi-head grouped query attention (GQA) with QK normalization, RoPE positional encoding (HF half-split style), and causal masking.

3. **FFN**: Dense gated projection with GeluTanh activation. Gate and up projections produce activation vectors; down projection maps back to hidden space.

4. **Norms**: RMSNorm with learned gamma weights (offset=1.0 for Gemma 3).

5. **GPU path**: A single Vulkan compute shader (`matvec.comp`) handles all matrix-vector products. Weights are uploaded to VRAM as f16, decoded on-the-fly in the shader. Input/output are f32.

## Model Support

Any model that LarQL can extract to a vindex:

| Family | Models |
|--------|--------|
| Gemma | Gemma 2/3 (2B-27B) |
| Llama | Llama 2/3 (7B-405B) |
| Mistral | Mistral 7B |
| Qwen | Qwen 2/2.5 (0.5B-72B) |
| Mixtral | Mixtral 8x7B, 8x22B |
| DeepSeek | DeepSeek V2/V3 |
| GPT-2 | GPT-2 (117M-1.5B) |

Extract via [LarQL](https://github.com/chrishayuk/larql):
```bash
larql extract-index <model> -o model.vindex --level all --f16
```

## Architecture

```
model.vindex/           (flat binary files, mmap'd)
├── gate_vectors.bin    W_gate [layers × features × hidden] f16
├── up_weights.bin      W_up   [layers × features × hidden] f16
├── down_weights.bin    W_down [layers × hidden × features] f16
├── attn_weights.bin    Q/K/V/O + QK norms per layer, f16
├── norms.bin           4 RMSNorm gammas per layer + final, f16
├── embeddings.bin      [vocab × hidden] f16
└── index.json          config, dimensions, layer info
         │
         ▼
   ┌─────────────┐
   │ vindex-infer │──→ CPU (rayon, any platform)
   │   (Rust)     │──→ Vulkan GPU (any vendor)
   └─────────────┘
         │
         ▼
   Paris (+21.24)
```

## Credits

- **[LarQL](https://github.com/chrishayuk/larql)** by Chris Hayuk — the model decomposition engine that creates vindexes. All vindex format design, extraction pipeline, and LQL query language are his work.
- **[LarQL CUDA fork](https://github.com/cronos3k/larql)** — Linux/Windows + CUDA backend port.

## License

Apache-2.0
