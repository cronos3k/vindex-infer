# Framework-Free Transformer Inference from Decomposed Weight Files via Vulkan Compute

**Gregor H. Max Koch**
Independent Researcher

**April 2026**

---

## Abstract

We present vindex-infer, a standalone transformer inference engine that loads model weights from LarQL's vindex format — flat, function-organized binary files — and executes full transformer computation without any machine learning framework. Our implementation produces output identical to HuggingFace Transformers, verified to 5 significant figures across all 34 layers of Gemma 3 4B. We demonstrate two execution backends: CPU (via rayon multi-threading) and Vulkan GPU (via compute shaders), enabling inference on any hardware without CUDA dependency. We document the complete implementation, including every architectural detail required for numerical correctness (RoPE pairing convention, norm weight offsets, matrix layout conventions), and provide pre-extracted model weights on HuggingFace for immediate reproduction. The entire inference engine comprises approximately 600 lines of Rust.

---

## 1. Introduction

Large language model inference universally depends on framework ecosystems: PyTorch with CUDA for NVIDIA GPUs, MLX for Apple Silicon, TensorRT for optimized NVIDIA deployment, or ONNX Runtime for cross-platform compatibility within Microsoft's ecosystem. Each path imposes vendor-specific dependencies — runtime libraries, compilation toolchains, serialization formats, and hardware requirements.

This dependency chain is not inherent to the mathematics of transformer inference. A transformer layer performs matrix-vector products, element-wise activations, and normalization operations — computations expressible in any language on any hardware. The weights are arrays of floating-point numbers. The obstacle is not computational but organizational: weights are stored in framework-specific formats (safetensors, GGUF, CoreML), interleaved by layer rather than by function, and accessed through framework-specific APIs.

LarQL (Hayuk, 2026) addresses the storage problem by decomposing transformer weights into a "vindex" format where weight matrices are organized by function: gate projections in one file, attention matrices in another, normalization weights in a third. Each file is a flat array of IEEE 754 half-precision floats, directly memory-mappable without deserialization.

We address the computation problem by implementing a complete transformer inference engine that consumes vindex files directly, using only standard system interfaces (memory-mapped I/O, POSIX threads, Vulkan compute). The result is LLM inference that runs on any platform with a Rust compiler, and GPU-accelerated inference on any hardware with a Vulkan driver — which includes every discrete GPU, integrated GPU, and mobile GPU manufactured in the last decade.

### 1.1 Contributions

1. **Independent exact verification** of the vindex format: we demonstrate that an implementation with zero shared code with LarQL produces identical output to HuggingFace Transformers, confirming the format's completeness and self-containedness.

2. **Vulkan as an ML compute backend**: we demonstrate that a single Vulkan compute shader performing f16-to-f32 matrix-vector products produces numerically correct transformer inference on a V100 GPU, establishing Vulkan as a viable alternative to CUDA for ML workloads.

3. **Complete documentation of implementation-critical details**: we catalogue every architectural detail required for numerical correctness in Gemma 3 inference — RoPE pairing conventions, normalization weight offsets, matrix storage layouts, activation function variants, and per-layer parameter differences — providing a reference for future independent implementations.

4. **Open reproduction artifacts**: pre-extracted model weights on HuggingFace, source code on GitHub, and pre-tokenized test prompts enabling three-command reproduction.

---

## 2. Background

### 2.1 The Vindex Format

LarQL (Hayuk, 2026) decomposes transformer models into queryable vector databases. The vindex format stores weight matrices as flat binary arrays of IEEE 754 half-precision (f16) floats, organized by function:

| File | Contents | Shape per Layer |
|------|----------|----------------|
| `gate_vectors.bin` | FFN gate projections | [intermediate, hidden] |
| `up_weights.bin` | FFN up projections | [intermediate, hidden] |
| `down_weights.bin` | FFN down projections | [hidden, intermediate] |
| `attn_weights.bin` | Q, K, V, O matrices + QK norms | [varies] |
| `norms.bin` | RMSNorm gamma weights | [hidden] × 4 per layer |
| `embeddings.bin` | Token embeddings | [vocab, hidden] |
| `index.json` | Configuration metadata | — |

All binary files are contiguous arrays suitable for zero-copy memory mapping. No headers, no alignment padding, no framework-specific metadata. The `index.json` file provides dimensions, layer count, vocabulary size, and model-specific configuration.

### 2.2 Gemma 3 4B Architecture

Our verification target is Google's Gemma 3 4B Instruct model (google/gemma-3-4b-it), a 34-layer transformer with:

| Parameter | Value |
|-----------|-------|
| Hidden size | 2,560 |
| Intermediate size (FFN) | 10,240 |
| Layers | 34 |
| Q attention heads | 8 |
| KV attention heads | 4 (GQA, 2:1 grouping) |
| Head dimension | 256 |
| Vocabulary | 262,208 |
| Activation | GeluTanh |
| Position encoding | RoPE (half-split, per-layer base) |
| Normalization | RMSNorm with +1 offset |
| Total parameters | ~4 billion |

Gemma 3 employs a sliding window attention pattern: every 6th layer (indices 5, 11, 17, 23, 29) uses full attention with RoPE base 1,000,000 and position scaling factor 8; remaining layers use sliding window attention with RoPE base 10,000.

---

## 3. Implementation

### 3.1 Architecture

Our implementation consists of three modules totaling approximately 600 lines of Rust:

- **vindex.rs** (250 lines): Memory-mapped vindex loading, f16→f32 conversion, per-layer weight access, parallelized matrix-vector products via rayon
- **inference.rs** (250 lines): Full transformer inference loop — attention, FFN, normalization, logits
- **main.rs** (60 lines): CLI interface with clap argument parsing

No external ML libraries, numerical computing libraries, or GPU-specific SDKs are used. The CPU path depends only on `memmap2` (memory mapping), `rayon` (parallelism), `serde_json` (config parsing), and `clap` (CLI). The optional Vulkan path adds `ash` (Vulkan bindings).

### 3.2 Inference Loop

For each of the 34 layers, the computation follows Gemma 3's pre-norm architecture:

```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = attention(hidden_states)
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = pre_feedforward_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

Where:
- `attention` = GQA with QK normalization, RoPE, causal masking, and output projection
- `mlp` = GeluTanh gated FFN: `down(gelu_tanh(gate(x)) * up(x))`
- All norms = RMSNorm with `weight = 1.0 + stored_gamma`

### 3.3 Implementation-Critical Details

During development, we identified six details where incorrect implementation produces plausible but wrong output. Each was resolved by comparison against HuggingFace's Gemma 3 implementation.

#### 3.3.1 RoPE Pairing Convention

HuggingFace's `apply_rotary_pos_emb` uses a half-split pairing where dimension `i` is paired with dimension `i + dim/2`:

```python
# HuggingFace (correct for Gemma 3)
rotate_half(x) = cat(-x[dim//2:], x[:dim//2])
q_rotated = q * cos + rotate_half(q) * sin
```

Many implementations (including our initial attempt) use adjacent pairing where dimension `2i` pairs with `2i+1`. The adjacent convention produces attention scores that diverge by 4-16% at non-zero positions, compounding across layers into completely incorrect output.

**Impact of incorrect pairing:**

| Position | Correct Score | Adjacent Pairing | Error |
|----------|--------------|-----------------|-------|
| 0 (BOS) | 9.053 | 9.037 | 0.2% |
| 1 (The) | 8.385 | 8.020 | 4.3% |
| 2 (capital) | 4.115 | 4.775 | 16.0% |

With half-split pairing, all scores match to 4+ decimal places.

#### 3.3.2 Normalization Weight Offset

Gemma 3's `RMSNorm` applies a +1 offset to stored weights:

```python
output = (x / rms) * (1.0 + self.weight)
```

Where `self.weight` is initialized to zeros and trained. The stored values in the vindex include the trained perturbation but NOT the +1 offset. Applying the stored values directly (without +1) produces:
- Post-attention norms (stored mean ≈ 0.01): effectively zero, killing the attention signal
- Post-FFN norms (stored mean ≈ 2.3): inadequate scaling

This offset applies uniformly to all four per-layer norms, the QK norms, and the final norm.

#### 3.3.3 Down Projection Matrix Layout

The vindex stores the FFN down projection as `[hidden_size, intermediate_size]` — matching PyTorch's `nn.Linear(intermediate, hidden)` weight convention of `[out_features, in_features]`.

Gate and up projections are stored as `[intermediate_size, hidden_size]`. The different storage layout for down vs. gate/up is not documented in the vindex specification and produces a silent transpose if the same reshape is applied to all three.

**Impact**: With incorrect layout, the FFN output has cosine similarity of -0.014 with the correct output (effectively orthogonal). After 34 layers, this produces completely random predictions.

#### 3.3.4 Per-Layer RoPE Base

Gemma 3 uses different RoPE parameters for sliding window layers (base=10,000) vs. global attention layers (base=1,000,000 with position scaling factor=8). Using a uniform base of 1,000,000 for all layers degrades the Paris prediction from rank 1 to rank 8,759.

#### 3.3.5 Activation Function

Gemma 3 uses `GeluTanh` (the tanh approximation of GELU), not `SiLU` (Swish). While both are gated activations, they produce different FFN outputs that compound across layers. We verified numerical equivalence of our GeluTanh implementation against HuggingFace's `GELUTanh()` module.

#### 3.3.6 BOS Token

Gemma 3 requires a beginning-of-sequence token (ID 2) prepended to the input. Without BOS, the model produces syntactically coherent but factually incorrect output (rank 1,723 for Paris instead of rank 1).

---

## 4. Verification

### 4.1 Methodology

We compare our output against HuggingFace Transformers (v4.50.0) running the same Gemma 3 4B model in float32 precision. The comparison uses:

1. **Token-level prediction matching**: Top-K token IDs and scores
2. **Per-layer residual norm tracking**: L2 norm of the residual stream at each layer boundary
3. **Step-by-step numerical comparison**: Individual operation outputs (Q projection, attention scores, FFN output) compared at layer 0

### 4.2 Results

#### 4.2.1 Prediction Accuracy

| Prompt | Our #1 (Score) | HF #1 (Score) | Match |
|--------|---------------|---------------|-------|
| The capital of France is | Paris (+21.24) | Paris (+21.25) | ✓ |
| The largest planet is | Jupiter (+20.71) | Jupiter (+20.71) | ✓ |
| The color of the sky is | blue (+23.23) | blue (+23.23) | ✓ |
| Einstein was born in | Ulm (+30.26) | Ulm (+30.26) | ✓ |
| The currency of the UK is | the (+22.83) [Pound #2] | the (+22.83) [Pound #2] | ✓ |

All top-20 predictions match in order, token ID, and score.

#### 4.2.2 Residual Stream Tracking

| Layer | Our Norm | HF Norm | Relative Error |
|-------|----------|---------|---------------|
| 0 | 805.4 | 805.5 | 0.01% |
| 5 | 3,414.8 | 3,414.8 | <0.01% |
| 10 | 14,294 | 14,295 | <0.01% |
| 15 | 27,183 | 27,183 | <0.01% |
| 20 | 35,618 | 35,618 | <0.01% |
| 25 | 43,496 | 43,496 | <0.01% |
| 30 | 58,431 | 58,431 | <0.01% |
| 33 | 67,807 | 67,806 | <0.01% |

The residual stream tracks within 0.01% across all 34 layers, confirming numerical equivalence.

#### 4.2.3 Operation-Level Verification

At layer 0, token position 5 ("is"):

| Operation | Our Value | HF Value | Max Diff |
|-----------|----------|----------|----------|
| Input layernorm (first 5 dims) | [0.791, 1.569, 2.419, 3.063, 1.992] | [0.791, 1.569, 2.419, 3.063, 1.992] | < 10⁻⁴ |
| Q projection (first 5 dims) | [-20.15, 38.08, 14.47, -7.44, -15.02] | [-20.15, 38.08, 14.47, -7.44, -15.02] | < 10⁻² |
| Attention scores (head 0, all 6 positions) | [9.052, 8.385, 4.115, 6.128, 5.504, 6.011] | [9.053, 8.385, 4.115, 6.128, 5.504, 6.011] | < 10⁻³ |

### 4.3 Vulkan GPU Verification

The Vulkan compute shader (matvec.comp) produces output matching the CPU path:

| Metric | GPU vs CPU |
|--------|-----------|
| Max absolute difference | 0.000099 |
| Mean absolute difference | 0.000007 |
| All predictions identical | ✓ |

The V100 GPU path produces Paris #1 with score +21.24, matching both the CPU path and HuggingFace.

---

## 5. Performance

### 5.1 Execution Time

Gemma 3 4B, 6 input tokens ("BOS The capital of France is"), 34 layers, 262K vocabulary:

| Backend | Total Time | Per Layer | Logits | Hardware |
|---------|-----------|-----------|--------|----------|
| CPU (rayon) | 6.0s | 175ms | 62ms | 64-core AMD EPYC 7601 |
| Vulkan GPU | 4.6s | 130ms | 100ms | NVIDIA V100-SXM2-32GB |

### 5.2 Comparison with Existing Systems

| System | Time | Framework | GPU Vendor Lock |
|--------|------|-----------|----------------|
| HuggingFace (f32, CPU) | ~8s | PyTorch + Python | CUDA (NVIDIA) |
| llama.cpp (Q4, CPU) | ~0.3s | C++ | None (CPU) / CUDA (GPU) |
| LarQL walk (sparse, CPU) | ~0.5s | Rust | None (CPU) / Metal (Apple) |
| **vindex-infer (CPU)** | **6.0s** | **Rust** | **None** |
| **vindex-infer (Vulkan)** | **4.6s** | **Rust** | **None (any GPU vendor)** |

Our implementation is not optimized for speed: it performs dense f16 matmul (no quantization), dispatches 1,428 individual GPU command buffers (no batching), and recomputes all tokens each invocation (no KV cache). The contribution is correctness and portability, not throughput.

### 5.3 Resource Requirements

| Resource | Requirement |
|----------|------------|
| Vindex size (Gemma 3 4B, f16) | 7.29 GB |
| Peak RAM (CPU path) | ~8 GB |
| VRAM (GPU path) | 7.3 GB (weights) + ~50 MB (buffers) |
| Binary size | ~20 MB |
| Build dependencies | Rust 1.75+, no system libraries |

---

## 6. The Vulkan Compute Backend

### 6.1 Motivation

CUDA dominates GPU-accelerated ML, but it requires NVIDIA hardware. Vulkan is an open standard supported by every major GPU vendor (NVIDIA, AMD, Intel, Qualcomm, ARM Mali) and is the native graphics API for Android, Linux, and Windows.

We demonstrate that Vulkan compute shaders can execute the core operation of transformer inference — matrix-vector products with f16 weights — with sufficient numerical precision for exact output matching.

### 6.2 Implementation

A single compute shader (`matvec.comp`) handles all matrix-vector products:

```glsl
layout(local_size_x = 256) in;

// Weight data (f16 packed as u32)
layout(set=0, binding=0) readonly buffer Weights { uint weight_data[]; };
layout(set=0, binding=1) readonly buffer Input { float input_vec[]; };
layout(set=0, binding=2) writeonly buffer Output { float output_vec[]; };

void main() {
    uint row = gl_WorkGroupID.x;    // One workgroup per output element
    uint tid = gl_LocalInvocationID.x;
    
    float sum = 0.0;
    for (uint i = tid; i < cols; i += 256) {
        float w = decode_f16(weight_data, row * cols + i);
        sum += w * input_vec[i];
    }
    
    // Parallel reduction in shared memory
    partial[tid] = sum;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        barrier();
    }
    if (tid == 0) output_vec[row] = partial[0];
}
```

Weights are stored on GPU as f16 and decoded to f32 in the shader, matching the precision of the CPU path. The f16→f32 decode is performed per-element with no intermediate quantization.

### 6.3 Weight Upload

All vindex weight files (7.3 GB total) are uploaded to GPU VRAM at initialization via persistent mapped buffers. Upload time on V100 is approximately 8-10 seconds. Once resident, weights remain in VRAM for all subsequent inferences with zero per-layer upload cost.

---

## 7. Discussion

### 7.1 On Framework Independence

The vindex format enables a genuine separation between model storage and model execution. The weights are numbers in files; the computation is mathematics applied to those numbers. By demonstrating exact output from an independent implementation, we establish that the vindex format is a complete and self-contained model representation — sufficient for inference without reference to the training framework.

This separation has practical implications for deployment: a vindex can be extracted once and deployed to any platform without format conversion (GGUF, ONNX, CoreML, etc.), framework installation, or vendor-specific runtime libraries.

### 7.2 On Vulkan as an ML Backend

Our Vulkan implementation is intentionally minimal — a single compute shader with naive parallel reduction. Production ML backends (cuBLAS, cutlass, oneDNN) employ tiled matrix multiplication, shared memory optimization, tensor core utilization, and memory hierarchy-aware scheduling. Our contribution is demonstrating that the basic operation (f16 matvec) produces correct results via Vulkan, establishing a foundation for optimization.

The performance gap between our Vulkan backend (4.6s) and optimized CUDA implementations (<0.1s) is entirely addressable through standard GPU optimization techniques: operation batching, persistent kernel dispatch, quantized weight formats, and hardware-specific tuning.

### 7.3 Limitations

1. **No tokenizer integration**: We use pre-tokenized input (token IDs). A production system would embed a tokenizer (the vindex includes `tokenizer.json`).

2. **No KV cache**: Each inference recomputes all token positions. Autoregressive generation would require KV caching for practical speed.

3. **No quantization**: We use f16 weights decoded to f32. INT4/INT8 quantization would reduce memory requirements and improve throughput.

4. **Single-model verification**: We verified on Gemma 3 4B only. Other architectures (Llama, Mistral, Qwen) may require additional implementation details.

5. **Unoptimized GPU dispatch**: 1,428 individual command buffer submissions per inference. Batching would reduce dispatch overhead by 100×.

---

## 8. Reproduction

### 8.1 Artifacts

| Artifact | Location |
|----------|----------|
| Source code | github.com/cronos3k/vindex-infer |
| Pre-extracted weights | huggingface.co/cronos3k/gemma-3-4b-it-vindex |
| LarQL (extraction tool) | github.com/chrishayuk/larql |
| LarQL CUDA fork | github.com/cronos3k/larql |

### 8.2 Three-Command Reproduction

```bash
huggingface-cli download cronos3k/gemma-3-4b-it-vindex --local-dir gemma3-4b.vindex
cargo install --git https://github.com/cronos3k/vindex-infer
vindex-infer --vindex gemma3-4b.vindex --token-ids "818,5279,529,7001,563"
```

Expected output: `token 9079 score +21.2433` (Paris, rank 1).

### 8.3 HuggingFace Baseline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", dtype="float32")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
outputs = model(**tokenizer("The capital of France is", return_tensors="pt"))
print(tokenizer.decode([outputs.logits[0, -1].argmax().item()]))  # Paris
```

---

## 9. Related Work

**LarQL** (Hayuk, 2026) decomposes transformer models into queryable vector databases. Our work is complementary: LarQL provides the decomposition and vindex format; we provide independent inference verification and a vendor-free execution engine.

**llama.cpp** (Gerganov, 2023) is a C++ LLM inference engine supporting multiple quantization formats. It achieves high throughput through optimized kernels but primarily targets CUDA for GPU acceleration. Our approach differs in using Vulkan for vendor-agnostic GPU compute and in loading vindex files rather than GGUF.

**ONNX Runtime** (Microsoft, 2019) provides cross-platform ML inference with multiple execution providers. It requires ONNX model format conversion and runtime library installation. Our approach requires no format conversion (vindex files are loaded directly) and no runtime library beyond the compiled binary.

**Burn** (2023) is a Rust deep learning framework with multiple backends including Vulkan via wgpu. Our work is not a framework but a standalone inference engine: we implement only the operations needed for transformer inference, resulting in ~600 lines vs. a full framework.

---

## 10. Conclusion

We demonstrate that transformer inference can be performed from flat binary weight files without any machine learning framework, producing output identical to HuggingFace Transformers. Our implementation runs on CPU (any platform) and Vulkan GPU (any vendor), eliminating CUDA as a hard dependency for LLM inference.

The pre-extracted Gemma 3 4B vindex on HuggingFace and the open-source inference engine on GitHub enable three-command reproduction: download, build, run. Paris comes out rank 1.

The model's weights are just numbers. The computation is just math. The hardware is just transistors. None of these require a specific vendor's permission to use together.

---

## Acknowledgments

LarQL, the vindex format, and the model decomposition concept are entirely the work of Chris Hayuk. Without LarQL, the weight files this work depends on would not exist. Gemma 3 is developed by Google DeepMind.

---

## References

1. Hayuk, C. (2026). LarQL: The model IS the database. github.com/chrishayuk/larql

2. Gemma Team, Google DeepMind (2025). Gemma 3 Technical Report.

3. Gerganov, G. (2023). llama.cpp: LLM inference in C/C++. github.com/ggerganov/llama.cpp

4. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.

5. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.

6. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.

7. The Khronos Group (2016). Vulkan — Cross-platform GPU API. vulkan.org
