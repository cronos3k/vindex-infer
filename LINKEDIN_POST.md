# LinkedIn Post

---

**I just ran a 4-billion parameter LLM without PyTorch, without CUDA, and without any ML framework. Here's how.**

Last week, Chris Hayuk (@chrishayuk) released LarQL — a tool that decomposes any open-source transformer into a queryable vector database called a "vindex." The model's weights are split by function: gate projections, attention matrices, embeddings — each in its own flat binary file.

I looked at those files and thought: if the weights are just numbers in files, why do we need PyTorch to multiply them?

So I wrote a Rust program that memory-maps the vindex files, runs the full 34-layer transformer computation (attention + FFN + norms), and produces the exact same output as HuggingFace Transformers. Verified to 5 significant figures across every layer.

"The capital of France is" → Paris (#1, score +21.24)
"The largest planet is" → Jupiter (#1)
"Einstein was born in" → Ulm (#1)

The whole inference engine is ~600 lines of Rust. No Python runtime. No CUDA toolkit. No ML libraries.

**Why this matters:**

Every LLM inference solution today locks you into a vendor stack:
- PyTorch → CUDA → NVIDIA
- MLX → Apple Silicon only
- TensorRT → NVIDIA only

The vindex format doesn't care what GPU you have. The inference runs on:
- CPU (any platform, via rayon multi-threading)
- Vulkan GPU (any vendor — NVIDIA, AMD, Intel, Qualcomm)

I also ported LarQL to CUDA for Linux/Windows (github.com/cronos3k/larql) and uploaded a pre-extracted Gemma 3 4B vindex to HuggingFace so anyone can test immediately.

**Try it yourself:**

```
huggingface-cli download cronos3k/gemma-3-4b-it-vindex --local-dir gemma3-4b.vindex
cargo install --git https://github.com/cronos3k/vindex-infer
vindex-infer --vindex gemma3-4b.vindex --token-ids "818,5279,529,7001,563"
```

Three commands. Any machine. Paris comes out #1.

Links:
- Inference engine: github.com/cronos3k/vindex-infer
- Pre-extracted model: huggingface.co/cronos3k/gemma-3-4b-it-vindex
- LarQL (original): github.com/chrishayuk/larql
- LarQL CUDA fork: github.com/cronos3k/larql

Credits: LarQL, the vindex format, and the decomposition concept are entirely the work of Chris Hayuk. My contribution is the independent inference engine, the Vulkan GPU backend, the CUDA port, and the verification that exact output is achievable from vindex files alone.

#LLM #Rust #Vulkan #OpenSource #AI #MachineLearning #InferencEngine

---
