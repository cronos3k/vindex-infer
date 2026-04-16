//! vindex-infer: Vendor-free LLM inference from decomposed weights.

mod vindex;
mod inference;

use std::path::PathBuf;
use clap::Parser;

#[derive(Parser)]
#[command(name = "vindex-infer")]
#[command(about = "Vendor-free LLM inference from decomposed weights. CPU or Vulkan GPU.")]
struct Args {
    /// Path to .vindex directory
    #[arg(short, long)]
    vindex: PathBuf,

    /// Input prompt (byte-level tokenization)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Explicit token IDs (comma-separated). Recommended over --prompt.
    #[arg(long)]
    token_ids: Option<String>,

    /// Top-K predictions to show
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// GPU index for Vulkan backend (omit for CPU-only)
    #[arg(long)]
    gpu: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("vindex-infer — Vendor-free LLM inference");
    println!();

    // Load vindex
    let vindex = vindex::Vindex::load(&args.vindex)?;
    println!("Model: {} ({} layers, h={}, vocab={})",
        vindex.config.model, vindex.config.num_layers,
        vindex.config.hidden_size, vindex.config.vocab_size);

    // Tokenize
    let mut token_ids: Vec<u32> = if let Some(ref ids) = args.token_ids {
        ids.split(',').map(|s| s.trim().parse().expect("Invalid token ID")).collect()
    } else if let Some(ref prompt) = args.prompt {
        println!("WARNING: Using byte-level tokenization. For exact results, use --token-ids");
        prompt.bytes().map(|b| b as u32).collect()
    } else {
        // Default: "The capital of France is" (Gemma 3 token IDs)
        vec![818, 5279, 529, 7001, 563]
    };

    // Prepend BOS if missing
    if token_ids.first() != Some(&2) {
        token_ids.insert(0, 2);
    }

    println!("Tokens: {:?}", token_ids);
    println!("Backend: {}", if args.gpu.is_some() { "Vulkan GPU" } else { "CPU (rayon)" });
    println!();

    // Run inference
    let result = inference::infer(&vindex, &token_ids, args.top_k);

    // Display results
    println!("Predictions:");
    for (i, (tid, score)) in result.top_k.iter().enumerate() {
        println!("  {:2}. token {:6}  score {:+.4}", i + 1, tid, score);
    }

    println!();
    println!("Time: {:.1}ms total ({:.0}ms layers, {:.0}ms logits)",
        result.total_ms, result.layer_ms, result.logits_ms);

    Ok(())
}
