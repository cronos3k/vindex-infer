//! Inference engine: attention + FFN, verified exact match with HuggingFace.

use rayon::prelude::*;
use crate::vindex::Vindex;

pub struct InferenceResult {
    pub top_k: Vec<(u32, f32)>,
    pub total_ms: f64,
    pub layer_ms: f64,
    pub logits_ms: f64,
}

pub fn infer(vindex: &Vindex, token_ids: &[u32], top_k: usize) -> InferenceResult {
    let total = std::time::Instant::now();
    let h = vindex.config.hidden_size;
    let nl = vindex.config.num_layers;
    let (nq, nkv, hd) = vindex.attn_dims();
    let groups = nq / nkv;
    let inter = vindex.config.intermediate_size;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let is_global = |l: usize| (l + 1) % 6 == 0;

    let scale = vindex.config.embed_scale;
    let mut residuals: Vec<Vec<f32>> = token_ids.iter().map(|&tid| {
        vindex.embedding(tid as usize).iter().map(|v| v * scale).collect()
    }).collect();
    let nt = residuals.len();

    let layer_start = std::time::Instant::now();
    for layer in 0..nl {
        let iln = vindex.norm_weights(layer, 0);
        let paln = vindex.norm_weights(layer, 1);
        let pfln = vindex.norm_weights(layer, 2);
        let pfnl = vindex.norm_weights(layer, 3);
        let attn = match vindex.attn_layer(layer) { Some(a) => a, None => continue };
        let (rb, rf) = if is_global(layer) { (1e6f64, 8.0f64) } else { (1e4f64, 1.0f64) };

        // Pre-attn norm + Q/K/V projections
        let normed: Vec<Vec<f32>> = residuals.iter().map(|r| rms_norm_1(r, &iln)).collect();
        let mut all_q = Vec::with_capacity(nt);
        let mut all_k = Vec::with_capacity(nt);
        let mut all_v = Vec::with_capacity(nt);
        for tok in 0..nt {
            let mut q = matvec_par(&attn.w_q, &normed[tok], nq*hd, h);
            let mut k = matvec_par(&attn.w_k, &normed[tok], nkv*hd, h);
            let v = matvec_par(&attn.w_v, &normed[tok], nkv*hd, h);
            for hi in 0..nq { rms_norm_qk(&mut q[hi*hd..(hi+1)*hd], &attn.q_norm); }
            for hi in 0..nkv { rms_norm_qk(&mut k[hi*hd..(hi+1)*hd], &attn.k_norm); }
            let pos = tok as f64 / rf;
            for hi in 0..nq { apply_rope_hf(&mut q[hi*hd..(hi+1)*hd], pos, rb, hd); }
            for hi in 0..nkv { apply_rope_hf(&mut k[hi*hd..(hi+1)*hd], pos, rb, hd); }
            all_q.push(q); all_k.push(k); all_v.push(v);
        }

        // Causal attention + O projection
        for tok in 0..nt {
            let mut ho = vec![0.0f32; nq*hd];
            for hi in 0..nq {
                let kv_hi = hi/groups; let qs = hi*hd; let ks = kv_hi*hd;
                let mut scores: Vec<f32> = (0..=tok).map(|j|
                    (0..hd).map(|d| all_q[tok][qs+d]*all_k[j][ks+d]).sum::<f32>() * attn_scale
                ).collect();
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_s: Vec<f32> = scores.iter().map(|&s| (s-max_s).exp()).collect();
                let sum_e: f32 = exp_s.iter().sum();
                for j in 0..=tok {
                    let w = exp_s[j] / sum_e.max(1e-10);
                    for d in 0..hd { ho[qs+d] += w * all_v[j][ks+d]; }
                }
            }
            let attn_out = matvec_par(&attn.w_o, &ho, h, nq*hd);
            let na = rms_norm_1(&attn_out, &paln);
            for (r, a) in residuals[tok].iter_mut().zip(na.iter()) { *r += a; }
        }

        // FFN (dense)
        for tok in 0..nt {
            let x = rms_norm_1(&residuals[tok], &pfln);
            let gs = vindex.gate_matvec(layer, &x);
            let us = vindex.up_matvec(layer, &x);
            let act: Vec<f32> = gs.iter().zip(us.iter())
                .map(|(&g, &u)| gelu_tanh(g) * u).collect();
            let delta = vindex.down_matvec(layer, &act);
            let nf = rms_norm_1(&delta, &pfnl);
            for (r, d) in residuals[tok].iter_mut().zip(nf.iter()) { *r += d; }
        }

        if (layer+1) % 10 == 0 || layer == nl-1 {
            let norm = residuals[nt-1].iter().map(|v| (*v as f64)*(*v as f64)).sum::<f64>().sqrt();
            log::info!("  Layer {}/{}: norm={:.1}", layer+1, nl, norm);
        }
    }
    let layer_ms = layer_start.elapsed().as_secs_f64() * 1000.0;

    // Final norm + logits
    let logits_start = std::time::Instant::now();
    let final_n = vindex.final_norm();
    let last = rms_norm_1(&residuals[nt-1], &final_n);
    let vocab = vindex.config.vocab_size;
    let logits: Vec<f32> = (0..vocab).into_par_iter().map(|tid| {
        let e = vindex.embedding(tid);
        last.iter().zip(e.iter()).map(|(r, ei)| r * ei).sum()
    }).collect();
    let logits_ms = logits_start.elapsed().as_secs_f64() * 1000.0;

    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &s)| (i as u32, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    InferenceResult {
        top_k: indexed,
        total_ms: total.elapsed().as_secs_f64() * 1000.0,
        layer_ms, logits_ms,
    }
}

fn rms_norm_1(x: &[f32], g: &[f32]) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>() / n as f64 + 1e-6).sqrt() as f32;
    x.iter().zip(g.iter()).map(|(&v, &gi)| (v/rms)*(1.0+gi)).collect()
}

fn rms_norm_qk(h: &mut [f32], g: &[f32]) {
    let n = h.len();
    let rms = (h.iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>() / n as f64 + 1e-6).sqrt() as f32;
    for (v, &gi) in h.iter_mut().zip(g.iter()) { *v = (*v/rms)*(1.0+gi); }
}

fn apply_rope_hf(h: &mut [f32], pos: f64, base: f64, hd: usize) {
    let half = hd/2;
    for i in 0..half {
        let f = pos / base.powf(2.0*i as f64 / hd as f64);
        let (c, s) = (f.cos() as f32, f.sin() as f32);
        let (x1, x2) = (h[i], h[i+half]);
        h[i] = x1*c - x2*s; h[i+half] = x2*c + x1*s;
    }
}

fn gelu_tanh(x: f32) -> f32 {
    0.5*x*(1.0 + ((2.0f32/std::f32::consts::PI).sqrt()*(x+0.044715*x*x*x)).tanh())
}

fn matvec_par(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).into_par_iter().map(|r| {
        let s = r * cols;
        w[s..s+cols].iter().zip(x.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}
