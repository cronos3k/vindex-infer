//! Vindex file loader — mmap binary weight files, zero-copy access.

use std::fs::File;
use std::path::{Path, PathBuf};
use memmap2::Mmap;
use rayon::prelude::*;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VindexConfig {
    pub version: u32,
    pub model: String,
    pub family: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub embed_scale: f32,
    pub extract_level: String,
    pub dtype: String,
    pub layer_bands: Option<LayerBands>,
    pub layers: Vec<LayerInfo>,
    pub down_top_k: Option<usize>,
    pub model_config: Option<ModelConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LayerBands { pub syntax: [usize; 2], pub knowledge: [usize; 2], pub output: [usize; 2] }
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LayerInfo { pub layer: usize, pub num_features: usize, pub offset: usize, pub length: usize }
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub model_type: Option<String>, pub head_dim: Option<usize>,
    pub num_q_heads: Option<usize>, pub num_kv_heads: Option<usize>,
    pub rope_base: Option<f64>, pub sliding_window: Option<usize>,
}

pub struct Vindex {
    pub config: VindexConfig,
    gate_mmap: Mmap, embed_mmap: Mmap,
    up_mmap: Option<Mmap>, down_mmap: Option<Mmap>,
    attn_mmap: Option<Mmap>, norms_mmap: Option<Mmap>,
}

impl Vindex {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config: VindexConfig = serde_json::from_str(&std::fs::read_to_string(path.join("index.json"))?)?;
        let h = config.hidden_size;
        let gate_mmap = Self::mmap_file(&path.join("gate_vectors.bin"))?;
        let embed_mmap = Self::mmap_file(&path.join("embeddings.bin"))?;
        let up_mmap = Self::try_mmap(&path.join("up_weights.bin"));
        let down_mmap = Self::try_mmap(&path.join("down_weights.bin"));
        let attn_mmap = Self::try_mmap(&path.join("attn_weights.bin"));
        let norms_mmap = Self::try_mmap(&path.join("norms.bin"));
        log::info!("Vindex loaded: {} layers, h={}, gate={:.1}GB, attn={}",
            config.num_layers, h, gate_mmap.len() as f64/1e9,
            attn_mmap.as_ref().map_or("absent".into(), |m| format!("{:.1}GB", m.len() as f64/1e9)));
        Ok(Self { config, gate_mmap, embed_mmap, up_mmap, down_mmap, attn_mmap, norms_mmap })
    }

    pub fn attn_dims(&self) -> (usize, usize, usize) {
        let mc = self.config.model_config.as_ref();
        (mc.and_then(|c| c.num_q_heads).unwrap_or(8),
         mc.and_then(|c| c.num_kv_heads).unwrap_or(4),
         mc.and_then(|c| c.head_dim).unwrap_or(256))
    }

    pub fn embedding(&self, tid: usize) -> Vec<f32> {
        self.read_f16(&self.embed_mmap, tid * self.config.hidden_size, self.config.hidden_size)
    }

    pub fn gate_layer(&self, layer: usize) -> Vec<f32> {
        let h = self.config.hidden_size; let n = self.config.intermediate_size;
        self.read_f16(&self.gate_mmap, layer * n * h, n * h)
    }

    pub fn gate_matvec(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        let h = self.config.hidden_size; let inter = self.config.intermediate_size;
        let base = layer * inter * h * 2;
        (0..inter).into_par_iter().map(|feat| {
            let start = base + feat * h * 2;
            let end = start + h * 2;
            if end > self.gate_mmap.len() { return 0.0; }
            self.gate_mmap[start..end].chunks_exact(2).enumerate()
                .map(|(i, c)| f16_to_f32(u16::from_le_bytes([c[0],c[1]])) * x[i]).sum()
        }).collect()
    }

    pub fn up_matvec(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        let h = self.config.hidden_size; let inter = self.config.intermediate_size;
        let mmap = match &self.up_mmap { Some(m) => m, None => return vec![0.0; inter] };
        let base = layer * inter * h * 2;
        (0..inter).into_par_iter().map(|feat| {
            let start = base + feat * h * 2;
            let end = start + h * 2;
            if end > mmap.len() { return 0.0; }
            mmap[start..end].chunks_exact(2).enumerate()
                .map(|(i, c)| f16_to_f32(u16::from_le_bytes([c[0],c[1]])) * x[i]).sum()
        }).collect()
    }

    pub fn down_matvec(&self, layer: usize, act: &[f32]) -> Vec<f32> {
        let h = self.config.hidden_size; let inter = self.config.intermediate_size;
        let mmap = match &self.down_mmap { Some(m) => m, None => return vec![0.0; h] };
        // down_weights: [hidden, intermediate] per layer
        let base = layer * h * inter * 2;
        (0..h).into_par_iter().map(|row| {
            let start = base + row * inter * 2;
            let end = start + inter * 2;
            if end > mmap.len() { return 0.0; }
            mmap[start..end].chunks_exact(2).enumerate()
                .map(|(i, c)| f16_to_f32(u16::from_le_bytes([c[0],c[1]])) * act[i]).sum()
        }).collect()
    }

    pub fn attn_layer(&self, layer: usize) -> Option<AttnWeights> {
        let mmap = self.attn_mmap.as_ref()?;
        let (nq, nkv, hd) = self.attn_dims();
        let h = self.config.hidden_size;
        let q_sz = nq*hd*h; let k_sz = nkv*hd*h; let v_sz = k_sz; let o_sz = h*nq*hd;
        let layer_sz = q_sz + k_sz + v_sz + o_sz + hd + hd;
        let off = layer * layer_sz;
        Some(AttnWeights {
            w_q: self.read_f16(mmap, off, q_sz),
            w_k: self.read_f16(mmap, off + q_sz, k_sz),
            w_v: self.read_f16(mmap, off + q_sz + k_sz, v_sz),
            w_o: self.read_f16(mmap, off + q_sz + k_sz + v_sz, o_sz),
            q_norm: self.read_f16(mmap, off + q_sz + k_sz + v_sz + o_sz, hd),
            k_norm: self.read_f16(mmap, off + q_sz + k_sz + v_sz + o_sz + hd, hd),
        })
    }

    pub fn norm_weights(&self, layer: usize, idx: usize) -> Vec<f32> {
        let mmap = match &self.norms_mmap { Some(m) => m, None => return vec![0.0; self.config.hidden_size] };
        let h = self.config.hidden_size;
        self.read_f16(mmap, (layer * 4 + idx) * h, h)
    }

    pub fn final_norm(&self) -> Vec<f32> {
        let mmap = match &self.norms_mmap { Some(m) => m, None => return vec![0.0; self.config.hidden_size] };
        let h = self.config.hidden_size;
        self.read_f16(mmap, self.config.num_layers * 4 * h, h)
    }

    pub fn gate_raw(&self) -> &[u8] { &self.gate_mmap }
    pub fn up_raw(&self) -> Option<&[u8]> { self.up_mmap.as_deref() }
    pub fn down_raw(&self) -> Option<&[u8]> { self.down_mmap.as_deref() }
    pub fn attn_raw(&self) -> Option<&[u8]> { self.attn_mmap.as_deref() }
    pub fn embed_raw(&self) -> &[u8] { &self.embed_mmap }

    fn read_f16(&self, mmap: &Mmap, offset: usize, count: usize) -> Vec<f32> {
        let start = offset * 2; let end = start + count * 2;
        if end > mmap.len() { return vec![0.0; count]; }
        mmap[start..end].chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    fn mmap_file(p: &Path) -> Result<Mmap, Box<dyn std::error::Error>> {
        Ok(unsafe { Mmap::map(&File::open(p)?)? })
    }
    fn try_mmap(p: &Path) -> Option<Mmap> {
        File::open(p).ok().and_then(|f| unsafe { Mmap::map(&f).ok() })
    }
}

pub struct AttnWeights {
    pub w_q: Vec<f32>, pub w_k: Vec<f32>, pub w_v: Vec<f32>, pub w_o: Vec<f32>,
    pub q_norm: Vec<f32>, pub k_norm: Vec<f32>,
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = exp; let mut f = frac;
        while (f & 0x400) == 0 { f <<= 1; e = e.wrapping_sub(1); }
        f &= 0x3FF;
        return f32::from_bits((sign << 31) | ((127u32.wrapping_sub(15).wrapping_add(e).wrapping_add(1)) << 23) | (f << 13));
    }
    if exp == 31 { return f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13)); }
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
}
