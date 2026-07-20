//! Cinder micro-mixer: a distilled contextualization operator (spec §4.1 #3).
//!
//! e_i = L2norm(t_i + Wp · GELU(DW(t_{i−4..i+4}) + b1) + b2), where DW is a
//! per-dimension 9-tap depthwise filter. ~2.8k params at d=48. This is the ONLY
//! floating-point compute in the Cinder index path; everything else is lookups.

use anyhow::{Context, Result};
use std::path::Path;

pub const MIXER_WINDOW: usize = 9;
pub const MIXER_CENTER: usize = 4;
const MAGIC: &[u8; 4] = b"SXCM";
const VERSION: u32 = 1;
const MAX_DIMS: usize = 4096; // implausibility cap for forged headers

#[derive(Debug)]
pub struct MicroMixer {
    pub dims: usize,
    pub dw: Vec<f32>,
    pub b1: Vec<f32>,
    pub wp: Vec<f32>,
    pub b2: Vec<f32>,
}

/// tanh-approximation GELU (matches the derivative Task 2 implements).
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

impl MicroMixer {
    pub fn zeros(dims: usize) -> Self {
        Self {
            dims,
            dw: vec![0.0; MIXER_WINDOW * dims],
            b1: vec![0.0; dims],
            wp: vec![0.0; dims * dims],
            b2: vec![0.0; dims],
        }
    }

    pub fn forward(&self, window: &[&[f32]; MIXER_WINDOW], center: &[f32], out: &mut [f32]) {
        let d = self.dims;
        debug_assert_eq!(center.len(), d);
        debug_assert_eq!(out.len(), d);
        // h = Σ_k dw[k] ⊙ x[k] + b1
        let mut h = self.b1.clone();
        for (k, row) in window.iter().enumerate() {
            let w = &self.dw[k * d..(k + 1) * d];
            for i in 0..d {
                h[i] += w[i] * row[i];
            }
        }
        // g = GELU(h)
        for v in &mut h {
            *v = gelu(*v);
        }
        // out = center + Wp·g + b2, then L2 normalize
        for (o, (row, (&c, &b))) in out.iter_mut().zip(
            self.wp
                .chunks_exact(d)
                .zip(center.iter().zip(self.b2.iter())),
        ) {
            let dot: f32 = row.iter().zip(h.iter()).map(|(&w, &g)| w * g).sum();
            *o = c + dot + b;
        }
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in out {
                *v /= norm;
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!(
            ".{}.{}.tmp",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("mixer"),
            std::process::id()
        ));
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.dims as u32).to_le_bytes());
        buf.extend_from_slice(&(MIXER_WINDOW as u32).to_le_bytes());
        for arr in [&self.dw, &self.b1, &self.wp, &self.b2] {
            for v in arr {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut file = std::fs::File::create_new(&tmp).with_context(|| {
            format!(
                "failed to create temp file {} (leftover from a crashed run? remove it and retry)",
                tmp.display()
            )
        })?;
        if let Err(e) = std::io::Write::write_all(&mut file, &buf) {
            drop(file);
            let _ = std::fs::remove_file(&tmp);
            return Err(
                anyhow::Error::new(e).context(format!("writing mixer temp file {}", tmp.display()))
            );
        }
        drop(file);
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            anyhow::Error::new(e).context(format!("saving mixer to {}", path.display()))
        })?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path)?;
        anyhow::ensure!(buf.len() >= 16, "mixer file too short for header");
        anyhow::ensure!(&buf[0..4] == MAGIC, "invalid mixer magic");
        let rd = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        anyhow::ensure!(rd(4) == VERSION, "unsupported mixer version {}", rd(4));
        let dims = rd(8) as usize;
        let window = rd(12) as usize;
        anyhow::ensure!(
            window == MIXER_WINDOW,
            "mixer window {window} != {MIXER_WINDOW}"
        );
        anyhow::ensure!(
            dims > 0 && dims <= MAX_DIMS,
            "implausible mixer dims {dims}"
        );
        // Checked total size BEFORE allocation.
        let n_f32 = dims
            .checked_mul(MIXER_WINDOW)
            .and_then(|n| n.checked_add(dims)) // dw + b1
            .and_then(|n| dims.checked_mul(dims).map(|m| n + m)) // + wp
            .and_then(|n| n.checked_add(dims)) // + b2
            .ok_or_else(|| anyhow::anyhow!("mixer size overflows"))?;
        let expected = n_f32
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("mixer size overflows"))?;
        anyhow::ensure!(
            buf.len() - 16 == expected,
            "mixer file wrong size: {} data bytes, want {expected}",
            buf.len() - 16
        );
        let mut cursor = 16;
        let mut read_vec = |n: usize| -> Vec<f32> {
            let out = buf[cursor..cursor + n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            cursor += n * 4;
            out
        };
        Ok(Self {
            dims,
            dw: read_vec(MIXER_WINDOW * dims),
            b1: read_vec(dims),
            wp: read_vec(dims * dims),
            b2: read_vec(dims),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_mixer(dims: usize) -> MicroMixer {
        // Deterministic non-trivial weights.
        let mut m = MicroMixer::zeros(dims);
        for (i, v) in m.dw.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.05;
        }
        for (i, v) in m.wp.iter_mut().enumerate() {
            *v = ((i % 5) as f32 - 2.0) * 0.03;
        }
        for (i, v) in m.b1.iter_mut().enumerate() {
            *v = (i as f32) * 0.01;
        }
        for (i, v) in m.b2.iter_mut().enumerate() {
            *v = -(i as f32) * 0.01;
        }
        m
    }

    #[test]
    fn forward_matches_hand_computed_reference() {
        // dims=2 so the whole computation is checkable by hand in the test body.
        let mut m = MicroMixer::zeros(2);
        m.dw = vec![0.0; 18];
        m.dw[MIXER_CENTER * 2] = 1.0;
        m.dw[MIXER_CENTER * 2 + 1] = 1.0; // pick center only
        m.wp = vec![1.0, 0.0, 0.0, 1.0]; // identity
        // b1=b2=0. So delta = GELU(center); e = norm(center + GELU(center)).
        let c = [0.6f32, 0.8];
        let rows: Vec<[f32; 2]> = (0..MIXER_WINDOW).map(|_| c).collect();
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|k| rows[k].as_slice());
        let mut out = [0.0f32; 2];
        m.forward(&window, &c, &mut out);
        let want = [c[0] + gelu(c[0]), c[1] + gelu(c[1])];
        let n = (want[0] * want[0] + want[1] * want[1]).sqrt();
        assert!(
            (out[0] - want[0] / n).abs() < 1e-6 && (out[1] - want[1] / n).abs() < 1e-6,
            "got {out:?}, want normalized {want:?}"
        );
    }

    #[test]
    fn output_is_unit_norm() {
        let m = tiny_mixer(48);
        let center = vec![0.3f32; 48];
        let row = vec![0.1f32; 48];
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|_| row.as_slice());
        let mut out = vec![0.0f32; 48];
        m.forward(&window, &center, &mut out);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "norm {n}");
    }

    #[test]
    fn zero_mixer_is_identity_normalize() {
        // All-zero weights => delta = Wp·GELU(b1)+b2 = 0 => e = norm(center).
        let m = MicroMixer::zeros(4);
        let center = [2.0f32, 0.0, 0.0, 0.0];
        let row = [9.0f32, 9.0, 9.0, 9.0];
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|_| row.as_slice());
        let mut out = [0.0f32; 4];
        m.forward(&window, &center, &mut out);
        assert_eq!(out, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p = tmp.path().join("cinder_mixer.bin");
        let m = tiny_mixer(48);
        m.save(&p).unwrap();
        let l = MicroMixer::load(&p).unwrap();
        assert_eq!(l.dims, 48);
        assert_eq!(l.dw, m.dw);
        assert_eq!(l.b1, m.b1);
        assert_eq!(l.wp, m.wp);
        assert_eq!(l.b2, m.b2);
    }

    #[test]
    fn load_rejects_forged_headers() {
        let tmp = tempfile::TempDir::new().unwrap();
        let forge = |dims: u32, window: u32| {
            let mut b = Vec::new();
            b.extend_from_slice(b"SXCM");
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&dims.to_le_bytes());
            b.extend_from_slice(&window.to_le_bytes());
            b
        };
        for (name, bytes) in [
            ("overflow", forge(u32::MAX, u32::MAX)),
            ("huge", forge(1_000_000, 9)),
            ("truncated", forge(48, 9)),
            ("bad-window", forge(48, 7)),
        ] {
            let p = tmp.path().join(name);
            std::fs::write(&p, bytes).unwrap();
            assert!(MicroMixer::load(&p).is_err(), "{name} must be rejected");
        }
        let p = tmp.path().join("bad-magic");
        std::fs::write(&p, b"NOPE").unwrap();
        assert!(MicroMixer::load(&p).is_err());
    }
}
