//! Cinder mixer training: hand-rolled backprop + Adam for [`MicroMixer`]
//! (spec §4.1 #3, plan Task 2).
//!
//! Trains the depthwise-mix + GELU + linear-mix `MicroMixer` (see
//! `embedding/mixer.rs`) to imitate a TEACHER contextual encoder
//! (`DocTokenEncoder`) using only STUDENT inputs gathered from a frozen
//! [`StaticTokenTable`] (Ember Plan A). Backprop, Adam, and the reservoir /
//! init / shuffle RNGs are all hand-rolled — this crate has no autodiff or
//! ML-framework dependency, and the model itself is tiny (~2.8k params at
//! d=48), so a small closed-form backward pass is simpler than a dependency.

use crate::embedding::colbert::TokenEmbeddings;
use crate::embedding::mixer::{MIXER_CENTER, MIXER_WINDOW, MicroMixer, gelu};
use crate::embedding::static_distill::DocTokenEncoder;
use crate::embedding::static_table::StaticTokenTable;
use anyhow::Result;

/// Fixed seed for reservoir-sampling `(window, teacher_row)` pairs while
/// streaming the corpus. Training is meant to be a reproducible offline
/// batch job (same rationale as `static_distill::RESERVOIR_SEED`).
const RESERVOIR_SEED: u64 = 0xC10D_5EED_0000_0001;
/// Fixed seed for `MicroMixer` weight initialization.
const INIT_SEED: u64 = 0xC10D_5EED_0000_0002;
/// Fixed seed for the deterministic shuffle applied before the train/holdout split.
const SHUFFLE_SEED: u64 = 0xC10D_5EED_0000_0003;

/// Uniform-initialization half-range: non-center weights start in
/// `[-INIT_SCALE, INIT_SCALE)`.
const INIT_SCALE: f32 = 0.05;

const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Width of the Ember Plan A linear-baseline mixing window —
/// `StaticTokenTable::mix_weights` is a `[f32; 5]` fit against exactly this
/// window width. Duplicated here because `static_token::WINDOW_LEN` /
/// `static_distill::WINDOW_LEN` are both private to their own modules.
const LINEAR_WINDOW_LEN: usize = 5;
/// Index of the center token within the linear-baseline window.
const LINEAR_CENTER_OFFSET: usize = 2;

/// Options controlling [`train_mixer`].
pub struct MixerTrainOptions {
    /// Max `(window, teacher_row)` pairs retained by reservoir sampling.
    pub sample_capacity: usize,
    /// Number of passes over the training split.
    pub epochs: usize,
    /// Encoder batch size during corpus ingestion, and minibatch size during
    /// the training epoch loop.
    pub batch: usize,
    /// Initial Adam learning rate; halved after every epoch.
    pub lr: f32,
    /// Fraction of sampled pairs held out for the reported cosine metrics.
    pub holdout_frac: f32,
}

impl Default for MixerTrainOptions {
    fn default() -> Self {
        Self {
            sample_capacity: 2_000_000,
            epochs: 3,
            batch: 32,
            lr: 1e-3,
            holdout_frac: 0.05,
        }
    }
}

/// Outcome of a [`train_mixer`] run.
#[derive(Debug)]
pub struct MixerTrainReport {
    /// Number of `(window, teacher_row)` pairs actually trained on (excludes
    /// the held-out split).
    pub pairs_trained: usize,
    /// Mean `cos(student, teacher)` on the held-out split for the trained
    /// [`MicroMixer`].
    pub holdout_cosine_mixer: f32,
    /// Mean `cos(student, teacher)` on the same held-out split for the Ember
    /// Plan A 5-tap linear baseline (`table.mix_weights`), for comparison.
    pub holdout_cosine_linear: f32,
}

/// Minimal splitmix64 PRNG, used only for reservoir sampling, weight
/// initialization, and the deterministic pre-training shuffle. Copied from
/// `static_distill::SplitMix64` / `centroid_train::SplitMix64` (both private
/// to their own modules) — a tiny inline generator avoids pulling in the
/// `rand` crate for these calls; none of them need cryptographic randomness.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish integer in `[0, bound)`. `bound` must be > 0.
    fn next_below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// Uniform-ish `f32` in `[0, 1)`, built from the top 24 bits of
    /// `next_u64` — enough precision for weight initialization.
    fn next_unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / (1u32 << 24) as f32
    }

    /// Uniform-ish `f32` in `[lo, hi)`.
    fn next_range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_unit_f32() * (hi - lo)
    }
}

/// Fisher-Yates shuffle of `items`, in place, using `rng`.
fn shuffle<T>(items: &mut [T], rng: &mut SplitMix64) {
    for i in (1..items.len()).rev() {
        let j = rng.next_below(i as u64 + 1) as usize;
        items.swap(i, j);
    }
}

/// Initialize a [`MicroMixer`] for training: `dw`'s center tap is left at
/// zero (the residual path already carries the center token, per the
/// spec — see `MicroMixer::forward`'s `center + Wp·g + b2`), every other
/// weight (`dw`'s non-center taps, all of `wp`) is drawn uniformly from
/// `[-INIT_SCALE, INIT_SCALE)` via a fixed-seed [`SplitMix64`], and biases
/// start at zero (`MicroMixer::zeros`'s default).
fn init_mixer(dims: usize, seed: u64) -> MicroMixer {
    let mut rng = SplitMix64::new(seed);
    let mut m = MicroMixer::zeros(dims);
    for k in 0..MIXER_WINDOW {
        if k == MIXER_CENTER {
            continue; // stays zero
        }
        for i in 0..dims {
            m.dw[k * dims + i] = rng.next_range_f32(-INIT_SCALE, INIT_SCALE);
        }
    }
    for v in &mut m.wp {
        *v = rng.next_range_f32(-INIT_SCALE, INIT_SCALE);
    }
    m
}

/// Gradients for every [`MicroMixer`] parameter tensor, same shapes as the
/// model itself. `pub(crate)` (rather than private) because it is the
/// return type of `pub(crate) fn backward`.
pub(crate) struct Gradients {
    pub(crate) dw: Vec<f32>,
    pub(crate) b1: Vec<f32>,
    pub(crate) wp: Vec<f32>,
    pub(crate) b2: Vec<f32>,
}

impl Gradients {
    fn zeros(dims: usize) -> Self {
        Self {
            dw: vec![0.0; MIXER_WINDOW * dims],
            b1: vec![0.0; dims],
            wp: vec![0.0; dims * dims],
            b2: vec![0.0; dims],
        }
    }

    fn add_assign(&mut self, other: &Gradients) {
        for (a, b) in self.dw.iter_mut().zip(other.dw.iter()) {
            *a += b;
        }
        for (a, b) in self.b1.iter_mut().zip(other.b1.iter()) {
            *a += b;
        }
        for (a, b) in self.wp.iter_mut().zip(other.wp.iter()) {
            *a += b;
        }
        for (a, b) in self.b2.iter_mut().zip(other.b2.iter()) {
            *a += b;
        }
    }

    fn scale(&mut self, s: f32) {
        for v in &mut self.dw {
            *v *= s;
        }
        for v in &mut self.b1 {
            *v *= s;
        }
        for v in &mut self.wp {
            *v *= s;
        }
        for v in &mut self.b2 {
            *v *= s;
        }
    }
}

/// Derivative of the tanh-approximation GELU used by [`gelu`] (see
/// `mixer.rs`). With `s = SQRT_2_OVER_PI·(x + 0.044715x³)`, `t = tanh(s)`:
/// `gelu'(x) = 0.5(1+t) + 0.5x(1−t²)·SQRT_2_OVER_PI·(1 + 3·0.044715x²)`.
fn gelu_prime(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const C: f32 = 0.044_715;
    let s = SQRT_2_OVER_PI * (x + C * x * x * x);
    let t = s.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * C * x * x)
}

/// Un-normalized forward pass, reused by [`backward`]. Mirrors
/// `MicroMixer::forward` exactly, minus the final L2 normalization — the
/// loss is defined in terms of `u` pre-normalization.
// Variable names (`m`, `d`, `h`, `g`, `u`) mirror the spec's math notation
// directly (`h = Σ dw⊙x + b1`, `g = GELU(h)`, `u = center + Wp·g + b2`).
#[allow(clippy::many_single_char_names)]
fn forward_raw(
    m: &MicroMixer,
    window: &[&[f32]; MIXER_WINDOW],
    center: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = m.dims;
    let mut h = m.b1.clone();
    for (k, row) in window.iter().enumerate() {
        let w = &m.dw[k * d..(k + 1) * d];
        for i in 0..d {
            h[i] += w[i] * row[i];
        }
    }
    let g: Vec<f32> = h.iter().map(|&v| gelu(v)).collect();
    let mut u = vec![0.0f32; d];
    for (o, uo) in u.iter_mut().enumerate() {
        let row = &m.wp[o * d..(o + 1) * d];
        let dot: f32 = row.iter().zip(g.iter()).map(|(&w, &gi)| w * gi).sum();
        *uo = center[o] + dot + m.b2[o];
    }
    (h, g, u)
}

/// Loss + analytic gradients for one training example.
///
/// `L = 1 − cos(u, y)` where `u = center + Wp·GELU(h) + b2` is the
/// PRE-normalization mixer output and `y` is a unit-norm teacher target
/// (`cos(u, y) = (u·y)/‖u‖` when `‖y‖ = 1`). Table rows (`window`, `center`)
/// are frozen — no input gradient is computed for them.
// See `forward_raw`'s note: variable names mirror the spec's math notation.
#[allow(clippy::many_single_char_names)]
pub(crate) fn backward(
    m: &MicroMixer,
    window: &[&[f32]; MIXER_WINDOW],
    center: &[f32],
    y: &[f32],
) -> (f32, Gradients) {
    let d = m.dims;
    let (h, g, u) = forward_raw(m, window, center);

    let norm: f32 = u.iter().map(|v| v * v).sum::<f32>().sqrt();
    let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    let inv_norm3 = inv_norm * inv_norm * inv_norm;
    let dot_uy: f32 = u.iter().zip(y.iter()).map(|(&ui, &yi)| ui * yi).sum();
    let loss = 1.0 - dot_uy * inv_norm;

    // dL/du = -( y/‖u‖ - (u·y)·u/‖u‖³ )
    let dl_du: Vec<f32> = u
        .iter()
        .zip(y.iter())
        .map(|(&ui, &yi)| -(yi * inv_norm - dot_uy * ui * inv_norm3))
        .collect();

    // db2 = dL/du
    let db2 = dl_du.clone();

    // dWp[o][i] = dL/du[o] · g[i]
    let mut dwp = vec![0.0f32; d * d];
    for o in 0..d {
        let dlo = dl_du[o];
        for i in 0..d {
            dwp[o * d + i] = dlo * g[i];
        }
    }

    // dg = Wpᵀ·dL/du
    let mut dg = vec![0.0f32; d];
    for (o, &dlo) in dl_du.iter().enumerate() {
        let row = &m.wp[o * d..(o + 1) * d];
        for (dgi, &w) in dg.iter_mut().zip(row.iter()) {
            *dgi += w * dlo;
        }
    }

    // dh = dg ⊙ gelu'(h); db1 = dh
    let dh: Vec<f32> = dg
        .iter()
        .zip(h.iter())
        .map(|(&dgi, &hi)| dgi * gelu_prime(hi))
        .collect();
    let db1 = dh.clone();

    // ddw[k][d] = dh[d] · x[k][d]
    let mut ddw = vec![0.0f32; MIXER_WINDOW * d];
    for (k, row) in window.iter().enumerate() {
        for i in 0..d {
            ddw[k * d + i] = dh[i] * row[i];
        }
    }

    (
        loss,
        Gradients {
            dw: ddw,
            b1: db1,
            wp: dwp,
            b2: db2,
        },
    )
}

/// Adam optimizer state, shaped like the model's gradients. Standard
/// bias-corrected update (β1=0.9, β2=0.999, ε=1e-8).
struct Adam {
    m: Gradients,
    v: Gradients,
    t: u64,
}

impl Adam {
    fn new(dims: usize) -> Self {
        Self {
            m: Gradients::zeros(dims),
            v: Gradients::zeros(dims),
            t: 0,
        }
    }

    fn step(&mut self, model: &mut MicroMixer, grads: &Gradients, lr: f32) {
        self.t += 1;
        let bc1 = 1.0 - ADAM_BETA1.powi(self.t as i32);
        let bc2 = 1.0 - ADAM_BETA2.powi(self.t as i32);
        Self::update(
            &mut model.dw,
            &mut self.m.dw,
            &mut self.v.dw,
            &grads.dw,
            lr,
            bc1,
            bc2,
        );
        Self::update(
            &mut model.b1,
            &mut self.m.b1,
            &mut self.v.b1,
            &grads.b1,
            lr,
            bc1,
            bc2,
        );
        Self::update(
            &mut model.wp,
            &mut self.m.wp,
            &mut self.v.wp,
            &grads.wp,
            lr,
            bc1,
            bc2,
        );
        Self::update(
            &mut model.b2,
            &mut self.m.b2,
            &mut self.v.b2,
            &grads.b2,
            lr,
            bc1,
            bc2,
        );
    }

    fn update(
        param: &mut [f32],
        m: &mut [f32],
        v: &mut [f32],
        g: &[f32],
        lr: f32,
        bc1: f32,
        bc2: f32,
    ) {
        for i in 0..param.len() {
            m[i] = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * g[i];
            v[i] = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * g[i] * g[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            param[i] -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
        }
    }
}

/// One retained `(window, teacher)` pair: the [`MIXER_WINDOW`] token ids
/// surrounding (and including) a position, and the TEACHER's contextual
/// embedding at that position. Stored as `f16` — reservoir capacity can run
/// into the millions, and the STUDENT side is reconstructed cheaply from
/// `table` at train time, so only the teacher target needs to be retained.
struct ReservoirItem {
    window_ids: [u32; MIXER_WINDOW],
    teacher_row: Vec<half::f16>,
}

/// Streaming reservoir-sample accumulator, mirroring
/// `static_distill::Accumulator` / `centroid_train::Accumulator`'s shape.
struct Accumulator {
    capacity: usize,
    reservoir: Vec<ReservoirItem>,
    seen: u64,
    rng: SplitMix64,
    dims: Option<usize>,
}

impl Accumulator {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            reservoir: Vec::new(),
            seen: 0,
            rng: SplitMix64::new(RESERVOIR_SEED),
            dims: None,
        }
    }

    fn ingest_batch(
        &mut self,
        encoder: &dyn DocTokenEncoder,
        table: &StaticTokenTable,
        texts: &[String],
    ) -> Result<()> {
        for (ids, emb) in encoder.encode_with_ids(texts)? {
            self.ingest_document(table, &ids, &emb)?;
        }
        Ok(())
    }

    fn ingest_document(
        &mut self,
        table: &StaticTokenTable,
        ids: &[u32],
        emb: &TokenEmbeddings,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        anyhow::ensure!(
            ids.len() == emb.nrows(),
            "train_mixer: token id/embedding row mismatch ({} ids vs {} rows); \
             DocTokenEncoder contract violated",
            ids.len(),
            emb.nrows()
        );

        let ncols = emb.ncols();
        match self.dims {
            None => self.dims = Some(ncols),
            Some(d) => anyhow::ensure!(
                d == ncols,
                "train_mixer: inconsistent embedding width across batches ({d} vs {ncols})"
            ),
        }

        for i in 0..ids.len() {
            let center_id = ids[i];
            // Unseen center token: the student side has nothing to mix from.
            if table.lookup(center_id).is_none() {
                continue;
            }

            // Same edge-replication convention as
            // `static_distill::Accumulator::ingest_document` /
            // `static_token::window_ids_at`, generalized from their 5-wide
            // window to MIXER_WINDOW/MIXER_CENTER: a window position past
            // either edge of the document reuses the CENTER token's own id.
            let mut window_ids = [center_id; MIXER_WINDOW];
            for offset in 1..=MIXER_CENTER {
                if i >= offset {
                    window_ids[MIXER_CENTER - offset] = ids[i - offset];
                }
                if i + offset < ids.len() {
                    window_ids[MIXER_CENTER + offset] = ids[i + offset];
                }
            }

            let row = emb.row(i);
            let teacher_row: Vec<half::f16> = row.iter().map(|&v| half::f16::from_f32(v)).collect();
            self.reservoir_sample(ReservoirItem {
                window_ids,
                teacher_row,
            });
        }
        Ok(())
    }

    /// Textbook reservoir sampling (Algorithm R) — see
    /// `static_distill::Accumulator::reservoir_sample` for the invariant.
    fn reservoir_sample(&mut self, item: ReservoirItem) {
        if self.reservoir.len() < self.capacity {
            self.reservoir.push(item);
        } else {
            let j = self.rng.next_below(self.seen + 1);
            if (j as usize) < self.capacity {
                self.reservoir[j as usize] = item;
            }
        }
        self.seen += 1;
    }
}

/// Gather STUDENT inputs for one reservoir item from `table`: the
/// [`MIXER_WINDOW`] window rows (a missing/unseen lookup contributes an
/// all-zero row, same convention `static_token::mix_token` uses for
/// non-center misses) and the center row (guaranteed present — positions
/// with an unseen center token are filtered out at ingestion time).
fn gather_student_inputs(
    table: &StaticTokenTable,
    window_ids: &[u32; MIXER_WINDOW],
    dims: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let window_rows: Vec<Vec<f32>> = window_ids
        .iter()
        .map(|&id| {
            table
                .lookup(id)
                .map_or_else(|| vec![0.0f32; dims], <[f32]>::to_vec)
        })
        .collect();
    let center_row = window_rows[MIXER_CENTER].clone();
    (window_rows, center_row)
}

/// L2-normalize `v` in place; leaves an all-zero `v` unchanged.
fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity; `0.0` if either vector is all-zero.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// The Ember Plan A 5-tap linear baseline, mirroring
/// `static_token::mix_token`'s formula exactly, evaluated over the middle 5
/// entries of a [`MIXER_WINDOW`]-wide window (offsets −2..=+2 from center).
/// That slice reproduces exactly what an independent 5-window edge-
/// replication build would produce, because both conventions apply the same
/// "reuse center past either edge" rule per-offset, independent of the outer
/// window's width.
fn mix_token_linear(table: &StaticTokenTable, window_ids: &[u32; MIXER_WINDOW]) -> Vec<f32> {
    let dims = table.dims;
    let sub: [u32; LINEAR_WINDOW_LEN] =
        std::array::from_fn(|k| window_ids[MIXER_CENTER - LINEAR_CENTER_OFFSET + k]);
    let weights = table.mix_weights;
    let rows: [Option<&[f32]>; LINEAR_WINDOW_LEN] = std::array::from_fn(|k| table.lookup(sub[k]));

    if rows[LINEAR_CENTER_OFFSET].is_some() {
        let mut v = vec![0.0f32; dims];
        for (k, row) in rows.iter().enumerate() {
            if let Some(row) = row {
                let w = weights[k];
                for (vi, &ri) in v.iter_mut().zip(row.iter()) {
                    *vi += w * ri;
                }
            }
        }
        normalize_in_place(&mut v);
        v
    } else {
        // Unreachable given the ingestion-time center filter (the center of
        // `sub` is the same table row as `window_ids`' center), but mirrors
        // `mix_token`'s fallback branch exactly for fidelity.
        let mut sum = vec![0.0f32; dims];
        let mut hits: u32 = 0;
        for (k, row) in rows.iter().enumerate() {
            if k == LINEAR_CENTER_OFFSET {
                continue;
            }
            if let Some(row) = row {
                hits += 1;
                for (si, &ri) in sum.iter_mut().zip(row.iter()) {
                    *si += ri;
                }
            }
        }
        if hits == 0 {
            return vec![0.0f32; dims];
        }
        let inv = 1.0 / hits as f32;
        for s in &mut sum {
            *s *= inv;
        }
        normalize_in_place(&mut sum);
        sum
    }
}

/// Streams `corpus` through the TEACHER `encoder` (contextual), reservoir-
/// samples `(window_ids: [u32; MIXER_WINDOW], teacher_row: f16[dims])`
/// pairs, then trains a [`MicroMixer`] by gathering STUDENT inputs from
/// `table` rows. Positions whose center-token table row is all-zero are
/// skipped (unseen tokens; miss rate ≤0.74% per Gate-1 data). Deterministic:
/// fixed seeds for reservoir, init, and shuffling.
///
/// # Errors
///
/// Returns an error if `corpus` (or every position in it) produces no usable
/// `(window, teacher)` pairs — an empty corpus must fail loudly rather than
/// silently returning an untrained model — if `table.dims` doesn't match the
/// teacher encoder's embedding width, or if any batch's `encode_with_ids`
/// call fails.
pub fn train_mixer(
    encoder: &dyn DocTokenEncoder,
    table: &StaticTokenTable,
    corpus: impl Iterator<Item = String>,
    opts: &MixerTrainOptions,
) -> Result<(MicroMixer, MixerTrainReport)> {
    let batch = opts.batch.max(1);
    let mut acc = Accumulator::new(opts.sample_capacity);
    let mut batch_buf: Vec<String> = Vec::with_capacity(batch);
    for text in corpus {
        batch_buf.push(text);
        if batch_buf.len() == batch {
            acc.ingest_batch(encoder, table, &batch_buf)?;
            batch_buf.clear();
        }
    }
    if !batch_buf.is_empty() {
        acc.ingest_batch(encoder, table, &batch_buf)?;
    }

    anyhow::ensure!(
        !acc.reservoir.is_empty(),
        "train_mixer: empty corpus — no usable (window, teacher) pairs \
         (either the corpus was empty or every token missed the table)"
    );
    let dims = acc.dims.expect("non-empty reservoir implies dims is set");
    anyhow::ensure!(
        table.dims == dims,
        "train_mixer: table dims {} != teacher embedding dims {dims}",
        table.dims
    );

    let reservoir = acc.reservoir;
    let n = reservoir.len();
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order, &mut SplitMix64::new(SHUFFLE_SEED));

    let holdout_n = if n >= 2 {
        (((n as f32) * opts.holdout_frac).round() as usize).clamp(1, n - 1)
    } else {
        0
    };
    let (holdout_idx, train_idx) = order.split_at(holdout_n);

    let mut model = init_mixer(dims, INIT_SEED);
    let mut adam = Adam::new(dims);
    let mut lr = opts.lr;
    for _epoch in 0..opts.epochs {
        for chunk in train_idx.chunks(batch) {
            let mut grad_sum = Gradients::zeros(dims);
            for &idx in chunk {
                let item = &reservoir[idx];
                let (window_rows, center_row) =
                    gather_student_inputs(table, &item.window_ids, dims);
                let window_refs: [&[f32]; MIXER_WINDOW] =
                    std::array::from_fn(|k| window_rows[k].as_slice());
                let y: Vec<f32> = item.teacher_row.iter().map(|v| v.to_f32()).collect();
                let (_loss, grads) = backward(&model, &window_refs, &center_row, &y);
                grad_sum.add_assign(&grads);
            }
            grad_sum.scale(1.0 / chunk.len() as f32);
            adam.step(&mut model, &grad_sum, lr);
        }
        lr *= 0.5;
    }

    let mut sum_mixer = 0.0f64;
    let mut sum_linear = 0.0f64;
    for &idx in holdout_idx {
        let item = &reservoir[idx];
        let y: Vec<f32> = item.teacher_row.iter().map(|v| v.to_f32()).collect();
        let (window_rows, center_row) = gather_student_inputs(table, &item.window_ids, dims);
        let window_refs: [&[f32]; MIXER_WINDOW] =
            std::array::from_fn(|k| window_rows[k].as_slice());

        let mut mixer_out = vec![0.0f32; dims];
        model.forward(&window_refs, &center_row, &mut mixer_out);
        sum_mixer += f64::from(cosine(&mixer_out, &y));

        let linear_out = mix_token_linear(table, &item.window_ids);
        sum_linear += f64::from(cosine(&linear_out, &y));
    }
    let holdout_n_f = holdout_idx.len().max(1) as f64;
    let report = MixerTrainReport {
        pairs_trained: train_idx.len(),
        holdout_cosine_mixer: (sum_mixer / holdout_n_f) as f32,
        holdout_cosine_linear: (sum_linear / holdout_n_f) as f32,
    };
    Ok((model, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mixer::{MIXER_WINDOW, MicroMixer};
    use ndarray::Array2;

    /// THE load-bearing test: analytic gradients match central finite differences.
    #[test]
    fn gradient_check_matches_finite_differences() {
        let dims = 3;
        let mut m = MicroMixer::zeros(dims);
        // Non-trivial deterministic weights.
        for (i, v) in m.dw.iter_mut().enumerate() {
            *v = ((i % 5) as f32 - 2.0) * 0.11;
        }
        for (i, v) in m.wp.iter_mut().enumerate() {
            *v = ((i % 3) as f32 - 1.0) * 0.17;
        }
        for (i, v) in m.b1.iter_mut().enumerate() {
            *v = 0.03 * i as f32;
        }
        for (i, v) in m.b2.iter_mut().enumerate() {
            *v = -0.02 * i as f32;
        }

        let window_rows: Vec<Vec<f32>> = (0..MIXER_WINDOW)
            .map(|k| (0..dims).map(|d| ((k + d) as f32 * 0.31).sin()).collect())
            .collect();
        let center: Vec<f32> = window_rows[super::super::mixer::MIXER_CENTER].clone();
        let mut y: Vec<f32> = (0..dims).map(|d| (d as f32 + 1.0).cos()).collect();
        let n: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in &mut y {
            *v /= n;
        }

        let grads = compute_gradients(&m, &window_rows, &center, &y);

        let eps = 1e-3f32;
        // Spot-check 12 parameters across all four tensors via finite differences.
        let mut checks: Vec<(&str, usize)> = vec![
            ("dw", 0),
            ("dw", 7),
            ("dw", MIXER_WINDOW * dims - 1),
            ("b1", 0),
            ("b1", dims - 1),
            ("wp", 0),
            ("wp", 4),
            ("wp", dims * dims - 1),
            ("b2", 0),
            ("b2", 1),
            ("b2", dims - 1),
            ("dw", 13),
        ];
        for (tensor, idx) in checks.drain(..) {
            let get = |m: &MicroMixer| loss(m, &window_rows, &center, &y);
            let mut mp = clone_mixer(&m);
            bump(&mut mp, tensor, idx, eps);
            let mut mm = clone_mixer(&m);
            bump(&mut mm, tensor, idx, -eps);
            let fd = (get(&mp) - get(&mm)) / (2.0 * eps);
            let an = grad_at(&grads, tensor, idx);
            assert!(
                (fd - an).abs() < 2e-3 * (1.0 + fd.abs().max(an.abs())),
                "{tensor}[{idx}]: finite-diff {fd} vs analytic {an}"
            );
        }
    }

    /// Training on a synthetic task the mixer CAN express must beat the linear
    /// baseline: teacher = normalize(center + 0.3·GELU(prev)) — nonlinear,
    /// context-dependent, inexpressible by a 5-tap linear mix.
    #[test]
    fn training_beats_linear_baseline_on_nonlinear_synthetic_task() {
        let dims = 8;
        let vocab = 16;
        let (encoder, table) = nonlinear_fake(vocab, dims); // defined below
        let corpus: Vec<String> = (0..300).map(|i| synth_doc(i, 24)).collect();
        let opts = MixerTrainOptions {
            sample_capacity: 50_000,
            epochs: 8,
            batch: 8,
            lr: 3e-3,
            holdout_frac: 0.1,
        };
        let (_m, report) = train_mixer(&encoder, &table, corpus.into_iter(), &opts).unwrap();
        assert!(
            report.holdout_cosine_mixer > report.holdout_cosine_linear + 0.01,
            "mixer {} must beat linear {} by >0.01 on a task built to require nonlinearity",
            report.holdout_cosine_mixer,
            report.holdout_cosine_linear
        );
        assert!(
            report.holdout_cosine_mixer > 0.9,
            "should nearly solve the synthetic task"
        );
    }

    #[test]
    fn training_is_deterministic() {
        let dims = 4;
        let vocab = 8;
        let (encoder, table) = nonlinear_fake(vocab, dims);
        let corpus: Vec<String> = (0..50).map(|i| synth_doc(i, 12)).collect();
        let opts = MixerTrainOptions {
            sample_capacity: 5_000,
            epochs: 2,
            batch: 8,
            lr: 1e-3,
            holdout_frac: 0.1,
        };
        let (a, _) = train_mixer(&encoder, &table, corpus.clone().into_iter(), &opts).unwrap();
        let (b, _) = train_mixer(&encoder, &table, corpus.into_iter(), &opts).unwrap();
        assert_eq!(a.dw, b.dw);
        assert_eq!(a.wp, b.wp);
    }

    #[test]
    fn empty_corpus_errors() {
        let (encoder, table) = nonlinear_fake(4, 4);
        let err = train_mixer(
            &encoder,
            &table,
            Vec::<String>::new().into_iter(),
            &MixerTrainOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().to_lowercase().contains("empty"));
    }

    // ── Test helpers ────────────────────────────────────────────────────

    /// Deterministic TEACHER for hermetic tests: token id = byte − b'a';
    /// "contextual" embedding = L2norm(onehot(id) + 0.3·GELU-elementwise(onehot(prev_id))).
    /// GELU applied to a `{0,1}`-valued vector just rescales its single `1`
    /// entry by the constant `gelu(1)` — the point of this fixture isn't
    /// GELU's curve, it's that the teacher depends on CONTEXT (the previous
    /// token), which the fixed, context-blind linear baseline built below
    /// (`mix_weights = [0,0,1,0,0]`, pure center lookup) cannot capture but
    /// the mixer's wider trainable window can.
    struct FakeEncoder {
        dims: usize,
        vocab_size: usize,
    }

    impl FakeEncoder {
        fn onehot(&self, id: u32) -> Vec<f32> {
            let mut v = vec![0.0f32; self.dims];
            v[id as usize % self.dims] = 1.0;
            v
        }

        fn embed_row(&self, id: u32, prev_id: u32) -> Vec<f32> {
            let onehot_id = self.onehot(id);
            let onehot_prev = self.onehot(prev_id);
            let mut row: Vec<f32> = onehot_id
                .iter()
                .zip(onehot_prev.iter())
                .map(|(&a, &b)| a + 0.3 * gelu(b))
                .collect();
            normalize_in_place(&mut row);
            row
        }
    }

    impl DocTokenEncoder for FakeEncoder {
        fn encode_with_ids(&self, texts: &[String]) -> Result<Vec<(Vec<u32>, TokenEmbeddings)>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let bytes = text.as_bytes();
                    let mut ids: Vec<u32> = Vec::with_capacity(bytes.len());
                    let mut emb = Array2::<f32>::zeros((bytes.len(), self.dims));
                    for (i, &b) in bytes.iter().enumerate() {
                        let id = u32::from(b - b'a');
                        let prev = if i == 0 { id } else { ids[i - 1] };
                        let row = self.embed_row(id, prev);
                        for (d, v) in row.into_iter().enumerate() {
                            emb[[i, d]] = v;
                        }
                        ids.push(id);
                    }
                    (ids, emb)
                })
                .collect())
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    /// A `FakeEncoder` (the teacher) plus a `StaticTokenTable` (the student
    /// lookup table) whose row for each id is `onehot(id)` and whose
    /// `mix_weights` are the fixed, context-blind `[0,0,1,0,0]` (pure center
    /// lookup) — the linear baseline `train_mixer`'s report compares against.
    fn nonlinear_fake(vocab: usize, dims: usize) -> (FakeEncoder, StaticTokenTable) {
        let encoder = FakeEncoder {
            dims,
            vocab_size: vocab,
        };
        let mut table = StaticTokenTable::new(vocab, dims, [0.0, 0.0, 1.0, 0.0, 0.0]);
        for id in 0..vocab as u32 {
            let row = encoder.onehot(id);
            table.set_row(id, &row);
        }
        (encoder, table)
    }

    /// Deterministic lowercase document generator via `SplitMix64`.
    fn synth_doc(seed: u64, len: usize) -> String {
        let mut rng = SplitMix64::new(seed);
        (0..len)
            .map(|_| (b'a' + rng.next_below(26) as u8) as char)
            .collect()
    }

    fn compute_gradients(
        m: &MicroMixer,
        window_rows: &[Vec<f32>],
        center: &[f32],
        y: &[f32],
    ) -> Gradients {
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|k| window_rows[k].as_slice());
        backward(m, &window, center, y).1
    }

    fn loss(m: &MicroMixer, window_rows: &[Vec<f32>], center: &[f32], y: &[f32]) -> f32 {
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|k| window_rows[k].as_slice());
        backward(m, &window, center, y).0
    }

    fn clone_mixer(m: &MicroMixer) -> MicroMixer {
        MicroMixer {
            dims: m.dims,
            dw: m.dw.clone(),
            b1: m.b1.clone(),
            wp: m.wp.clone(),
            b2: m.b2.clone(),
        }
    }

    fn bump(m: &mut MicroMixer, tensor: &str, idx: usize, delta: f32) {
        let target = match tensor {
            "dw" => &mut m.dw,
            "b1" => &mut m.b1,
            "wp" => &mut m.wp,
            "b2" => &mut m.b2,
            other => panic!("bump: unknown tensor {other}"),
        };
        target[idx] += delta;
    }

    fn grad_at(g: &Gradients, tensor: &str, idx: usize) -> f32 {
        match tensor {
            "dw" => g.dw[idx],
            "b1" => g.b1[idx],
            "wp" => g.wp[idx],
            "b2" => g.b2[idx],
            other => panic!("grad_at: unknown tensor {other}"),
        }
    }
}
