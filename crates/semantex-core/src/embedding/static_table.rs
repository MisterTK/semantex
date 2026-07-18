use std::path::Path;

/// A static token table storing embeddings in f16 format on disk, f32 in memory.
///
/// This is an encoder-free lookup table where embeddings are pre-computed and stored
/// as floating-point vectors indexed by token ID. Designed for fast inference without
/// neural network evaluation.
#[derive(Debug, Clone)]
pub struct StaticTokenTable {
    /// Dimensionality of each embedding vector.
    pub dims: usize,

    /// Mixture weights (5 components).
    pub mix_weights: [f32; 5],

    /// Storage: vocab_size rows, dims columns of f32 values.
    /// Row i is for token_id i.
    data: Vec<Vec<f32>>,
}

impl StaticTokenTable {
    /// Create a new static token table.
    ///
    /// All rows are initially zero-filled (all-zero rows report as `None` on lookup).
    pub fn new(vocab_size: usize, dims: usize, mix_weights: [f32; 5]) -> Self {
        StaticTokenTable {
            dims,
            mix_weights,
            data: vec![vec![0.0; dims]; vocab_size],
        }
    }

    /// Set the embedding row for a token.
    ///
    /// # Panics
    ///
    /// Panics if the row length does not match `self.dims`.
    pub fn set_row(&mut self, token_id: u32, row: &[f32]) {
        let idx = token_id as usize;
        assert_eq!(
            row.len(),
            self.dims,
            "row length {} does not match dims {}",
            row.len(),
            self.dims
        );
        #[allow(clippy::manual_assert)]
        if idx >= self.data.len() {
            panic!(
                "token_id {token_id} out of bounds (vocab_size {})",
                self.data.len()
            );
        }
        self.data[idx].copy_from_slice(row);
    }

    /// Look up the embedding for a token.
    ///
    /// Returns `None` if the token is out-of-vocabulary or if the row is all zeros
    /// (indicating a never-seen token).
    pub fn lookup(&self, token_id: u32) -> Option<&[f32]> {
        let idx = token_id as usize;
        if idx >= self.data.len() {
            return None;
        }
        let row = &self.data[idx];
        // Check if row is all zeros
        if row.iter().all(|&v| v == 0.0) {
            return None;
        }
        Some(row)
    }

    /// Save the table to a binary file.
    ///
    /// Binary format (little-endian):
    /// - Magic: "SXST" (4 bytes)
    /// - Version: u32 = 1
    /// - Vocab size: u32
    /// - Dims: u32
    /// - Mix weights: [f32; 5] (20 bytes)
    /// - Data: vocab_size × dims f16 values
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        // Write magic
        file.write_all(b"SXST")?;

        // Write version
        file.write_all(&1u32.to_le_bytes())?;

        // Write vocab_size and dims
        let vocab_size = self.data.len() as u32;
        let dims = self.dims as u32;
        file.write_all(&vocab_size.to_le_bytes())?;
        file.write_all(&dims.to_le_bytes())?;

        // Write mix_weights
        for &weight in &self.mix_weights {
            file.write_all(&weight.to_le_bytes())?;
        }

        // Write data as f16
        for row in &self.data {
            for &val in row {
                let half = half::f16::from_f32(val);
                file.write_all(&half.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load a table from a binary file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut cursor = 0;

        // Read magic
        if cursor + 4 > buf.len() {
            anyhow::bail!("file too short for magic");
        }
        let magic = &buf[cursor..cursor + 4];
        if magic != b"SXST" {
            anyhow::bail!("invalid magic: expected SXST, got {magic:?}");
        }
        cursor += 4;

        // Read version
        if cursor + 4 > buf.len() {
            anyhow::bail!("file too short for version");
        }
        let version = u32::from_le_bytes([
            buf[cursor],
            buf[cursor + 1],
            buf[cursor + 2],
            buf[cursor + 3],
        ]);
        if version != 1 {
            anyhow::bail!("unsupported version: {version}");
        }
        cursor += 4;

        // Read vocab_size and dims
        if cursor + 8 > buf.len() {
            anyhow::bail!("file too short for vocab_size and dims");
        }
        let vocab_size = u32::from_le_bytes([
            buf[cursor],
            buf[cursor + 1],
            buf[cursor + 2],
            buf[cursor + 3],
        ]) as usize;
        cursor += 4;
        let dims = u32::from_le_bytes([
            buf[cursor],
            buf[cursor + 1],
            buf[cursor + 2],
            buf[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Read mix_weights
        if cursor + 20 > buf.len() {
            anyhow::bail!("file too short for mix_weights");
        }
        let mut mix_weights = [0.0f32; 5];
        for weight in &mut mix_weights {
            let bytes = &buf[cursor..cursor + 4];
            *weight = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            cursor += 4;
        }

        // Read data — all size arithmetic checked. A forged header can put
        // arbitrary u32s in vocab_size/dims; unchecked `vocab_size * dims * 2`
        // can wrap (32-bit usize, or u32::MAX² on 64-bit), letting the bounds
        // check pass and the row loop read past `buf`. Additionally reject any
        // claimed data size larger than the actual file remainder BEFORE
        // allocating vocab_size rows, so a 40-byte forged file can never
        // request a multi-GB allocation.
        let expected_data_size = vocab_size
            .checked_mul(dims)
            .and_then(|n| n.checked_mul(2)) // f16 is 2 bytes
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "implausibly large table header: vocab_size {vocab_size} × dims {dims} overflows"
                )
            })?;
        let remaining = buf.len() - cursor; // cursor ≤ buf.len() holds here
        if expected_data_size > remaining {
            anyhow::bail!(
                "file too short for data: header claims {expected_data_size} bytes, \
                 {remaining} remain"
            );
        }
        let mut data = vec![vec![0.0f32; dims]; vocab_size];
        for row in &mut data {
            for col in row {
                let bytes = &buf[cursor..cursor + 2];
                let half_val = half::f16::from_le_bytes([bytes[0], bytes[1]]);
                *col = half_val.to_f32();
                cursor += 2;
            }
        }

        Ok(StaticTokenTable {
            dims,
            mix_weights,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_save_load() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("table.bin");
        let mut t = StaticTokenTable::new(10, 4, [0.1, 0.2, 0.4, 0.2, 0.1]);
        t.set_row(3, &[1.0, 0.0, -1.0, 0.5]);
        t.save(&path).unwrap();
        let loaded = StaticTokenTable::load(&path).unwrap();
        assert_eq!(loaded.dims, 4);
        assert_eq!(loaded.mix_weights, [0.1, 0.2, 0.4, 0.2, 0.1]);
        let row = loaded.lookup(3).unwrap();
        // f16 round-trip: exact for these values.
        assert_eq!(row, &[1.0, 0.0, -1.0, 0.5]);
        assert!(loaded.lookup(4).is_none(), "all-zero row reads as None");
        assert!(loaded.lookup(999).is_none(), "out of vocab reads as None");
    }

    #[test]
    fn load_rejects_bad_magic() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("bad.bin");
        std::fs::write(&path, b"NOPE1234").unwrap();
        let err = StaticTokenTable::load(&path).unwrap_err();
        assert!(err.to_string().contains("magic"), "got: {err}");
    }

    /// Build a syntactically valid SXST header with attacker-chosen
    /// vocab_size/dims and no (or tiny) data section.
    fn forged_header(vocab_size: u32, dims: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SXST");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&vocab_size.to_le_bytes());
        buf.extend_from_slice(&dims.to_le_bytes());
        for _ in 0..5 {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        buf
    }

    #[test]
    fn load_rejects_overflowing_vocab_times_dims() {
        // vocab_size * dims * 2 wraps a 32-bit usize and, at u32::MAX * u32::MAX,
        // even exceeds u64 when multiplied out naively — the checked path must
        // reject it as an error, not pass the bounds check.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("forged.bin");
        std::fs::write(&path, forged_header(u32::MAX, u32::MAX)).unwrap();
        let err = StaticTokenTable::load(&path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("implausibly large") || msg.contains("overflow"),
            "got: {msg}"
        );
    }

    #[test]
    fn load_rejects_huge_but_non_overflowing_header() {
        // 100M vocab × 65k dims doesn't wrap on 64-bit but is an implausible
        // (multi-TB) allocation request from a 40-byte file — must error out
        // BEFORE any allocation is attempted.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("huge.bin");
        std::fs::write(&path, forged_header(100_000_000, 65_536)).unwrap();
        let err = StaticTokenTable::load(&path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("implausibly large") || msg.contains("file too short"),
            "got: {msg}"
        );
    }

    #[test]
    fn load_rejects_truncated_data_section() {
        // Plausible header (10 × 4) but zero data bytes → clean error.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("trunc.bin");
        std::fs::write(&path, forged_header(10, 4)).unwrap();
        let err = StaticTokenTable::load(&path).unwrap_err();
        assert!(err.to_string().contains("file too short"), "got: {err}");
    }
}
