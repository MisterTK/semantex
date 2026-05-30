use crate::chunking::Chunker;
use crate::types::{Chunk, ChunkType};
use anyhow::Result;
use std::path::Path;

/// Approximate characters per token
const CHARS_PER_TOKEN: usize = 4;

pub struct PdfChunker {
    /// Chunk size in approximate tokens
    chunk_size: usize,
    /// Overlap in approximate tokens
    chunk_overlap: usize,
}

impl PdfChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl Chunker for PdfChunker {
    fn chunk(&self, path: &Path, _content: &str) -> Result<Vec<Chunk>> {
        // Extract text from PDF; on ANY failure return an empty vec so one bad
        // PDF never blocks indexing the rest of the repo. `pdf_extract` not only
        // returns `Err` for some malformed files, it also `unwrap()`s internally
        // and *panics* on others (e.g. "beginbfrange exected hexstring"). With
        // `panic = "unwind"` (see workspace Cargo.toml) we catch that panic here
        // and treat it the same as a normal extraction error.
        let extracted =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pdf_extract::extract_text(path)));
        let text = match extracted {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                tracing::warn!("Failed to extract text from PDF {}: {}", path.display(), e);
                return Ok(Vec::new());
            }
            Err(_) => {
                tracing::warn!(
                    "PDF text extraction panicked for {} — skipping file",
                    path.display()
                );
                return Ok(Vec::new());
            }
        };

        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size_chars = self.chunk_size * CHARS_PER_TOKEN;
        let chunk_overlap_chars = self.chunk_overlap * CHARS_PER_TOKEN;

        // If the entire text fits in one chunk, return it as-is
        if text.len() <= chunk_size_chars {
            return Ok(vec![Chunk {
                id: 0,
                file_path: path.to_path_buf(),
                start_line: 1,
                end_line: 1,
                content: text,
                chunk_type: ChunkType::PdfPage { page_number: 0 },
            }]);
        }

        let mut chunks = Vec::new();
        let mut offset = 0usize;
        let mut window_index = 0u32;
        let bytes = text.as_bytes();

        while offset < text.len() {
            let end = (offset + chunk_size_chars).min(text.len());

            // Try to break at a line boundary
            let split_at = if end < text.len() {
                let search_start = if end > chunk_overlap_chars {
                    end - chunk_overlap_chars
                } else {
                    offset
                };
                let mut best = end;
                for i in (search_start..end).rev() {
                    if bytes[i] == b'\n' {
                        best = i + 1;
                        break;
                    }
                }
                best
            } else {
                end
            };

            let chunk_content = &text[offset..split_at];

            if !chunk_content.trim().is_empty() {
                chunks.push(Chunk {
                    id: 0,
                    file_path: path.to_path_buf(),
                    start_line: 1,
                    end_line: 1,
                    content: chunk_content.to_string(),
                    chunk_type: ChunkType::PdfPage {
                        page_number: window_index,
                    },
                });
                window_index += 1;
            }

            // Advance by (chunk_size - overlap) chars
            let step = if chunk_size_chars > chunk_overlap_chars {
                chunk_size_chars - chunk_overlap_chars
            } else {
                chunk_size_chars
            };
            let new_offset = offset + step.max(1);
            if split_at > new_offset {
                offset = if split_at > chunk_overlap_chars {
                    split_at - chunk_overlap_chars
                } else {
                    split_at
                };
            } else {
                offset = new_offset;
            }
        }

        Ok(chunks)
    }
}
