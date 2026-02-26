use crate::chunking::Chunker;
use crate::types::{Chunk, ChunkType};
use anyhow::Result;
use std::path::Path;

/// Approximate characters per token
const CHARS_PER_TOKEN: usize = 4;

pub struct TextChunker {
    /// Chunk size in approximate tokens
    chunk_size: usize,
    /// Overlap in approximate tokens
    chunk_overlap: usize,
}

impl TextChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl Chunker for TextChunker {
    fn chunk(&self, path: &Path, content: &str) -> Result<Vec<Chunk>> {
        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size_chars = self.chunk_size * CHARS_PER_TOKEN;
        let chunk_overlap_chars = self.chunk_overlap * CHARS_PER_TOKEN;

        // If the entire content fits in one chunk, return it as-is
        if content.len() <= chunk_size_chars {
            let end_line = content.chars().filter(|&c| c == '\n').count() as u32;
            return Ok(vec![Chunk {
                id: 0,
                file_path: path.to_path_buf(),
                start_line: 1,
                end_line: end_line.max(1),
                content: content.to_string(),
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            }]);
        }

        let mut chunks = Vec::new();
        let mut offset = 0usize;
        let mut window_index = 0u32;
        let bytes = content.as_bytes();

        while offset < content.len() {
            let mut end = (offset + chunk_size_chars).min(content.len());
            // Ensure end is on a UTF-8 char boundary
            if !content.is_char_boundary(end) {
                end = (0..end)
                    .rev()
                    .find(|&i| content.is_char_boundary(i))
                    .unwrap_or(offset);
            }

            // Try to break at a line boundary: search backwards from end within the overlap window
            let split_at = if end < content.len() {
                let search_start = if end > chunk_overlap_chars {
                    end - chunk_overlap_chars
                } else {
                    offset
                };
                // Find the last newline in [search_start, end)
                let mut best = end;
                for i in (search_start..end).rev() {
                    if bytes[i] == b'\n' {
                        best = i + 1; // split right after the newline
                        break;
                    }
                }
                best
            } else {
                end
            };

            // Ensure split_at is on a UTF-8 char boundary
            let split_at = if content.is_char_boundary(split_at) {
                split_at
            } else {
                // Round down to nearest char boundary
                (0..split_at)
                    .rev()
                    .find(|&i| content.is_char_boundary(i))
                    .unwrap_or(offset)
            };

            let chunk_content = &content[offset..split_at];

            if !chunk_content.trim().is_empty() {
                // Count newlines before offset to determine start_line
                let start_line =
                    content[..offset].chars().filter(|&c| c == '\n').count() as u32 + 1;
                let end_line =
                    start_line + chunk_content.chars().filter(|&c| c == '\n').count() as u32;
                let end_line = end_line.max(start_line);

                chunks.push(Chunk {
                    id: 0,
                    file_path: path.to_path_buf(),
                    start_line,
                    end_line,
                    content: chunk_content.to_string(),
                    chunk_type: ChunkType::TextWindow { window_index },
                });
                window_index += 1;
            }

            // Advance by (chunk_size - overlap) chars, but at least 1 char to avoid infinite loop
            let step = if chunk_size_chars > chunk_overlap_chars {
                chunk_size_chars - chunk_overlap_chars
            } else {
                chunk_size_chars
            };
            let new_offset = offset + step.max(1);
            // If we already consumed past new_offset due to line-boundary snapping,
            // use split_at minus overlap as the next start
            if split_at > new_offset {
                offset = if split_at > chunk_overlap_chars {
                    split_at - chunk_overlap_chars
                } else {
                    split_at
                };
            } else {
                offset = new_offset;
            }

            // Ensure forward progress
            if offset <= chunks.last().map_or(0, |_| offset.saturating_sub(1)) {
                offset = split_at;
            }

            // Ensure offset is on a UTF-8 char boundary
            if !content.is_char_boundary(offset) {
                offset = (offset..content.len())
                    .find(|&i| content.is_char_boundary(i))
                    .unwrap_or(content.len());
            }
        }

        Ok(chunks)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_small_content_single_chunk() {
        let chunker = TextChunker::new(256, 64);
        let chunks = chunker
            .chunk(Path::new("test.txt"), "Hello, world!\n")
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].content, "Hello, world!\n");
    }

    #[test]
    fn test_empty_content() {
        let chunker = TextChunker::new(256, 64);
        let chunks = chunker.chunk(Path::new("test.txt"), "").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let chunker = TextChunker::new(256, 64);
        let chunks = chunker.chunk(Path::new("test.txt"), "   \n  \n  ").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_multiple_chunks() {
        // chunk_size=10 tokens = 40 chars, overlap=2 tokens = 8 chars
        let chunker = TextChunker::new(10, 2);
        let content = "a".repeat(100);
        let chunks = chunker.chunk(Path::new("test.txt"), &content).unwrap();
        assert!(chunks.len() > 1);
        // All chunks should have window indices
        for (i, chunk) in chunks.iter().enumerate() {
            match chunk.chunk_type {
                ChunkType::TextWindow { window_index } => {
                    assert_eq!(window_index, i as u32);
                }
                _ => panic!("Expected TextWindow chunk type"),
            }
        }
    }
}
