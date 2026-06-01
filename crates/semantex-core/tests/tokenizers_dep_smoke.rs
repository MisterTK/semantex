//! Smoke test proving `tokenizers` is a usable direct dependency of
//! semantex-core (it is otherwise pulled transitively via fastembed). The ONNX
//! reranker (search/onnx_reranker.rs) needs `Tokenizer::from_file`.

#[test]
fn tokenizer_type_is_importable() {
    // Compiling this reference is the test: if `tokenizers` is not a direct
    // dependency, this fails to build with "unresolved import".
    fn _accepts(_t: &tokenizers::Tokenizer) {}
    // Also assert the encode/ids API surface we rely on exists by referencing it.
    fn _from_file_exists(
        p: &std::path::Path,
    ) -> Result<tokenizers::Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
        tokenizers::Tokenizer::from_file(p)
    }
    let _ = _accepts;
    let _ = _from_file_exists;
}
