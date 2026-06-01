//! Smoke test proving `toml` is a usable direct dependency of semantex-core.
//! The model registry (`model/manifest.rs`) parses a user `models.toml`; this
//! confirms the dep resolves and the `from_str` API surface we rely on exists.

#[test]
fn toml_parses_a_table() {
    // Compiling this is half the test: an undeclared crate fails to build.
    let parsed: toml::Value = toml::from_str(
        r#"
        [[model]]
        id = "demo"
    "#,
    )
    .expect("toml must parse a simple table array");
    let models = parsed.get("model").and_then(toml::Value::as_array).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(
        models[0].get("id").and_then(toml::Value::as_str),
        Some("demo")
    );
}
