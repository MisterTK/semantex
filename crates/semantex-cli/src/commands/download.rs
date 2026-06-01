use anyhow::Result;
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::embedding::{runtime_manager, single_vector_model};

pub fn run(config: &SemantexConfig) -> Result<()> {
    let models_dir = config.models_dir();
    println!(
        "{} models to {}",
        "Downloading".green().bold(),
        models_dir.display()
    );

    single_vector_model::ensure_coderank_model(&models_dir)?;

    // Provision the ONNX Runtime shared library too: `ort` runs in load-dynamic
    // mode, so the runtime is fetched at first use rather than linked at build
    // time. Pre-fetching it here makes `download-models` a complete offline-prep
    // step. Respect an explicit ORT_DYLIB_PATH (the user supplied their own lib).
    if std::env::var_os("ORT_DYLIB_PATH").is_none_or(|v| v.is_empty()) {
        let runtime_root = SemantexConfig::semantex_home().join("runtime");
        let lib = runtime_manager::ensure_onnxruntime(&runtime_root)?;
        println!(
            "{} ONNX Runtime {} at {}",
            "Provisioned".green().bold(),
            runtime_manager::ONNXRUNTIME_VERSION,
            lib.display()
        );
    }

    println!("{}", "All models downloaded successfully!".green().bold());
    Ok(())
}
