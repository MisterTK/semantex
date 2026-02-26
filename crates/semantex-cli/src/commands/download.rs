use anyhow::Result;
use colored::Colorize;
use semantex_core::config::SemantexConfig;
use semantex_core::embedding::model_manager;

pub fn run(config: &SemantexConfig) -> Result<()> {
    let models_dir = config.models_dir();
    println!(
        "{} models to {}",
        "Downloading".green().bold(),
        models_dir.display()
    );

    model_manager::ensure_colbert_model(&models_dir)?;

    println!("{}", "All models downloaded successfully!".green().bold());
    Ok(())
}
