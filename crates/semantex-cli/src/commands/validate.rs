use anyhow::Result;
use std::path::Path;

pub fn run(project_path: &Path) -> Result<()> {
    let report = semantex_core::index::validate::validate(project_path)?;

    for check in &report.checks {
        let icon = if check.passed { "PASS" } else { "FAIL" };
        println!("[{icon}] {}: {}", check.name, check.message);
    }

    println!("\n{}", report.summary());

    if !report.all_passed() {
        std::process::exit(1);
    }
    Ok(())
}
