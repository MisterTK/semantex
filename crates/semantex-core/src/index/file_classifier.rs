use serde::{Deserialize, Serialize};
use std::path::Path;

/// Classifies a file's role in the codebase based on path patterns and imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileRole {
    /// Service layer (business logic)
    Service,
    /// Controller / handler / route
    Controller,
    /// Data model / entity / schema / DTO
    Model,
    /// Data access layer (repository / DAO)
    Repository,
    /// Utility / helper / common lib
    Utility,
    /// Configuration / settings
    Config,
    /// Test / spec
    Test,
    /// Database migration
    Migration,
    /// Middleware / interceptor / guard
    Middleware,
    /// Documentation / markdown / legal / prose
    Documentation,
    /// Unclassified
    Unknown,
}

impl FileRole {
    /// Score boost multiplier for Semantic queries.
    /// Identifier/Keyword queries are unaffected (return 1.0).
    pub fn semantic_boost(self) -> f32 {
        match self {
            Self::Service | Self::Controller => 1.15,
            Self::Middleware => 1.10,
            Self::Repository => 1.05,
            Self::Utility | Self::Unknown => 1.00,
            Self::Config => 0.90,
            Self::Model => 0.85,
            Self::Migration => 0.80,
            Self::Documentation => 0.75,
            Self::Test => 0.70,
        }
    }

    /// Convert to a string representation for SQLite storage.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Service => "service",
            Self::Controller => "controller",
            Self::Model => "model",
            Self::Repository => "repository",
            Self::Utility => "utility",
            Self::Config => "config",
            Self::Test => "test",
            Self::Migration => "migration",
            Self::Middleware => "middleware",
            Self::Documentation => "documentation",
            Self::Unknown => "unknown",
        }
    }

    /// Parse from a string representation (as stored in SQLite).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "service" => Self::Service,
            "controller" => Self::Controller,
            "model" => Self::Model,
            "repository" => Self::Repository,
            "utility" => Self::Utility,
            "config" => Self::Config,
            "test" => Self::Test,
            "migration" => Self::Migration,
            "middleware" => Self::Middleware,
            "documentation" => Self::Documentation,
            _ => Self::Unknown,
        }
    }
}

/// Path-based heuristics for file role classification.
/// Each entry maps a set of path substrings to a `FileRole`.
const PATH_PATTERNS: &[(&[&str], FileRole)] = &[
    (&["services/", "service/", "/svc/"], FileRole::Service),
    (
        &[
            "controllers/",
            "controller/",
            "handlers/",
            "handler/",
            "routes/",
            "router/",
        ],
        FileRole::Controller,
    ),
    (
        &[
            "models/",
            "model/",
            "entities/",
            "entity/",
            "schemas/",
            "schema/",
            "dto/",
        ],
        FileRole::Model,
    ),
    (
        &[
            "repositories/",
            "repository/",
            "repos/",
            "repo/",
            "dal/",
            "dao/",
        ],
        FileRole::Repository,
    ),
    (
        &["utils/", "util/", "helpers/", "helper/", "lib/", "common/"],
        FileRole::Utility,
    ),
    (
        &["config/", "configuration/", "settings/"],
        FileRole::Config,
    ),
    (
        &[
            "test/",
            "tests/",
            "__tests__/",
            "spec/",
            "_test.",
            ".test.",
            ".spec.",
        ],
        FileRole::Test,
    ),
    (
        &["migrations/", "migration/", "migrate/", "db/migrate/"],
        FileRole::Migration,
    ),
    (
        &[
            "middleware/",
            "interceptor/",
            "interceptors/",
            "guard/",
            "guards/",
        ],
        FileRole::Middleware,
    ),
    (
        &[
            "docs/",
            "doc/",
            "documentation/",
            "legal/",
            ".md",
            "changelog",
            "license",
            "readme",
        ],
        FileRole::Documentation,
    ),
];

/// Classify a file by its relative path.
pub fn classify_file(relative_path: &Path) -> FileRole {
    let path_str = relative_path.to_string_lossy().to_lowercase();

    for (patterns, role) in PATH_PATTERNS {
        for pattern in *patterns {
            if path_str.contains(pattern) {
                return *role;
            }
        }
    }

    FileRole::Unknown
}

/// Enhanced classification using import analysis.
/// Called during indexing when AST-level import information is available.
/// Falls back to path-based classification first; uses imports only when
/// the path is ambiguous (`Unknown`).
pub fn classify_with_imports(relative_path: &Path, imports: &[String]) -> FileRole {
    let path_role = classify_file(relative_path);
    if path_role != FileRole::Unknown {
        return path_role;
    }

    // Import-based heuristics (fallback when path is ambiguous)
    let import_str = imports.join(" ").to_lowercase();
    if import_str.contains("orm")
        || import_str.contains("prisma")
        || import_str.contains("typeorm")
        || import_str.contains("sqlalchemy")
        || import_str.contains("diesel")
    {
        return FileRole::Repository;
    }
    if import_str.contains("express")
        || import_str.contains("fastapi")
        || import_str.contains("actix")
        || import_str.contains("axum")
        || import_str.contains("gin")
    {
        return FileRole::Controller;
    }
    if import_str.contains("jest")
        || import_str.contains("pytest")
        || import_str.contains("mocha")
        || import_str.contains("#[cfg(test)]")
    {
        return FileRole::Test;
    }

    FileRole::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn classify_service_path() {
        assert_eq!(
            classify_file(&PathBuf::from("src/services/auth.ts")),
            FileRole::Service
        );
    }

    #[test]
    fn classify_test_path() {
        assert_eq!(
            classify_file(&PathBuf::from("test/auth.test.ts")),
            FileRole::Test
        );
    }

    #[test]
    fn classify_unknown_path() {
        assert_eq!(
            classify_file(&PathBuf::from("src/index.ts")),
            FileRole::Unknown
        );
    }

    #[test]
    fn classify_with_prisma_import() {
        assert_eq!(
            classify_with_imports(
                &PathBuf::from("src/data.ts"),
                &["import { PrismaClient }".to_string()]
            ),
            FileRole::Repository
        );
    }

    #[test]
    fn semantic_boost_values() {
        assert!(FileRole::Service.semantic_boost() > FileRole::Test.semantic_boost());
        assert!((FileRole::Unknown.semantic_boost() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn roundtrip_str_conversion() {
        let roles = [
            FileRole::Service,
            FileRole::Controller,
            FileRole::Model,
            FileRole::Repository,
            FileRole::Utility,
            FileRole::Config,
            FileRole::Test,
            FileRole::Migration,
            FileRole::Middleware,
            FileRole::Documentation,
            FileRole::Unknown,
        ];
        for role in roles {
            assert_eq!(FileRole::from_str(role.as_str()), role);
        }
    }

    #[test]
    fn classify_documentation_paths() {
        assert_eq!(
            classify_file(&PathBuf::from("docs/architecture.md")),
            FileRole::Documentation
        );
        assert_eq!(
            classify_file(&PathBuf::from("README.md")),
            FileRole::Documentation
        );
        assert_eq!(
            classify_file(&PathBuf::from("assets/legal/terms.md")),
            FileRole::Documentation
        );
        assert_eq!(
            classify_file(&PathBuf::from("CHANGELOG.md")),
            FileRole::Documentation
        );
    }

    #[test]
    fn documentation_boost_less_than_service() {
        assert!(FileRole::Documentation.semantic_boost() < FileRole::Service.semantic_boost());
        assert!(FileRole::Documentation.semantic_boost() < FileRole::Unknown.semantic_boost());
        assert!(FileRole::Documentation.semantic_boost() > FileRole::Test.semantic_boost());
    }
}
