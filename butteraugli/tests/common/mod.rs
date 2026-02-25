//! Common test utilities for butteraugli tests.
//!
//! Provides path resolution for testdata files without hardcoded paths.
//!
//! ## Environment Variables
//! - `JPEGLI_TESTDATA`: Path to jpegli testdata directory
//! - `CJPEGLI_PATH`: Path to cjpegli binary
//!
//! ## Panics
//! Functions panic with helpful messages if resources aren't found.
//! Use `try_*` variants for optional resources.

pub mod generators;

use std::path::PathBuf;

/// Get path to the jpegli testdata directory.
///
/// # Panics
/// Panics if directory cannot be found. Set `JPEGLI_TESTDATA` env var.
#[track_caller]
pub fn get_testdata_dir() -> PathBuf {
    try_get_testdata_dir().unwrap_or_else(|| {
        panic!(
            "Jpegli testdata directory not found.\n\
             Set JPEGLI_TESTDATA environment variable to the testdata directory.\n\
             Expected structure: $JPEGLI_TESTDATA/jxl/flower/flower_small.rgb.png"
        )
    })
}

/// Try to get path to the jpegli testdata directory. Returns None if not found.
pub fn try_get_testdata_dir() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(dir) = std::env::var("JPEGLI_TESTDATA") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            // Sibling jpegli project
            PathBuf::from(&manifest).join("../jpegli/testdata"),
            PathBuf::from(&manifest).join("../jpegli-rs/internal/jpegli-cpp/testdata"),
            // Local testdata
            PathBuf::from(&manifest).join("testdata"),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Get path to the flower_small test image.
///
/// # Panics
/// Panics if image cannot be found.
#[track_caller]
pub fn get_flower_small_path() -> PathBuf {
    try_get_flower_small_path().unwrap_or_else(|| {
        panic!(
            "Test image flower_small.rgb.png not found.\n\
             Set JPEGLI_TESTDATA environment variable or ensure testdata is available."
        )
    })
}

/// Try to get path to the flower_small test image.
pub fn try_get_flower_small_path() -> Option<PathBuf> {
    let path = try_get_testdata_dir()?.join("jxl/flower/flower_small.rgb.png");
    if path.exists() { Some(path) } else { None }
}

/// Find cjpegli binary.
///
/// # Panics
/// Panics if binary cannot be found.
#[track_caller]
pub fn get_cjpegli_path() -> PathBuf {
    find_cjpegli().unwrap_or_else(|| {
        panic!(
            "cjpegli binary not found.\n\
             Set CJPEGLI_PATH environment variable or build jpegli:\n\
             cd ../jpegli && cmake -B build && cmake --build build"
        )
    })
}

/// Find cjpegli binary. Returns None if not found.
pub fn find_cjpegli() -> Option<PathBuf> {
    // Check environment variable
    if let Ok(path) = std::env::var("CJPEGLI_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            PathBuf::from(&manifest).join("../jpegli/build/tools/cjpegli"),
            PathBuf::from(&manifest).join("../jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli"),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Macro to skip test if optional resource is missing (for truly optional tests).
/// Use sparingly - prefer panicking for missing required resources.
#[macro_export]
macro_rules! skip_if_unavailable {
    ($opt:expr, $msg:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!("SKIPPING TEST: {}", $msg);
                return;
            }
        }
    };
}
