//! Build C++ libjxl butteraugli into a static lib for FFI benchmarking.
//!
//! Layout:
//!   - libjxl source: `experiments/libjxl-vendor/` (auto-cloned, gitignored)
//!   - build dir:     `experiments/libjxl-vendor/build/`
//!
//! Override with `LIBJXL_DIR` (and optionally `LIBJXL_BUILD_DIR`) to use an
//! existing checkout. We deliberately avoid the user's working `~/work/jxl-efforts/libjxl`
//! by default — it may have unrelated uncommitted changes.
//!
//! Fairness knobs:
//!   - `-march=x86-64-v2` — pins the compile-time baseline to SSE4 (x86-64-v2)
//!     so we're not sneaking in -march=native.
//!   - `JPEGXL_HWY_TARGETS_OFF_BY_DEFAULT` reduced to leave AVX2 + AVX3 + AVX3_ZEN4
//!     in the runtime-dispatch set. AVX3_SPR stays off (overkill, asymmetric vs
//!     archmage's default v4 tier).
//!
//! Skips silently (no FFI cfg) if the build fails — you still get the Rust-only bench.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=ffi/");
    println!("cargo:rerun-if-env-changed=LIBJXL_DIR");
    println!("cargo:rerun-if-env-changed=LIBJXL_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=BUTTERAUGLI_BENCH_NO_CPP");

    if std::env::var_os("BUTTERAUGLI_BENCH_NO_CPP").is_some() {
        println!("cargo:warning=BUTTERAUGLI_BENCH_NO_CPP set — skipping C++ butteraugli FFI build");
        return;
    }

    if let Err(msg) = try_build_cpp_ffi() {
        println!("cargo:warning=Skipping C++ butteraugli FFI: {msg}");
    }
}

fn try_build_cpp_ffi() -> Result<(), String> {
    let (libjxl_dir, build_dir) = locate_libjxl()?;

    if !build_dir.join("lib/libjxl-internal.a").exists() {
        return Err(format!(
            "libjxl-internal.a not found in {}",
            build_dir.display()
        ));
    }

    compile_ffi_shim(&libjxl_dir, &build_dir);
    link_libjxl(&build_dir)?;

    println!("cargo:rustc-cfg=has_cpp_butteraugli");
    Ok(())
}

fn locate_libjxl() -> Result<(PathBuf, PathBuf), String> {
    if let Ok(dir) = std::env::var("LIBJXL_DIR") {
        let dir = PathBuf::from(dir);
        if dir.exists() {
            let build = std::env::var("LIBJXL_BUILD_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| dir.join("build"));
            if !build.join("lib/libjxl-internal.a").exists() {
                cmake_build(&dir, &build)?;
            }
            return Ok((dir, build));
        }
    }

    // Default: vendored clone under workspace experiments/
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .ok_or("manifest dir has no parent")?
        .to_path_buf();
    let vendor_dir = workspace_root.join("experiments/libjxl-vendor");
    let build_dir = vendor_dir.join("build");

    if !vendor_dir.join("CMakeLists.txt").exists() {
        clone_libjxl(&vendor_dir)?;
    }

    if !build_dir.join("lib/libjxl-internal.a").exists() {
        cmake_build(&vendor_dir, &build_dir)?;
    }

    Ok((vendor_dir, build_dir))
}

fn clone_libjxl(target: &Path) -> Result<(), String> {
    eprintln!("cloning libjxl into {}", target.display());

    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir parent: {e}"))?;
    }

    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "https://github.com/libjxl/libjxl.git",
        ])
        .arg(target)
        .status()
        .map_err(|e| format!("git clone failed: {e}"))?;

    if !status.success() {
        return Err("git clone libjxl failed".into());
    }

    let status = Command::new("git")
        .args([
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "third_party/highway",
            "third_party/skcms",
            "third_party/brotli",
        ])
        .current_dir(target)
        .status()
        .map_err(|e| format!("git submodule update failed: {e}"))?;

    if !status.success() {
        return Err("git submodule update failed".into());
    }

    Ok(())
}

fn cmake_build(source_dir: &Path, build_dir: &Path) -> Result<(), String> {
    eprintln!(
        "building libjxl: {} -> {}",
        source_dir.display(),
        build_dir.display()
    );

    std::fs::create_dir_all(build_dir).map_err(|e| format!("mkdir failed: {e}"))?;

    // Pin to x86-64-v2 + enable AVX3 / AVX3_ZEN4 in highway runtime dispatch.
    // SVE / RVV / SSSE3 stay off by default (irrelevant on x86-64 modern CPUs).
    // AVX3_SPR stays off (Sapphire Rapids only, asymmetric vs archmage v4).
    let baseline_flags = if cfg!(target_arch = "x86_64") {
        "-march=x86-64-v2 -O3"
    } else {
        "-O3"
    };

    let status = Command::new("cmake")
        .arg(format!("-S{}", source_dir.display()))
        .arg(format!("-B{}", build_dir.display()))
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            &format!("-DCMAKE_C_FLAGS={baseline_flags}"),
            &format!("-DCMAKE_CXX_FLAGS={baseline_flags}"),
            "-DJPEGXL_HWY_TARGETS_OFF_BY_DEFAULT=AVX3_SPR;SVE;SVE2;SVE_256;SVE2_128;SSSE3;RVV",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DJPEGXL_STATIC=ON",
            "-DJPEGXL_ENABLE_TOOLS=ON",
            "-DJPEGXL_ENABLE_DOXYGEN=OFF",
            "-DJPEGXL_ENABLE_MANPAGES=OFF",
            "-DJPEGXL_ENABLE_BENCHMARK=OFF",
            "-DJPEGXL_ENABLE_EXAMPLES=OFF",
            "-DJPEGXL_ENABLE_JNI=OFF",
            "-DJPEGXL_ENABLE_SJPEG=OFF",
            "-DJPEGXL_ENABLE_OPENEXR=OFF",
            "-DJPEGXL_ENABLE_JPEGLI=OFF",
            "-DJPEGXL_ENABLE_TCMALLOC=OFF",
            "-DJPEGXL_BUNDLE_LIBPNG=OFF",
            "-DBUILD_TESTING=OFF",
        ])
        .status()
        .map_err(|e| format!("cmake configure failed: {e}"))?;

    if !status.success() {
        return Err("cmake configure failed".into());
    }

    let nproc = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".into());

    let status = Command::new("cmake")
        .arg("--build")
        .arg(build_dir)
        .args(["--parallel", &nproc])
        .status()
        .map_err(|e| format!("cmake build failed: {e}"))?;

    if !status.success() {
        return Err("cmake build failed".into());
    }

    Ok(())
}

fn compile_ffi_shim(libjxl_dir: &Path, build_dir: &Path) {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("ffi/butteraugli_ffi.cpp")
        .include(libjxl_dir)
        .include(libjxl_dir.join("lib/include"))
        .include(build_dir.join("lib/include"))
        .include("ffi")
        .opt_level(3)
        .compile("butteraugli_ffi");
}

fn link_libjxl(build_dir: &Path) -> Result<(), String> {
    for subdir in ["lib", "tools", "third_party/highway", "third_party/brotli"] {
        let p = build_dir.join(subdir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }

    println!("cargo:rustc-link-lib=static=jxl-internal");
    println!("cargo:rustc-link-lib=static=jxl_tool");
    println!("cargo:rustc-link-lib=static=jxl_gauss_blur");
    println!("cargo:rustc-link-lib=static=hwy");

    if build_dir.join("lib/libjxl_cms.a").exists() {
        println!("cargo:rustc-link-lib=static=jxl_cms");
    } else {
        println!("cargo:rustc-link-lib=dylib=jxl_cms");
    }

    for lib in ["brotlienc", "brotlidec", "brotlicommon"] {
        let static_name = format!("lib{lib}-static.a");
        if build_dir
            .join("third_party/brotli")
            .join(&static_name)
            .exists()
        {
            println!("cargo:rustc-link-lib=static={lib}-static");
        } else if build_dir.join("lib").join(format!("lib{lib}.a")).exists() {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    Ok(())
}
