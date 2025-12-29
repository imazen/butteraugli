//! Integration tests for butteraugli CLI.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Get path to the butteraugli binary.
fn butteraugli_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // Go up from butteraugli-cli to workspace root
    path.push("target");
    path.push(if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    });
    path.push(if cfg!(windows) {
        "butteraugli.exe"
    } else {
        "butteraugli"
    });
    path
}

/// Create a simple PNG file with solid color.
fn create_solid_png(path: &std::path::Path, r: u8, g: u8, b: u8) {
    // Create a minimal 16x16 RGB PNG
    let width = 16u32;
    let height = 16u32;

    let mut data = Vec::new();

    // PNG signature
    data.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR chunk
    let ihdr_data = [
        (width >> 24) as u8,
        (width >> 16) as u8,
        (width >> 8) as u8,
        width as u8,
        (height >> 24) as u8,
        (height >> 16) as u8,
        (height >> 8) as u8,
        height as u8,
        8, // bit depth
        2, // color type (RGB)
        0, // compression
        0, // filter
        0, // interlace
    ];
    write_png_chunk(&mut data, b"IHDR", &ihdr_data);

    // IDAT chunk - uncompressed image data
    // For simplicity, use zlib with no compression
    let row_size = 1 + width as usize * 3; // filter byte + RGB
    let mut raw_data = Vec::with_capacity(height as usize * row_size);
    for _ in 0..height {
        raw_data.push(0); // filter type: none
        raw_data.extend(std::iter::repeat([r, g, b]).take(width as usize).flatten());
    }

    // Compress with zlib (deflate)
    let compressed = miniz_compress(&raw_data);
    write_png_chunk(&mut data, b"IDAT", &compressed);

    // IEND chunk
    write_png_chunk(&mut data, b"IEND", &[]);

    fs::write(path, data).expect("Failed to write PNG");
}

fn write_png_chunk(data: &mut Vec<u8>, chunk_type: &[u8; 4], chunk_data: &[u8]) {
    let len = chunk_data.len() as u32;
    data.extend_from_slice(&len.to_be_bytes());
    data.extend_from_slice(chunk_type);
    data.extend_from_slice(chunk_data);

    // CRC32
    let mut crc_data = Vec::new();
    crc_data.extend_from_slice(chunk_type);
    crc_data.extend_from_slice(chunk_data);
    let crc = crc32(&crc_data);
    data.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Minimal zlib compression (store only, no actual compression).
fn miniz_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();

    // Zlib header (no compression)
    out.push(0x78); // CMF
    out.push(0x01); // FLG

    // Deflate blocks
    let mut remaining = data;
    while !remaining.is_empty() {
        let chunk_size = remaining.len().min(65535);
        let is_final = chunk_size == remaining.len();

        out.push(if is_final { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00 (stored)
        out.push((chunk_size & 0xFF) as u8);
        out.push(((chunk_size >> 8) & 0xFF) as u8);
        out.push((!chunk_size & 0xFF) as u8);
        out.push(((!chunk_size >> 8) & 0xFF) as u8);
        out.extend_from_slice(&remaining[..chunk_size]);
        remaining = &remaining[chunk_size..];
    }

    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());

    out
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Create temp directory for test files.
fn temp_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("butteraugli-test-{}-{}", std::process::id(), id));
    fs::create_dir_all(&dir).expect("Failed to create temp dir");
    dir
}

#[test]
fn test_identical_images() {
    let dir = temp_dir();
    let img1 = dir.join("img1.png");
    let img2 = dir.join("img2.png");

    create_solid_png(&img1, 128, 128, 128);
    create_solid_png(&img2, 128, 128, 128);

    let output = Command::new(butteraugli_bin())
        .args([img1.to_str().unwrap(), img2.to_str().unwrap()])
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success(), "Exit code should be 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Butteraugli score:"), "Should output score");

    // Clean up
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_different_images() {
    let dir = temp_dir();
    let img1 = dir.join("black.png");
    let img2 = dir.join("white.png");

    create_solid_png(&img1, 0, 0, 0);
    create_solid_png(&img2, 255, 255, 255);

    let output = Command::new(butteraugli_bin())
        .args([img1.to_str().unwrap(), img2.to_str().unwrap()])
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Butteraugli score:"));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_quiet_mode() {
    let dir = temp_dir();
    let img1 = dir.join("img1.png");
    let img2 = dir.join("img2.png");

    create_solid_png(&img1, 100, 100, 100);
    create_solid_png(&img2, 100, 100, 100);

    let output = Command::new(butteraugli_bin())
        .args(["--quiet", img1.to_str().unwrap(), img2.to_str().unwrap()])
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should just be a number
    let score: f64 = stdout.trim().parse().expect("Should output just a number");
    assert!(score >= 0.0, "Score should be non-negative");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_max_score_pass() {
    let dir = temp_dir();
    let img1 = dir.join("img1.png");
    let img2 = dir.join("img2.png");

    // Identical images should have score ~0
    create_solid_png(&img1, 128, 128, 128);
    create_solid_png(&img2, 128, 128, 128);

    let output = Command::new(butteraugli_bin())
        .args([
            "--max-score",
            "1.0",
            img1.to_str().unwrap(),
            img2.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run butteraugli");

    assert!(
        output.status.success(),
        "Should pass when score < max-score"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_max_score_fail() {
    let dir = temp_dir();
    let img1 = dir.join("black.png");
    let img2 = dir.join("white.png");

    create_solid_png(&img1, 0, 0, 0);
    create_solid_png(&img2, 255, 255, 255);

    let output = Command::new(butteraugli_bin())
        .args([
            "--max-score",
            "0.1",
            img1.to_str().unwrap(),
            img2.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run butteraugli");

    assert_eq!(
        output.status.code(),
        Some(1),
        "Should exit with code 1 when score > max-score"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_json_output() {
    let dir = temp_dir();
    let img1 = dir.join("img1.png");
    let img2 = dir.join("img2.png");

    create_solid_png(&img1, 128, 128, 128);
    create_solid_png(&img2, 128, 128, 128);

    let output = Command::new(butteraugli_bin())
        .args(["--json", img1.to_str().unwrap(), img2.to_str().unwrap()])
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"score\""), "JSON should contain score");
    assert!(
        stdout.contains("\"quality_rating\""),
        "JSON should contain quality_rating"
    );
    assert!(stdout.contains("\"width\""), "JSON should contain width");
    assert!(stdout.contains("\"height\""), "JSON should contain height");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_quality_format() {
    let dir = temp_dir();
    let img1 = dir.join("img1.png");
    let img2 = dir.join("img2.png");

    create_solid_png(&img1, 128, 128, 128);
    create_solid_png(&img2, 128, 128, 128);

    let output = Command::new(butteraugli_bin())
        .args([
            "--quality",
            "--color=never",
            img1.to_str().unwrap(),
            img2.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Quality:"),
        "Should show quality description"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_missing_file() {
    let output = Command::new(butteraugli_bin())
        .args(["nonexistent1.png", "nonexistent2.png"])
        .output()
        .expect("Failed to run butteraugli");

    assert_eq!(
        output.status.code(),
        Some(2),
        "Should exit with code 2 on error"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error"), "Should print error message");
}

#[test]
fn test_batch_mode() {
    let dir = temp_dir();
    let dir1 = dir.join("ref");
    let dir2 = dir.join("dist");
    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    // Create matching files
    create_solid_png(&dir1.join("a.png"), 100, 100, 100);
    create_solid_png(&dir2.join("a.png"), 100, 100, 100);
    create_solid_png(&dir1.join("b.png"), 50, 50, 50);
    create_solid_png(&dir2.join("b.png"), 60, 60, 60);

    let output = Command::new(butteraugli_bin())
        .args([
            "--batch",
            "--color=never",
            dir1.to_str().unwrap(),
            dir2.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("a.png"), "Should list a.png");
    assert!(stdout.contains("b.png"), "Should list b.png");
    assert!(stdout.contains("Summary"), "Should show summary");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_version() {
    let output = Command::new(butteraugli_bin())
        .arg("--version")
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("butteraugli"), "Should show name");
    assert!(stdout.contains("0."), "Should show version");
}

#[test]
fn test_help() {
    let output = Command::new(butteraugli_bin())
        .arg("--help")
        .output()
        .expect("Failed to run butteraugli");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("REFERENCE"), "Should show REFERENCE arg");
    assert!(stdout.contains("DISTORTED"), "Should show DISTORTED arg");
    assert!(stdout.contains("--max-score"), "Should show --max-score");
    assert!(stdout.contains("--batch"), "Should show --batch");
}
