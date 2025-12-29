//! butteraugli CLI - Perceptual image quality metric
//!
//! Compare two images and compute a butteraugli distance score.

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use butteraugli::{compute_butteraugli, ButteraugliParams, ButteraugliResult, ImageF};
use clap::{ArgAction, ColorChoice, Parser, ValueEnum};
use colored::Colorize;
use image::GenericImageView;
use serde::Serialize;

/// Butteraugli perceptual image quality metric
///
/// Computes the perceptual distance between two images. Lower scores mean
/// the images are more similar. A score of 0 means identical images.
///
/// Score interpretation:
///   0.0       - Identical images
///   0.0 - 0.5 - Imperceptible difference
///   0.5 - 1.0 - Barely noticeable
///   1.0 - 2.0 - Noticeable but acceptable
///   2.0 - 3.0 - Clearly visible difference
///   3.0+      - Large difference
#[derive(Parser, Debug)]
#[command(name = "butteraugli")]
#[command(author, version, about, long_about = None)]
#[command(after_help = "EXAMPLES:
    Compare two images:
        butteraugli original.png compressed.jpg

    Show quality rating with colors:
        butteraugli -q original.png compressed.jpg

    CI mode - fail if score exceeds threshold:
        butteraugli --max-score 1.5 original.png compressed.jpg

    Compare all PNGs in two directories:
        butteraugli --batch dir1/ dir2/

    Output JSON for scripting:
        butteraugli --json original.png compressed.jpg

    Save difference heatmap:
        butteraugli --diffmap diff.png original.png compressed.jpg

    HDR content (higher intensity target):
        butteraugli --intensity-target 250 hdr_ref.png hdr_test.png

EXIT CODES:
    0 - Success (score within threshold if --max-score specified)
    1 - Score exceeded threshold (--max-score)
    2 - Error (file not found, invalid image, etc.)")]
struct Cli {
    /// Reference image or directory (original/source)
    #[arg(value_name = "REFERENCE")]
    reference: PathBuf,

    /// Distorted image or directory (compressed/modified)
    #[arg(value_name = "DISTORTED")]
    distorted: PathBuf,

    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Output JSON (shorthand for --format json)
    #[arg(long, conflicts_with = "format")]
    json: bool,

    /// Show quality rating with colors (shorthand for --format quality)
    #[arg(short, long, conflicts_with = "format")]
    quality: bool,

    /// Save diffmap (heatmap) to file
    #[arg(short, long, value_name = "FILE")]
    diffmap: Option<PathBuf>,

    /// Maximum acceptable score (exit code 1 if exceeded)
    ///
    /// Useful for CI pipelines to enforce quality thresholds.
    /// Common thresholds: 1.0 (good), 1.5 (acceptable), 2.0 (bad)
    #[arg(long, value_name = "SCORE")]
    max_score: Option<f64>,

    /// Batch mode: compare matching files in two directories
    #[arg(long, short = 'b')]
    batch: bool,

    /// File extensions to include in batch mode (comma-separated)
    #[arg(
        long,
        default_value = "png,jpg,jpeg,webp,gif,bmp",
        value_delimiter = ','
    )]
    extensions: Vec<String>,

    /// Intensity target (viewing conditions, default: 80 nits)
    ///
    /// Lower values make the metric more sensitive to differences
    /// in dark regions. Default is 80 (typical indoor viewing).
    /// Use 250+ for HDR content.
    #[arg(long, default_value = "80.0", value_name = "NITS")]
    intensity_target: f32,

    /// High-frequency asymmetry factor
    ///
    /// Controls sensitivity to high-frequency artifacts.
    /// Higher values penalize blurring more than ringing.
    #[arg(long, default_value = "1.0", value_name = "FACTOR")]
    hf_asymmetry: f32,

    /// X channel multiplier (color sensitivity)
    #[arg(long, default_value = "1.0", value_name = "FACTOR")]
    xmul: f32,

    /// Quiet mode - only output the score number
    #[arg(long, short = 's', action = ArgAction::SetTrue)]
    quiet: bool,

    /// Control color output
    #[arg(long, value_enum, default_value = "auto")]
    color: ColorChoice,

    /// Continue on errors in batch mode
    #[arg(long)]
    keep_going: bool,

    /// Show summary statistics in batch mode
    #[arg(long)]
    summary: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    /// Plain text output with score
    Text,
    /// JSON output with all metrics
    Json,
    /// Include quality rating interpretation (with colors)
    Quality,
    /// Minimal - just the score number
    Score,
}

#[derive(Serialize)]
struct JsonOutput {
    score: f64,
    quality_rating: String,
    quality_description: String,
    reference: String,
    distorted: String,
    width: u32,
    height: u32,
    params: JsonParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    threshold_exceeded: Option<bool>,
}

#[derive(Serialize)]
struct JsonParams {
    intensity_target: f32,
    hf_asymmetry: f32,
    xmul: f32,
}

#[derive(Serialize)]
struct BatchJsonOutput {
    results: Vec<JsonOutput>,
    summary: BatchSummary,
}

#[derive(Serialize)]
struct BatchSummary {
    total: usize,
    passed: usize,
    failed: usize,
    errors: usize,
    min_score: f64,
    max_score: f64,
    mean_score: f64,
}

struct ComparisonResult {
    reference: PathBuf,
    distorted: PathBuf,
    result: Result<(ButteraugliResult, u32, u32), String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Set up color output
    setup_colors(&cli);

    if cli.batch || (cli.reference.is_dir() && cli.distorted.is_dir()) {
        run_batch(&cli)
    } else {
        run_single(&cli)
    }
}

fn setup_colors(cli: &Cli) {
    match cli.color {
        ColorChoice::Always => colored::control::set_override(true),
        ColorChoice::Never => colored::control::set_override(false),
        ColorChoice::Auto => {
            // Disable colors if not a terminal
            if !io::stdout().is_terminal() {
                colored::control::set_override(false);
            }
        }
    }
}

fn run_single(cli: &Cli) -> ExitCode {
    match compare_images(cli, &cli.reference, &cli.distorted) {
        Ok((result, width, height)) => {
            // Save diffmap if requested
            if let Some(diffmap_path) = &cli.diffmap {
                if let Some(diffmap) = &result.diffmap {
                    if let Err(e) = save_diffmap(diffmap, diffmap_path) {
                        if !cli.quiet {
                            eprintln!("{}: {}", "error".red().bold(), e);
                        }
                        return ExitCode::from(2);
                    }
                    if !cli.quiet && get_format(cli) != OutputFormat::Json {
                        eprintln!("Diffmap saved to: {}", diffmap_path.display());
                    }
                }
            }

            // Output results
            if let Err(e) = output_single_result(cli, &result, width, height) {
                if !cli.quiet {
                    eprintln!("{}: {}", "error".red().bold(), e);
                }
                return ExitCode::from(2);
            }

            // Check threshold
            if let Some(max_score) = cli.max_score {
                if result.score > max_score {
                    return ExitCode::from(1);
                }
            }

            ExitCode::SUCCESS
        }
        Err(e) => {
            if !cli.quiet {
                eprintln!("{}: {}", "error".red().bold(), e);
            }
            ExitCode::from(2)
        }
    }
}

fn run_batch(cli: &Cli) -> ExitCode {
    if !cli.reference.is_dir() {
        eprintln!(
            "{}: reference path '{}' is not a directory",
            "error".red().bold(),
            cli.reference.display()
        );
        return ExitCode::from(2);
    }
    if !cli.distorted.is_dir() {
        eprintln!(
            "{}: distorted path '{}' is not a directory",
            "error".red().bold(),
            cli.distorted.display()
        );
        return ExitCode::from(2);
    }

    // Find matching files
    let pairs = match find_matching_files(&cli.reference, &cli.distorted, &cli.extensions) {
        Ok(pairs) => pairs,
        Err(e) => {
            eprintln!("{}: {}", "error".red().bold(), e);
            return ExitCode::from(2);
        }
    };

    if pairs.is_empty() {
        eprintln!(
            "{}: no matching image files found",
            "warning".yellow().bold()
        );
        return ExitCode::from(2);
    }

    // Compare all pairs
    let mut results: Vec<ComparisonResult> = Vec::new();
    let mut had_errors = false;
    let mut threshold_exceeded = false;

    for (ref_path, dist_path) in &pairs {
        let comparison = compare_images(cli, ref_path, dist_path);

        if let Err(ref e) = comparison {
            had_errors = true;
            if !cli.keep_going {
                eprintln!("{}: {}: {}", "error".red().bold(), ref_path.display(), e);
                return ExitCode::from(2);
            }
        }

        if let Ok((ref result, _, _)) = comparison {
            if let Some(max_score) = cli.max_score {
                if result.score > max_score {
                    threshold_exceeded = true;
                }
            }
        }

        results.push(ComparisonResult {
            reference: ref_path.clone(),
            distorted: dist_path.clone(),
            result: comparison,
        });
    }

    // Output results
    if let Err(e) = output_batch_results(cli, &results) {
        eprintln!("{}: {}", "error".red().bold(), e);
        return ExitCode::from(2);
    }

    if threshold_exceeded {
        ExitCode::from(1)
    } else if had_errors {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}

fn find_matching_files(
    ref_dir: &Path,
    dist_dir: &Path,
    extensions: &[String],
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let extensions: Vec<String> = extensions.iter().map(|e| e.to_lowercase()).collect();

    let mut pairs = Vec::new();

    let entries = std::fs::read_dir(ref_dir)
        .map_err(|e| format!("failed to read directory '{}': {}", ref_dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read directory entry: {}", e))?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        // Check extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        if !extensions.contains(&ext) {
            continue;
        }

        // Find matching file in distorted directory
        let filename = path.file_name().unwrap();
        let dist_path = dist_dir.join(filename);

        if dist_path.exists() {
            pairs.push((path, dist_path));
        }
    }

    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(pairs)
}

fn compare_images(
    cli: &Cli,
    ref_path: &Path,
    dist_path: &Path,
) -> Result<(ButteraugliResult, u32, u32), String> {
    // Load images
    let ref_img = image::open(ref_path)
        .map_err(|e| format!("failed to load '{}': {}", ref_path.display(), e))?;
    let dist_img = image::open(dist_path)
        .map_err(|e| format!("failed to load '{}': {}", dist_path.display(), e))?;

    // Check dimensions match
    let (ref_w, ref_h) = ref_img.dimensions();
    let (dist_w, dist_h) = dist_img.dimensions();

    if ref_w != dist_w || ref_h != dist_h {
        return Err(format!(
            "dimension mismatch: {}x{} vs {}x{}",
            ref_w, ref_h, dist_w, dist_h
        ));
    }

    // Convert to RGB8
    let ref_rgb = ref_img.to_rgb8();
    let dist_rgb = dist_img.to_rgb8();

    // Set up parameters
    let params = ButteraugliParams::default()
        .with_intensity_target(cli.intensity_target)
        .with_hf_asymmetry(cli.hf_asymmetry)
        .with_xmul(cli.xmul);

    // Compute butteraugli
    let result = compute_butteraugli(
        ref_rgb.as_raw(),
        dist_rgb.as_raw(),
        ref_w as usize,
        ref_h as usize,
        &params,
    )
    .map_err(|e| format!("butteraugli failed: {e}"))?;

    Ok((result, ref_w, ref_h))
}

fn get_format(cli: &Cli) -> OutputFormat {
    if cli.json {
        OutputFormat::Json
    } else if cli.quality {
        OutputFormat::Quality
    } else if cli.quiet {
        OutputFormat::Score
    } else {
        cli.format
    }
}

fn save_diffmap(diffmap: &ImageF, path: &Path) -> Result<(), String> {
    let width = diffmap.width();
    let height = diffmap.height();

    // Convert diffmap to RGB heatmap
    let mut rgb_data = Vec::with_capacity(width * height * 3);

    // Find max value for normalization
    let mut max_val = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            max_val = max_val.max(diffmap.get(x, y));
        }
    }
    let max_val = max_val.max(1.0);

    for y in 0..height {
        for x in 0..width {
            let val = diffmap.get(x, y);
            let normalized = (val / max_val).clamp(0.0, 1.0);
            let (r, g, b) = heatmap_color(normalized);
            rgb_data.push(r);
            rgb_data.push(g);
            rgb_data.push(b);
        }
    }

    // Save as PNG
    image::save_buffer(
        path,
        &rgb_data,
        width as u32,
        height as u32,
        image::ColorType::Rgb8,
    )
    .map_err(|e| format!("failed to save diffmap: {e}"))
}

/// Convert a value 0-1 to a heatmap color (blue -> cyan -> green -> yellow -> red)
fn heatmap_color(val: f32) -> (u8, u8, u8) {
    let v = val.clamp(0.0, 1.0);

    if v < 0.25 {
        // Blue to Cyan
        let t = v / 0.25;
        (0, (t * 255.0) as u8, 255)
    } else if v < 0.5 {
        // Cyan to Green
        let t = (v - 0.25) / 0.25;
        (0, 255, (255.0 * (1.0 - t)) as u8)
    } else if v < 0.75 {
        // Green to Yellow
        let t = (v - 0.5) / 0.25;
        ((t * 255.0) as u8, 255, 0)
    } else {
        // Yellow to Red
        let t = (v - 0.75) / 0.25;
        (255, (255.0 * (1.0 - t)) as u8, 0)
    }
}

fn quality_rating(score: f64) -> (&'static str, &'static str, colored::Color) {
    use colored::Color;
    if score < 0.5 {
        ("excellent", "Imperceptible difference", Color::Green)
    } else if score < 1.0 {
        ("good", "Barely noticeable difference", Color::Green)
    } else if score < 2.0 {
        ("acceptable", "Noticeable but acceptable", Color::Yellow)
    } else if score < 3.0 {
        ("poor", "Clearly visible difference", Color::Red)
    } else {
        ("bad", "Large, obvious difference", Color::Red)
    }
}

fn output_single_result(
    cli: &Cli,
    result: &ButteraugliResult,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let format = get_format(cli);
    let (rating, description, color) = quality_rating(result.score);

    match format {
        OutputFormat::Score => {
            println!("{:.6}", result.score);
        }
        OutputFormat::Text => {
            let score_str = format!("{:.4}", result.score);
            if let Some(max_score) = cli.max_score {
                if result.score > max_score {
                    println!(
                        "Butteraugli score: {} (exceeds threshold {})",
                        score_str.color(color),
                        max_score
                    );
                } else {
                    println!("Butteraugli score: {}", score_str.color(color));
                }
            } else {
                println!("Butteraugli score: {}", score_str.color(color));
            }
        }
        OutputFormat::Quality => {
            let score_str = format!("{:.4}", result.score);
            let rating_colored = rating.color(color).bold();
            println!(
                "Butteraugli score: {} ({})",
                score_str.color(color),
                rating_colored
            );
            println!("Quality: {}", description);

            if let Some(max_score) = cli.max_score {
                if result.score > max_score {
                    println!(
                        "{}",
                        format!("Threshold exceeded: {:.4} > {}", result.score, max_score)
                            .red()
                            .bold()
                    );
                } else {
                    println!(
                        "{}",
                        format!("Threshold passed: {:.4} <= {}", result.score, max_score).green()
                    );
                }
            }
        }
        OutputFormat::Json => {
            let threshold_exceeded = cli.max_score.map(|max| result.score > max);
            let output = JsonOutput {
                score: result.score,
                quality_rating: rating.to_string(),
                quality_description: description.to_string(),
                reference: cli.reference.display().to_string(),
                distorted: cli.distorted.display().to_string(),
                width,
                height,
                params: JsonParams {
                    intensity_target: cli.intensity_target,
                    hf_asymmetry: cli.hf_asymmetry,
                    xmul: cli.xmul,
                },
                threshold_exceeded,
            };
            let json = serde_json::to_string_pretty(&output)
                .map_err(|e| format!("failed to serialize JSON: {e}"))?;
            println!("{json}");
        }
    }

    Ok(())
}

fn output_batch_results(cli: &Cli, results: &[ComparisonResult]) -> Result<(), String> {
    let format = get_format(cli);

    // Collect scores for summary
    let mut scores: Vec<f64> = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    let mut errors = 0;

    for cr in results {
        match &cr.result {
            Ok((result, _, _)) => {
                scores.push(result.score);
                if let Some(max_score) = cli.max_score {
                    if result.score > max_score {
                        failed += 1;
                    } else {
                        passed += 1;
                    }
                } else {
                    passed += 1;
                }
            }
            Err(_) => {
                errors += 1;
            }
        }
    }

    let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    };

    match format {
        OutputFormat::Json => {
            let mut json_results = Vec::new();
            for cr in results {
                match &cr.result {
                    Ok((result, width, height)) => {
                        let (rating, desc, _) = quality_rating(result.score);
                        let threshold_exceeded = cli.max_score.map(|max| result.score > max);
                        json_results.push(JsonOutput {
                            score: result.score,
                            quality_rating: rating.to_string(),
                            quality_description: desc.to_string(),
                            reference: cr.reference.display().to_string(),
                            distorted: cr.distorted.display().to_string(),
                            width: *width,
                            height: *height,
                            params: JsonParams {
                                intensity_target: cli.intensity_target,
                                hf_asymmetry: cli.hf_asymmetry,
                                xmul: cli.xmul,
                            },
                            threshold_exceeded,
                        });
                    }
                    Err(_) => {
                        // Skip errors in JSON output
                    }
                }
            }

            let batch_output = BatchJsonOutput {
                results: json_results,
                summary: BatchSummary {
                    total: results.len(),
                    passed,
                    failed,
                    errors,
                    min_score: if min_score.is_finite() {
                        min_score
                    } else {
                        0.0
                    },
                    max_score: if max_score.is_finite() {
                        max_score
                    } else {
                        0.0
                    },
                    mean_score,
                },
            };

            let json = serde_json::to_string_pretty(&batch_output)
                .map_err(|e| format!("failed to serialize JSON: {e}"))?;
            println!("{json}");
        }
        OutputFormat::Score => {
            for cr in results {
                if let Ok((result, _, _)) = &cr.result {
                    println!("{:.6}", result.score);
                }
            }
        }
        _ => {
            // Text or Quality format
            let name_width = results
                .iter()
                .map(|cr| cr.reference.file_name().unwrap_or_default().len())
                .max()
                .unwrap_or(20);

            for cr in results {
                let filename = cr
                    .reference
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?");

                match &cr.result {
                    Ok((result, _, _)) => {
                        let (rating, _, color) = quality_rating(result.score);
                        let score_str = format!("{:.4}", result.score);

                        let status = if let Some(max) = cli.max_score {
                            if result.score > max {
                                "FAIL".red().bold()
                            } else {
                                "PASS".green().bold()
                            }
                        } else {
                            rating.color(color).bold()
                        };

                        println!(
                            "{:width$}  {:>8}  {}",
                            filename,
                            score_str.color(color),
                            status,
                            width = name_width
                        );
                    }
                    Err(e) => {
                        println!(
                            "{:width$}  {:>8}  {}",
                            filename,
                            "-".dimmed(),
                            format!("ERROR: {}", e).red(),
                            width = name_width
                        );
                    }
                }
            }

            // Summary
            if cli.summary || results.len() > 1 {
                println!();
                println!("{}", "Summary:".bold());
                println!(
                    "  Total: {}  Passed: {}  Failed: {}  Errors: {}",
                    results.len(),
                    passed.to_string().green(),
                    if failed > 0 {
                        failed.to_string().red()
                    } else {
                        failed.to_string().normal()
                    },
                    if errors > 0 {
                        errors.to_string().red()
                    } else {
                        errors.to_string().normal()
                    }
                );
                if !scores.is_empty() {
                    println!(
                        "  Scores: min={:.4}  max={:.4}  mean={:.4}",
                        min_score, max_score, mean_score
                    );
                }
            }
        }
    }

    // Flush stdout
    let _ = io::stdout().flush();

    Ok(())
}
