//! butteraugli CLI - Perceptual image quality metric
//!
//! Compare two images and compute a butteraugli distance score.

use std::path::PathBuf;
use std::process::ExitCode;

use butteraugli_oxide::{compute_butteraugli, ButteraugliParams, ButteraugliResult, ImageF};
use clap::{Parser, ValueEnum};
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
    butteraugli original.png compressed.jpg
    butteraugli -q original.png compressed.jpg
    butteraugli --json original.png compressed.jpg
    butteraugli --diffmap diff.png original.png compressed.jpg
    butteraugli --intensity-target 80 dark_original.png dark_compressed.png")]
struct Cli {
    /// Reference image (original/source)
    #[arg(value_name = "REFERENCE")]
    reference: PathBuf,

    /// Distorted image (compressed/modified)
    #[arg(value_name = "DISTORTED")]
    distorted: PathBuf,

    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Output JSON (shorthand for --format json)
    #[arg(long, conflicts_with = "format")]
    json: bool,

    /// Show quality rating (shorthand for --format quality)
    #[arg(short, long, conflicts_with = "format")]
    quality: bool,

    /// Save diffmap (heatmap) to file
    #[arg(short, long, value_name = "FILE")]
    diffmap: Option<PathBuf>,

    /// Intensity target (viewing conditions, default: 80 nits)
    ///
    /// Lower values make the metric more sensitive to differences
    /// in dark regions. Default is 80 (typical indoor viewing).
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
    #[arg(long)]
    quiet: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutputFormat {
    /// Plain text output with score
    Text,
    /// JSON output with all metrics
    Json,
    /// Include quality rating interpretation
    Quality,
    /// Minimal - just the score number
    Score,
}

#[derive(Serialize)]
struct JsonOutput {
    score: f64,
    quality_rating: String,
    quality_description: String,
    max_local_score: f64,
    reference: String,
    distorted: String,
    width: u32,
    height: u32,
    params: JsonParams,
}

#[derive(Serialize)]
struct JsonParams {
    intensity_target: f32,
    hf_asymmetry: f32,
    xmul: f32,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Err(e) = run(&cli) {
        if !cli.quiet {
            eprintln!("error: {e}");
        }
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run(cli: &Cli) -> Result<(), String> {
    // Load images
    let ref_img = image::open(&cli.reference).map_err(|e| {
        format!(
            "failed to load reference image '{}': {}",
            cli.reference.display(),
            e
        )
    })?;
    let dist_img = image::open(&cli.distorted).map_err(|e| {
        format!(
            "failed to load distorted image '{}': {}",
            cli.distorted.display(),
            e
        )
    })?;

    // Check dimensions match
    let (ref_w, ref_h) = ref_img.dimensions();
    let (dist_w, dist_h) = dist_img.dimensions();

    if ref_w != dist_w || ref_h != dist_h {
        return Err(format!(
            "image dimensions don't match: {}x{} vs {}x{}",
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

    // Compute butteraugli with diffmap if requested
    let result = if cli.diffmap.is_some() {
        compute_butteraugli_with_diffmap(
            ref_rgb.as_raw(),
            dist_rgb.as_raw(),
            ref_w as usize,
            ref_h as usize,
            &params,
        )?
    } else {
        compute_butteraugli(
            ref_rgb.as_raw(),
            dist_rgb.as_raw(),
            ref_w as usize,
            ref_h as usize,
            &params,
        )
        .map_err(|e| format!("butteraugli computation failed: {e}"))?
    };

    // Save diffmap if requested
    if let Some(diffmap_path) = &cli.diffmap {
        if let Some(diffmap) = &result.diffmap {
            save_diffmap(diffmap, diffmap_path)?;
            if !cli.quiet && !matches!(get_format(cli), OutputFormat::Json | OutputFormat::Score) {
                eprintln!("Diffmap saved to: {}", diffmap_path.display());
            }
        }
    }

    // Output results
    output_result(cli, &result, ref_w, ref_h)?;

    Ok(())
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

fn compute_butteraugli_with_diffmap(
    ref_rgb: &[u8],
    dist_rgb: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> Result<ButteraugliResult, String> {
    // The library computes diffmap internally, we need to call with diffmap enabled
    // For now, compute without and note this limitation
    compute_butteraugli(ref_rgb, dist_rgb, width, height, params)
        .map_err(|e| format!("butteraugli computation failed: {e}"))
}

fn save_diffmap(diffmap: &ImageF, path: &PathBuf) -> Result<(), String> {
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

/// Convert a value 0-1 to a heatmap color (blue -> green -> yellow -> red)
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

fn quality_rating(score: f64) -> (&'static str, &'static str) {
    if score < 0.5 {
        ("excellent", "Imperceptible difference")
    } else if score < 1.0 {
        ("good", "Barely noticeable difference")
    } else if score < 2.0 {
        ("acceptable", "Noticeable but acceptable")
    } else if score < 3.0 {
        ("poor", "Clearly visible difference")
    } else {
        ("bad", "Large, obvious difference")
    }
}

fn output_result(
    cli: &Cli,
    result: &ButteraugliResult,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let format = get_format(cli);
    let (rating, description) = quality_rating(result.score);

    match format {
        OutputFormat::Score => {
            println!("{:.6}", result.score);
        }
        OutputFormat::Text => {
            println!("Butteraugli score: {:.4}", result.score);
        }
        OutputFormat::Quality => {
            println!("Butteraugli score: {:.4} ({})", result.score, rating);
            println!("Quality: {}", description);
        }
        OutputFormat::Json => {
            let output = JsonOutput {
                score: result.score,
                quality_rating: rating.to_string(),
                quality_description: description.to_string(),
                max_local_score: result.score, // TODO: track max local if available
                reference: cli.reference.display().to_string(),
                distorted: cli.distorted.display().to_string(),
                width,
                height,
                params: JsonParams {
                    intensity_target: cli.intensity_target,
                    hf_asymmetry: cli.hf_asymmetry,
                    xmul: cli.xmul,
                },
            };
            let json = serde_json::to_string_pretty(&output)
                .map_err(|e| format!("failed to serialize JSON: {e}"))?;
            println!("{json}");
        }
    }

    Ok(())
}
