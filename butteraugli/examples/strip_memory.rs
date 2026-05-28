//! Compare peak heap memory between the full-image and strip-walker
//! butteraugli paths.
//!
//! Run under heaptrack:
//!
//! ```sh
//! cargo build --release --example strip_memory
//! heaptrack target/release/examples/strip_memory full   7680 5120
//! heaptrack target/release/examples/strip_memory strip  7680 5120 256
//! heaptrack target/release/examples/strip_memory wstrip 7680 5120 256
//! heaptrack_print heaptrack.strip_memory.*.zst | grep "peak heap"
//! ```

use std::time::Instant;

use butteraugli::{
    ButteraugliParams, ButteraugliReference, Img, RGB8, butteraugli, butteraugli_strip,
};

fn make_pair(width: usize, height: usize) -> (Vec<RGB8>, Vec<RGB8>) {
    let mut src = Vec::with_capacity(width * height);
    let mut dst = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 7 + y * 13) & 0xff) as u8;
            let g = ((x * 11 + y * 3 + 50) & 0xff) as u8;
            let b = ((x * 5 + y * 17 + 100) & 0xff) as u8;
            src.push(RGB8::new(r, g, b));
            dst.push(RGB8::new(
                r.saturating_add(5),
                g.saturating_sub(3),
                b.saturating_add(2),
            ));
        }
    }
    (src, dst)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "full".to_string());
    let width: usize = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7680);
    let height: usize = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5120);
    let strip_h: u32 = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    eprintln!("butteraugli strip_memory: mode={mode} size={width}x{height} strip_h={strip_h}");
    let t0 = Instant::now();
    let (src, dst) = make_pair(width, height);
    eprintln!("  fixture built in {:.2}s", t0.elapsed().as_secs_f64());
    let src_img = Img::new(src.clone(), width, height);
    let dst_img = Img::new(dst.clone(), width, height);
    let params = ButteraugliParams::default();

    let t1 = Instant::now();
    let result = match mode.as_str() {
        "full" => butteraugli(src_img.as_ref(), dst_img.as_ref(), &params).expect("full"),
        "strip" => {
            butteraugli_strip(src_img.as_ref(), dst_img.as_ref(), &params, strip_h).expect("strip")
        }
        "wstrip" => {
            let src_bytes: Vec<u8> = src.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
            let dst_bytes: Vec<u8> = dst.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
            let r = ButteraugliReference::new(&src_bytes, width, height, params.clone())
                .expect("reference");
            r.compare_strip(&dst_bytes, strip_h).expect("compare_strip")
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    };
    eprintln!(
        "  score = {:.6} pnorm_3 = {:.6} in {:.2}s",
        result.score,
        result.pnorm_3,
        t1.elapsed().as_secs_f64()
    );
    println!("{:.6}", result.score);
}
