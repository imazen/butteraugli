//! One planar reference construction and three sequential warm comparisons.
//! Profile each process separately with /usr/bin/time -v; includes caller inputs.
use butteraugli::{ButteraugliParams, ButteraugliReference, Numerics};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let w: usize = args[1].parse().unwrap();
    let h: usize = args[2].parse().unwrap();
    let mode = if args[3] == "exact" {
        Numerics::LibjxlExact
    } else {
        Numerics::Fast
    };
    let a: [Vec<f32>; 3] = std::array::from_fn(|c| {
        (0..w * h)
            .map(|i| ((i * 13 + i / w * 17 + c * 71) % 256) as f32 / 255.0)
            .collect()
    });
    let b: [Vec<f32>; 3] =
        std::array::from_fn(|c| a[c].iter().map(|v| (v * 32.0).floor() / 32.0).collect());
    let params = ButteraugliParams::new()
        .with_numerics(mode)
        .with_compute_diffmap(true);
    println!(
        "estimate={} reference_estimate={}",
        ButteraugliReference::estimated_planar_peak_bytes(w, h, &params).unwrap(),
        ButteraugliReference::estimated_reference_bytes(w, h, &params)
    );
    let r = ButteraugliReference::new_linear_planar(&a[0], &a[1], &a[2], w, h, w, params).unwrap();
    let mut map = Vec::new();
    for _ in 0..3 {
        let scores = r
            .compare_linear_planar_into(&b[0], &b[1], &b[2], w, &mut map)
            .unwrap();
        std::hint::black_box((&scores, &map));
        println!(
            "score={:.17} p3={:.17} retained={} precompute={}",
            scores.0,
            scores.1,
            r.memory_bytes(),
            r.precompute_bytes()
        );
    }
}
