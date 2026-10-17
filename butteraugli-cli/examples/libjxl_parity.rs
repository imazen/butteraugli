use butteraugli::{ButteraugliParams, Img, RGB, RGB8, butteraugli, butteraugli_linear};
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
};
fn main() {
    let args: Vec<_> = env::args().collect();
    let a = image::open(&args[1]).unwrap().to_rgb8();
    let b = image::open(&args[2]).unwrap().to_rgb8();
    assert_eq!(a.dimensions(), b.dimensions());
    let (w, h) = (a.width() as usize, a.height() as usize);
    let aa: Vec<_> = a.pixels().map(|p| RGB8::new(p[0], p[1], p[2])).collect();
    let bb: Vec<_> = b.pixels().map(|p| RGB8::new(p[0], p[1], p[2])).collect();
    let params = ButteraugliParams::default()
        .with_numerics(if args.iter().any(|s| s == "--exact") {
            butteraugli::Numerics::LibjxlExact
        } else {
            butteraugli::Numerics::Fast
        })
        .with_intensity_target(80.0)
        .with_compute_diffmap(true);
    if let Some(n) = args.iter().find_map(|s| {
        s.strip_prefix("--bench=")
            .map(|n| n.parse::<usize>().unwrap())
    }) {
        let score = || {
            butteraugli(
                Img::new(aa.as_slice(), w, h),
                Img::new(bb.as_slice(), w, h),
                &params,
            )
            .unwrap()
        };
        std::hint::black_box(score());
        let mut samples = Vec::new();
        for _ in 0..n {
            let start = std::time::Instant::now();
            let result = score();
            let elapsed = start.elapsed().as_nanos();
            std::hint::black_box(result);
            samples.push(elapsed);
        }
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        println!("BENCH\t{}\t{:?}", sorted[n / 2], samples);
    }
    let r = if args.iter().any(|s| s == "--linear") {
        let conv = |v: &Vec<RGB8>| -> Vec<RGB<f32>> {
            v.iter()
                .map(|p| {
                    let f = |v: u8| {
                        let x = v as f64 / 255.;
                        if x <= 0.04045 {
                            (x / 12.92) as f32
                        } else {
                            ((x + 0.055) / 1.055).powf(2.4) as f32
                        }
                    };
                    RGB::new(f(p.r), f(p.g), f(p.b))
                })
                .collect()
        };
        let aaa = conv(&aa);
        let bbb = conv(&bb);
        butteraugli_linear(
            Img::new(aaa, w, h).as_ref(),
            Img::new(bbb, w, h).as_ref(),
            &params,
        )
        .unwrap()
    } else {
        butteraugli(
            Img::new(aa, w, h).as_ref(),
            Img::new(bb, w, h).as_ref(),
            &params,
        )
        .unwrap()
    };
    println!("{:.17}\t{:.17}", r.score, r.pnorm_3);
    let mut f = BufWriter::new(File::create(&args[3]).unwrap());
    let dm = r.diffmap.unwrap();
    for y in 0..h {
        for x in 0..w {
            f.write_all(&dm.buf()[y * dm.stride() + x].to_le_bytes())
                .unwrap();
        }
    }
}
