use butteraugli::{ButteraugliParams, Img, RGB8, butteraugli};
fn pair(w: usize, h: usize) -> (Vec<RGB8>, Vec<RGB8>) {
    let a: Vec<_> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let c = |c: usize| ((x * 13 + y * 17 + c * 71 + (x * y) % 29) % 256) as u8;
            RGB8::new(c(0), c(1), c(2))
        })
        .collect();
    let b = a
        .iter()
        .map(|p| RGB8::new(p.r / 8 * 8, p.g / 8 * 8, p.b / 8 * 8))
        .collect();
    (a, b)
}

#[test]
fn capture_main_bits() {
    for (w, h) in [(16,16), (17,19), (65,49)] {
        let (a,b) = pair(w,h);
        let r = butteraugli(Img::new(a,w,h).as_ref(), Img::new(b,w,h).as_ref(),
            &ButteraugliParams::new().with_compute_diffmap(true)).unwrap();
        let hash = r.diffmap.unwrap().buf().iter().fold(0xcbf29ce484222325_u64,
            |h,v| (h ^ u64::from(v.to_bits())).wrapping_mul(0x100000001b3));
        println!("PIN {w} {h} {:016x} {:016x} {hash:016x}", r.score.to_bits(), r.pnorm_3.to_bits());
    }
}
