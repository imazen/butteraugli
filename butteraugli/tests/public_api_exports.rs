//! Compile and exercise the generic RGB root export independently of rustdoc's inventory.
use butteraugli::{ButteraugliParams, ButteraugliReference, Img, RGB};

#[test]
fn exported_rgb_supports_the_linear_reference_api() {
    let (w, h) = (37, 41);
    let source: Vec<RGB<f32>> = (0..w * h)
        .map(|i| RGB::new((i * 7 % 251) as f32 / 251.0, 0.3, 0.4))
        .collect();
    let target: Vec<RGB<f32>> = source
        .iter()
        .map(|p| RGB::new((p.r + 0.03).min(1.0), p.g, p.b))
        .collect();
    let flat = |pixels: &[RGB<f32>]| {
        pixels
            .iter()
            .flat_map(|p| [p.r, p.g, p.b])
            .collect::<Vec<_>>()
    };
    let (source_flat, target_flat) = (flat(&source), flat(&target));
    let source = Img::new(source, w, h);
    let target = Img::new(target, w, h);
    let params = ButteraugliParams::default();
    let typed = ButteraugliReference::from_linear(source.as_ref(), params.clone()).unwrap();
    let flat = ButteraugliReference::new_linear(&source_flat, w, h, params).unwrap();
    let actual = typed.compare_linear_imgref(target.as_ref()).unwrap();
    let expected = flat.compare_linear(&target_flat).unwrap();
    assert_eq!(actual.score, expected.score);
    assert_eq!(actual.pnorm_3, expected.pnorm_3);
    assert_eq!(
        actual.diffmap.unwrap().buf(),
        expected.diffmap.unwrap().buf()
    );
}
