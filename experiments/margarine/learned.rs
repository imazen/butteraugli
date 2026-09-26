//! Private runtime for fit_probe.py's nonnegative log-feature model.
//! Loading a model does not establish its human-quality or resource gates.
use std::{error::Error, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone)]
pub(super) struct Model {
    scales: [f64; 168],
    weights: [[f64; 5]; 168],
}

impl Model {
    pub(super) fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    fn parse(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        if lines.next() != Some("feature\tscale\tmax\tp1\tp2\tp3\tp6") {
            return Err("unrecognized model header".into());
        }
        let mut result = Self {
            scales: [0.0; 168],
            weights: [[0.0; 5]; 168],
        };
        for (slot, original) in (0..228)
            .filter(|&i| super::student::edge_feature(i))
            .enumerate()
        {
            let line = lines.next().ok_or("model is missing a feature row")?;
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 7 || fields[0] != format!("feature_{original:03}") {
                return Err(format!("wrong model layout at feature {original}").into());
            }
            let scale: f64 = fields[1].parse()?;
            if !scale.is_finite() || scale <= 0.0 {
                return Err("model scale must be finite and positive".into());
            }
            result.scales[slot] = scale;
            for (weight, text) in result.weights[slot].iter_mut().zip(&fields[2..]) {
                *weight = text.parse()?;
                if !weight.is_finite() || *weight < 0.0 {
                    return Err("model weights must be finite and nonnegative".into());
                }
            }
        }
        if lines.next().is_some() {
            return Err("model has extra rows".into());
        }
        Ok(result)
    }

    pub(super) fn predict(&self, features: &[f64]) -> Result<[f64; 5]> {
        let features: &[f64; 168] = features.try_into().map_err(|_| "expected 168 features")?;
        let mut sums = [0.0; 5];
        for ((&value, &scale), weights) in features.iter().zip(&self.scales).zip(&self.weights) {
            if !value.is_finite() || value < 0.0 {
                return Err("features must be finite and nonnegative".into());
            }
            let transformed = (value / scale).ln_1p();
            for (sum, &weight) in sums.iter_mut().zip(weights) {
                *sum += transformed * weight;
            }
        }
        let result = sums.map(f64::exp_m1);
        if !result.iter().all(|v| v.is_finite()) {
            return Err("model produced a nonfinite prediction".into());
        }
        Ok(result)
    }
}

pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("usage: --student MODEL.tsv REF DIST".into());
    }
    let model = Model::load(&args[1])?;
    let (width, height, features) = super::resources_rgb8::edge_features(&args[2], &args[3])?;
    let scores = model.predict(&features)?;
    println!("mode\twidth\theight\tmax\tp1\tp2\tp3\tp6");
    println!(
        "margarine-probe\t{width}\t{height}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}\t{:.17e}",
        scores[0], scores[1], scores[2], scores[3], scores[4]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_model() -> String {
        let mut text = "feature\tscale\tmax\tp1\tp2\tp3\tp6\n".to_owned();
        for original in (0..228).filter(|&i| crate::student::edge_feature(i)) {
            text.push_str(&format!("feature_{original:03}\t1\t0\t0\t0\t0\t0\n"));
        }
        text
    }

    #[test]
    fn checks_model_layout_and_numeric_contract() {
        let text = valid_model();
        assert!(Model::parse(&text).is_ok());
        assert!(Model::parse(&text.replace("feature_003", "feature_004")).is_err());
        assert!(Model::parse(&text.replace("feature_003\t1", "feature_003\t0")).is_err());
        assert!(Model::parse(&text.replace("feature_003\t1\t0", "feature_003\t1\tNaN")).is_err());
        assert!(Model::parse(&(text + "extra\n")).is_err());
    }

    #[test]
    fn prediction_preserves_zero_and_uses_the_exported_formula() {
        let mut model = Model::parse(&valid_model()).unwrap();
        model.scales[0] = 4.0;
        model.weights[0][0] = 2.0;
        assert_eq!(model.predict(&[0.0; 168]).unwrap(), [0.0; 5]);
        let mut features = [0.0; 168];
        features[0] = 4.0;
        let prediction = model.predict(&features).unwrap();
        assert!((prediction[0] - 3.0).abs() < 1e-14);
        assert_eq!(&prediction[1..], &[0.0; 4]);
        features[0] = -1.0;
        assert!(model.predict(&features).is_err());
        assert!(model.predict(&[0.0; 167]).is_err());
    }
}
