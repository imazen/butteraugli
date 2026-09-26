//! Paired source-cluster bootstrap. A draw selects complete source groups with
//! replacement, and both metrics see exactly the same selected rows.
use super::{Result, Row, has_spread};
use std::collections::BTreeMap;
use std::io::Write;
use zenstats::{LightPanel, ValAggregate, compute_panel, spearman};

const NAMES: [&str; 10] = [
    "signed_srocc",
    "srocc",
    "plcc",
    "krocc",
    "or",
    "pwrc",
    "z_rmse",
    "geomean3",
    "harmean3",
    "min3",
];

fn statistics(rows: &[&Row], candidate: bool) -> Option<[f64; 10]> {
    let target: Vec<_> = rows.iter().map(|r| r.target).collect();
    let pred: Vec<_> = rows
        .iter()
        .map(|r| if candidate { -r.candidate } else { -r.teacher })
        .collect();
    if rows.len() < 4 || !has_spread(&target) || !has_spread(&pred) {
        return None;
    }
    let p = compute_panel(&pred, &target);
    let light = LightPanel {
        srocc: p.srocc,
        plcc: p.plcc,
        pwrc: p.pwrc,
        n: p.n,
    };
    Some([
        spearman(&pred, &target),
        p.srocc,
        p.plcc,
        p.krocc,
        p.or_ratio,
        p.pwrc,
        p.z_rmse,
        light.aggregate(ValAggregate::GeomeanSPP),
        light.aggregate(ValAggregate::HarmeanSPP),
        light.aggregate(ValAggregate::MinSPP),
    ])
}

// SplitMix64, followed by rejection sampling so group counts need not be powers
// of two. The seed is mandatory in the CLI and recorded with every result.
pub(super) fn choose(state: &mut u64, n: usize) -> usize {
    let n = n as u64;
    let threshold = n.wrapping_neg() % n;
    loop {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        if z >= threshold {
            return (z % n) as usize;
        }
    }
}

fn resample<'a>(groups: &[Vec<&'a Row>], state: &mut u64) -> Vec<&'a Row> {
    let mut rows = Vec::new();
    for _ in groups {
        rows.extend_from_slice(&groups[choose(state, groups.len())]);
    }
    rows
}

pub(super) fn run(rows: &[Row], output: &str, draws: usize, seed: u64) -> Result<()> {
    if draws < 100 {
        return Err("bootstrap needs at least 100 draws".into());
    }
    let mut datasets: BTreeMap<&str, BTreeMap<&str, Vec<&Row>>> = BTreeMap::new();
    for row in rows {
        datasets
            .entry(&row.dataset)
            .or_default()
            .entry(&row.source)
            .or_default()
            .push(row);
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create_new(output)?);
    writeln!(
        out,
        "dataset\tsources\trows\tdraws\tseed\tstat\tbetter\tcandidate_minus_teacher\tp025\tp975\tundefined_draws\tstatus"
    )?;
    for (dataset, groups) in datasets {
        let groups: Vec<_> = groups.into_values().collect();
        if groups.len() < 2 {
            return Err(format!("{dataset}: needs at least two source clusters").into());
        }
        let original: Vec<_> = groups.iter().flatten().copied().collect();
        let teacher = statistics(&original, false).ok_or("undefined teacher panel")?;
        let candidate = statistics(&original, true).ok_or("undefined candidate panel")?;
        let mut samples: [Vec<f64>; 10] = std::array::from_fn(|_| Vec::with_capacity(draws));
        let mut state = seed;
        for i in 0..draws {
            let sampled = resample(&groups, &mut state);
            if let (Some(a), Some(b)) = (statistics(&sampled, false), statistics(&sampled, true)) {
                for j in 0..10 {
                    let delta = b[j] - a[j];
                    if delta.is_finite() {
                        samples[j].push(delta);
                    }
                }
            }
            if (i + 1) % 25 == 0 {
                eprintln!(
                    "bootstrap {dataset}: {}/{} source-cluster draws",
                    i + 1,
                    draws
                );
            }
        }
        for (j, sample) in samples.iter_mut().enumerate() {
            sample.sort_by(f64::total_cmp);
            let (lo, hi, status) = if sample.len() == draws {
                (
                    sample[((draws - 1) as f64 * 0.025).floor() as usize].to_string(),
                    sample[((draws - 1) as f64 * 0.975).ceil() as usize].to_string(),
                    "measured",
                )
            } else {
                ("NA".into(), "NA".into(), "undefined_resamples")
            };
            let better = if matches!(j, 4 | 6) {
                "lower"
            } else {
                "higher"
            };
            writeln!(
                out,
                "{dataset}\t{}\t{}\t{draws}\t{seed}\t{}\t{better}\t{}\t{lo}\t{hi}\t{}\t{status}",
                groups.len(),
                original.len(),
                NAMES[j],
                candidate[j] - teacher[j],
                draws - sample.len()
            )?;
        }
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draws_repeat_whole_sources_and_keep_pairs_aligned() {
        let rows: Vec<_> = (0..3)
            .flat_map(|s| {
                (0..s + 2).map(move |q| Row {
                    dataset: "d".into(),
                    source: s.to_string(),
                    codec: "c".into(),
                    pair: format!("{s}-{q}"),
                    target: q as f64,
                    teacher: (s * 10 + q) as f64,
                    candidate: (s * 10 + q) as f64 + 1.0,
                    sigma: None,
                })
            })
            .collect();
        let groups: Vec<_> = (0..3)
            .map(|s| rows.iter().filter(|r| r.source == s.to_string()).collect())
            .collect();
        let mut seed = 42;
        for _ in 0..100 {
            let draw = resample(&groups, &mut seed);
            let mut total_copies = 0;
            for group in &groups {
                let copies = draw.iter().filter(|r| std::ptr::eq(**r, group[0])).count();
                total_copies += copies;
                for row in group {
                    assert_eq!(
                        draw.iter().filter(|r| std::ptr::eq(**r, *row)).count(),
                        copies
                    );
                }
            }
            assert_eq!(total_copies, 3);
            assert!(draw.iter().all(|r| r.candidate - r.teacher == 1.0));
        }
    }
}
