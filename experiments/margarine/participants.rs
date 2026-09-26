//! Participant-cluster uncertainty for within-source metric disagreements.
//! This is a ranking diagnostic: distortion levels are not encoded byte budgets.
//! All observations by a sampled worker share one resampling weight, including
//! repeated worker/image judgments and judgments on different images.
use super::{Result, Row, parse};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const NORMS: [&str; 5] = ["max", "p1", "p2", "p3", "p6"];

struct Opinions {
    // Each image retains every (worker, rating), including repeated judgments.
    values: Vec<Vec<(usize, f64)>>,
    workers: usize,
}

fn basename(pair: &str) -> Result<&str> {
    Path::new(pair)
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "invalid image identity".into())
}

fn read_opinions(input: impl BufRead, rows: &[Row]) -> Result<Opinions> {
    read_declared_opinions(input, rows, false)
}

fn read_declared_opinions(
    input: impl BufRead,
    rows: &[Row],
    processed_live1: bool,
) -> Result<Opinions> {
    let mut images = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        if images.insert(basename(&row.pair)?, i).is_some() {
            return Err("ambiguous basename across scored images".into());
        }
    }
    let mut lines = input.lines();
    if lines.next().transpose()?.as_deref() != Some("image\tworker\trating\tdist_url\tref_url") {
        return Err("unexpected sanitized opinion header".into());
    }
    let mut values = vec![Vec::new(); rows.len()];
    let mut workers = BTreeMap::new();
    for line in lines {
        let line = line?;
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 5 || fields.iter().any(|v| v.is_empty()) {
            return Err("invalid opinion row".into());
        }
        let image = *images.get(fields[0]).ok_or("unscored opinion image")?;
        if fields[1].len() != 64 || !fields[1].bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("worker identity must be the sanitized hash".into());
        }
        let rating = if processed_live1 {
            let value: f64 = fields[2].parse()?;
            if !value.is_finite() || !(1.0..=100.0).contains(&value) {
                return Err("LIVE Release 1 processed opinion outside [1,100]".into());
            }
            value
        } else {
            let value: u8 = fields[2].parse()?;
            if !(1..=5).contains(&value) {
                return Err("KADID rating outside [1,5]".into());
            }
            f64::from(value)
        };
        let next = workers.len();
        let worker = *workers.entry(fields[1].to_owned()).or_insert(next);
        values[image].push((worker, rating));
    }
    if workers.len() < 2 || values.iter().any(|v| v.len() < 2) {
        return Err("missing participant coverage".into());
    }
    Ok(Opinions {
        values,
        workers: workers.len(),
    })
}

fn means(opinions: &Opinions, weights: &[usize]) -> Result<Vec<f64>> {
    opinions
        .values
        .iter()
        .map(|values| {
            let mut sum = 0.0;
            let mut count = 0;
            for &(worker, rating) in values {
                sum += weights[worker] as f64 * rating;
                count += weights[worker];
            }
            if count == 0 {
                Err("participant resample has an unrated image; interval unavailable".into())
            } else {
                Ok(sum / count as f64)
            }
        })
        .collect()
}

fn quantile(sorted: &[f64], probability: f64, upper: bool) -> f64 {
    let position = (sorted.len() - 1) as f64 * probability;
    sorted[if upper {
        position.ceil()
    } else {
        position.floor()
    } as usize]
}

// The largest centered error difference across any pair in each source is
// max(error)-min(error). Its global maximum covers all within-source pairs,
// without selecting the comparison family using observed human ratings.
fn maximum_pair_error(errors: &[f64], groups: &[Vec<usize>]) -> f64 {
    groups
        .iter()
        .map(|group| {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for &i in group {
                lo = lo.min(errors[i]);
                hi = hi.max(errors[i]);
            }
            hi - lo
        })
        .fold(0.0, f64::max)
}

pub(super) fn run(
    scored: &str,
    opinions: &str,
    output: &str,
    draws: usize,
    seed: u64,
    processed_live1: bool,
) -> Result<()> {
    if draws < 100 {
        return Err("participant bootstrap needs at least 100 draws".into());
    }
    let panels: Vec<_> = NORMS
        .iter()
        .map(|norm| {
            parse(&std::fs::read_to_string(
                Path::new(scored).join(format!("scores-{norm}.tsv")),
            )?)
        })
        .collect::<Result<_>>()?;
    let rows = &panels[0];
    if processed_live1 {
        let dataset = &rows[0].dataset;
        if !dataset.starts_with("live_r1_") || rows.iter().any(|r| &r.dataset != dataset) {
            return Err("LIVE Release 1 opinions require one declared codec/session cohort".into());
        }
    } else if rows.iter().any(|r| r.dataset != "kadid") {
        return Err("this raw-rating adapter is specific to KADID".into());
    }
    for panel in &panels[1..] {
        if panel.len() != rows.len()
            || panel.iter().zip(rows).any(|(a, b)| {
                (&a.dataset, &a.source, &a.pair, a.target)
                    != (&b.dataset, &b.source, &b.pair, b.target)
            })
        {
            return Err("pooling variants have different label identities or ordering".into());
        }
    }
    let input = BufReader::new(File::open(opinions)?);
    let opinions = if processed_live1 {
        read_declared_opinions(input, rows, true)?
    } else {
        read_opinions(input, rows)?
    };
    let original = means(&opinions, &vec![1; opinions.workers])?;
    if processed_live1
        && original
            .iter()
            .zip(rows)
            .any(|(a, b)| (a - b.target).abs() > 1e-9)
    {
        return Err("processed LIVE opinions do not reproduce scored labels".into());
    }
    let mut groups = BTreeMap::<&str, Vec<usize>>::new();
    for (i, row) in rows.iter().enumerate() {
        groups.entry(&row.source).or_default().push(i);
    }
    let groups: Vec<_> = groups.into_values().collect();
    std::fs::create_dir(output)?;
    let output = Path::new(output);
    std::fs::write(
        output.join("method.txt"),
        if processed_live1 {
            "LIVE Release 1: resample processed observer columns within one codec/session. Conditional on published normalization and outlier selection; no cross-cohort alignment.\n"
        } else {
            "KADID: resample workers with all their retained raw observations.\n"
        },
    )?;
    let mut progress = BufWriter::new(File::create_new(output.join("progress.log"))?);
    let mut samples = vec![vec![0.0; draws]; rows.len()];
    let mut state = seed;
    let mut maximum_errors = Vec::with_capacity(draws);
    for draw in 0..draws {
        let mut weights = vec![0usize; opinions.workers];
        for _ in 0..opinions.workers {
            weights[super::uncertainty::choose(&mut state, opinions.workers)] += 1;
        }
        let drawn = means(&opinions, &weights)?;
        let errors: Vec<_> = drawn.iter().zip(&original).map(|(a, b)| a - b).collect();
        maximum_errors.push(maximum_pair_error(&errors, &groups));
        for (values, &mean) in samples.iter_mut().zip(&drawn) {
            values[draw] = mean;
        }
        if (draw + 1) % 25 == 0 {
            writeln!(progress, "participant draw {}/{draws}", draw + 1)?;
            progress.flush()?;
            eprintln!("participant draw {}/{draws}", draw + 1);
        }
    }
    maximum_errors.sort_by(f64::total_cmp);
    let simultaneous = quantile(&maximum_errors, 0.95, true);
    let mut metadata = BufWriter::new(File::create_new(output.join("bootstrap.tsv"))?);
    writeln!(
        metadata,
        "images\tworkers\tobservations\tsources\tdraws\tseed\tsimultaneous_95_radius"
    )?;
    writeln!(
        metadata,
        "{}\t{}\t{}\t{}\t{draws}\t{seed}\t{simultaneous}",
        rows.len(),
        opinions.workers,
        opinions.values.iter().map(Vec::len).sum::<usize>(),
        groups.len()
    )?;
    let mut identities = BufWriter::new(File::create_new(output.join("images.tsv"))?);
    let mut binary = BufWriter::new(File::create_new(output.join("participant-means.f64le"))?);
    writeln!(
        identities,
        "index\tsource\tpair\tmean\tobservations\tworkers"
    )?;
    for (i, row) in rows.iter().enumerate() {
        let unique: BTreeSet<_> = opinions.values[i].iter().map(|v| v.0).collect();
        writeln!(
            identities,
            "{i}\t{}\t{}\t{}\t{}\t{}",
            row.source,
            row.pair,
            original[i],
            opinions.values[i].len(),
            unique.len()
        )?;
        for value in &samples[i] {
            binary.write_all(&value.to_le_bytes())?;
        }
    }
    binary.flush()?;
    let mut pairs = BufWriter::new(File::create_new(output.join("disagreements.tsv"))?);
    writeln!(
        pairs,
        "norm\tsource\tteacher_preferred\tcandidate_preferred\thuman_loss\tpointwise_p025\tpointwise_p975\tsimultaneous_lower\tsimultaneous_upper"
    )?;
    let mut summary = BufWriter::new(File::create_new(output.join("summary.tsv"))?);
    writeln!(
        summary,
        "norm\tall_pairs\tteacher_ties\tcandidate_ties\treversals\tpoint_loss_positive\tpointwise_95_harm\tpointwise_95_benefit\tsimultaneous_95_harm\tsimultaneous_95_benefit"
    )?;
    for (norm, panel) in NORMS.iter().zip(&panels) {
        let mut counts = [0usize; 9];
        for group in &groups {
            for (position, &i) in group.iter().enumerate() {
                for &j in &group[position + 1..] {
                    counts[0] += 1;
                    let td = panel[i].teacher - panel[j].teacher;
                    let cd = panel[i].candidate - panel[j].candidate;
                    counts[1] += usize::from(td == 0.0);
                    counts[2] += usize::from(cd == 0.0);
                    if td == 0.0 || cd == 0.0 || td.signum() == cd.signum() {
                        continue;
                    }
                    let (a, b) = if td < 0.0 { (i, j) } else { (j, i) };
                    let loss = original[a] - original[b];
                    let mut deltas: Vec<_> = samples[a]
                        .iter()
                        .zip(&samples[b])
                        .map(|(a, b)| a - b)
                        .collect();
                    deltas.sort_by(f64::total_cmp);
                    let lo = quantile(&deltas, 0.025, false);
                    let hi = quantile(&deltas, 0.975, true);
                    counts[3] += 1;
                    counts[4] += usize::from(loss > 0.0);
                    counts[5] += usize::from(lo > 0.0);
                    counts[6] += usize::from(hi < 0.0);
                    counts[7] += usize::from(loss - simultaneous > 0.0);
                    counts[8] += usize::from(loss + simultaneous < 0.0);
                    writeln!(
                        pairs,
                        "{norm}\t{}\t{}\t{}\t{loss}\t{lo}\t{hi}\t{}\t{}",
                        panel[a].source,
                        panel[a].pair,
                        panel[b].pair,
                        loss - simultaneous,
                        loss + simultaneous
                    )?;
                }
            }
        }
        write!(summary, "{norm}")?;
        for count in counts {
            write!(summary, "\t{count}")?;
        }
        writeln!(summary)?;
        summary.flush()?;
        pairs.flush()?;
        writeln!(
            progress,
            "{norm}: {} strict pair reversals; matched-rate choice gate unavailable",
            counts[3]
        )?;
        progress.flush()?;
        eprintln!("{norm}: {} strict pair reversals", counts[3]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opinion_join_preserves_repeats_and_rejects_unscored_images() {
        let rows = parse(&format!(
            "{}\nkadid\ts\tc\timages/i.png\t3\tquality\t1\t2",
            super::super::HEADER
        ))
        .unwrap();
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let input = format!(
            "image\tworker\trating\tdist_url\tref_url\ni.png\t{a}\t1\td\tr\ni.png\t{a}\t1\td\tr\ni.png\t{b}\t5\td\tr\n"
        );
        let opinions = read_opinions(input.as_bytes(), &rows).unwrap();
        assert_eq!(opinions.workers, 2);
        assert_eq!(opinions.values[0].len(), 3);
        assert_eq!(means(&opinions, &[1, 1]).unwrap(), [7.0 / 3.0]);
        assert!(read_opinions(input.replace("i.png", "other.png").as_bytes(), &rows).is_err());
    }

    #[test]
    fn processed_live_opinions_preserve_fractional_values_without_weakening_kadid() {
        let rows = parse(&format!(
            "{}\nlive_r1_jpeg_s1\ts\tjpeg\timg1.bmp\t50\tquality\t1\t2",
            super::super::HEADER
        ))
        .unwrap();
        let input = format!(
            "image\tworker\trating\tdist_url\tref_url\nimg1.bmp\t{}\t12.5\td\tr\nimg1.bmp\t{}\t87.5\td\tr\n",
            "a".repeat(64),
            "b".repeat(64)
        );
        assert!(read_opinions(input.as_bytes(), &rows).is_err());
        let opinions = read_declared_opinions(input.as_bytes(), &rows, true).unwrap();
        assert_eq!(means(&opinions, &[1, 1]).unwrap(), [50.0]);
        for invalid in ["0", "101", "NaN", "inf"] {
            assert!(
                read_declared_opinions(input.replace("12.5", invalid).as_bytes(), &rows, true)
                    .is_err()
            );
        }
    }

    #[test]
    fn repeated_observations_share_worker_weights_across_images() {
        let opinions = Opinions {
            workers: 2,
            values: vec![vec![(0, 1.0), (0, 1.0), (1, 5.0)], vec![(0, 2.0), (1, 4.0)]],
        };
        assert_eq!(means(&opinions, &[2, 0]).unwrap(), [1.0, 2.0]);
        assert_eq!(means(&opinions, &[0, 2]).unwrap(), [5.0, 4.0]);
        assert_eq!(means(&opinions, &[1, 1]).unwrap(), [7.0 / 3.0, 3.0]);
        assert!(means(&opinions, &[0, 0]).is_err());
    }

    #[test]
    fn simultaneous_radius_covers_every_pair_in_the_declared_family() {
        let errors = [0.5, -0.2, 0.1, 3.0, 2.1];
        let groups = vec![vec![0, 1, 2], vec![3, 4]];
        let radius = maximum_pair_error(&errors, &groups);
        for group in &groups {
            for &a in group {
                for &b in group {
                    assert!((errors[a] - errors[b]).abs() <= radius);
                }
            }
        }
        assert_eq!(radius, 3.0 - 2.1);
    }
}
