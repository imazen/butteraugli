#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::{BufWriter, Write};
use zenstats::{LightPanel, ValAggregate, compute_panel, spearman};

mod participants;
mod uncertainty;

const HEADER: &str = "dataset\tsource\tcodec\tpair\ttarget\tdirection\tteacher\tcandidate";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct Row {
    dataset: String,
    source: String,
    codec: String,
    pair: String,
    target: f64,        // normalized polarity only; larger = better
    teacher: f64,       // raw distance; smaller = better
    candidate: f64,     // raw distance; smaller = better
    sigma: Option<f64>, // native published label dispersion, not inferred standard error
}

fn parse(input: &str) -> Result<Vec<Row>> {
    let mut lines = input.lines();
    let header = lines.next().ok_or("missing header")?;
    let with_sigma = header == format!("{HEADER}\tsigma");
    if header != HEADER && !with_sigma {
        return Err(format!("expected header: {HEADER}").into());
    }
    let mut seen = BTreeSet::new();
    let mut orientations = BTreeMap::new();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != if with_sigma { 9 } else { 8 }
            || fields[..8.min(fields.len())].iter().any(|f| f.is_empty())
        {
            return Err(format!("line {}: expected eight nonempty fields", i + 2).into());
        }
        if !seen.insert((fields[0], fields[3])) {
            return Err(format!("line {}: duplicate dataset/pair", i + 2).into());
        }
        let sign = match fields[5] {
            "quality" => 1.0,
            "distortion" => -1.0,
            _ => {
                return Err(
                    format!("line {}: direction must be quality or distortion", i + 2).into(),
                );
            }
        };
        if orientations
            .insert(fields[0], fields[5])
            .is_some_and(|s| s != fields[5])
        {
            return Err(format!("line {}: mixed target directions in dataset", i + 2).into());
        }
        let numbers = [fields[4], fields[6], fields[7]].map(str::parse::<f64>);
        let [target, teacher, candidate] = numbers;
        let (target, teacher, candidate) = (target?, teacher?, candidate?);
        if ![target, teacher, candidate].iter().all(|v| v.is_finite())
            || teacher < 0.0
            || candidate < 0.0
        {
            return Err(format!(
                "line {}: nonfinite target or invalid metric distance",
                i + 2
            )
            .into());
        }
        let sigma = if with_sigma && !fields[8].is_empty() {
            let value = fields[8].parse::<f64>()?;
            if !value.is_finite() || value <= 0.0 {
                return Err(
                    format!("line {}: supplied sigma must be finite and positive", i + 2).into(),
                );
            }
            Some(value)
        } else {
            None
        };
        rows.push(Row {
            dataset: fields[0].into(),
            source: fields[1].into(),
            codec: fields[2].into(),
            pair: fields[3].into(),
            target: sign * target,
            teacher,
            candidate,
            sigma,
        });
    }
    if rows.is_empty() {
        return Err("no rows".into());
    }
    Ok(rows)
}

fn has_spread(values: &[f64]) -> bool {
    values.iter().any(|v| *v != values[0])
}

fn panel(out: &mut impl Write, scope: &str, dataset: &str, key: &str, rows: &[&Row]) -> Result<()> {
    let target: Vec<_> = rows.iter().map(|r| r.target).collect();
    for (arm, pred) in [
        (
            "teacher",
            rows.iter().map(|r| -r.teacher).collect::<Vec<_>>(),
        ),
        (
            "candidate",
            rows.iter().map(|r| -r.candidate).collect::<Vec<_>>(),
        ),
    ] {
        if rows.len() < 4 || !has_spread(&target) || !has_spread(&pred) {
            writeln!(
                out,
                "{scope}\t{dataset}\t{key}\t{arm}\t{}\tunavailable\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA",
                rows.len()
            )?;
            continue;
        }
        let p = compute_panel(&pred, &target);
        // Use the shared implementation, without another fit or different rows.
        let light = LightPanel {
            srocc: p.srocc,
            plcc: p.plcc,
            pwrc: p.pwrc,
            n: p.n,
        };
        writeln!(
            out,
            "{scope}\t{dataset}\t{key}\t{arm}\t{}\tmeasured\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            rows.len(),
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
        )?;
    }
    Ok(())
}

/// Supplied dispersion is reported separately from corpus-standardized panels.
/// Missing sigma never silently removes rows from a statistic.
fn published_sigma_panel(out: &mut impl Write, rows: &[Row]) -> Result<()> {
    writeln!(
        out,
        "dataset\tarm\tn\tsigma_count\tstatus\tor_published_sigma\tz_rmse_published_sigma"
    )?;
    let mut groups: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for row in rows {
        groups.entry(&row.dataset).or_default().push(row);
    }
    for (dataset, rows) in groups {
        let target: Vec<_> = rows.iter().map(|r| r.target).collect();
        let sigma: Vec<_> = rows.iter().filter_map(|r| r.sigma).collect();
        for (arm, prediction) in [
            (
                "teacher",
                rows.iter().map(|r| -r.teacher).collect::<Vec<_>>(),
            ),
            (
                "candidate",
                rows.iter().map(|r| -r.candidate).collect::<Vec<_>>(),
            ),
        ] {
            if sigma.len() != rows.len()
                || rows.len() < 4
                || !has_spread(&target)
                || !has_spread(&prediction)
            {
                writeln!(
                    out,
                    "{dataset}\t{arm}\t{}\t{}\tunavailable\tNA\tNA",
                    rows.len(),
                    sigma.len()
                )?;
                continue;
            }
            let transformed = zenstats::rescale_logistic(&prediction, &target);
            writeln!(
                out,
                "{dataset}\t{arm}\t{}\t{}\tmeasured\t{}\t{}",
                rows.len(),
                sigma.len(),
                zenstats::outlier_ratio_per_sample(&transformed, &target, &sigma),
                zenstats::z_rmse_per_sample(&transformed, &target, &sigma)
            )?;
        }
    }
    Ok(())
}

#[derive(Default, Debug, PartialEq)]
struct Orders {
    decisive: u64,
    reversed: u64,
    collapsed: u64,
    teacher_ties: u64,
}

fn orders(rows: &[&Row], epsilon: f64) -> Orders {
    let mut result = Orders::default();
    for (i, a) in rows.iter().enumerate() {
        for b in &rows[i + 1..] {
            // Cross-source order is not an encoder choice. Callers group by source.
            let teacher = a.teacher - b.teacher;
            let candidate = a.candidate - b.candidate;
            if teacher.abs() <= epsilon {
                result.teacher_ties += 1;
                continue;
            }
            result.decisive += 1;
            if candidate == 0.0 {
                result.collapsed += 1;
            } else if teacher.signum() != candidate.signum() {
                result.reversed += 1;
            }
        }
    }
    result
}

// Rank bands use only human targets. Keep every equal-target block together,
// assigning it by its first rank; populated bands can therefore differ in size.
fn quality_bands<'a>(rows: &[&'a Row], count: usize) -> Vec<Vec<&'a Row>> {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|a, b| a.target.total_cmp(&b.target));
    let mut bands = vec![Vec::new(); count];
    let mut start = 0;
    while start < sorted.len() {
        let mut end = start + 1;
        while end < sorted.len() && sorted[end].target == sorted[start].target {
            end += 1;
        }
        bands[start * count / sorted.len()].extend_from_slice(&sorted[start..end]);
        start = end;
    }
    bands
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "--quality-bands") {
        if args.len() != 4 {
            return Err(
                "usage: margarine-eval --quality-bands SCORES.tsv NEW_OUTPUT.tsv BANDS".into(),
            );
        }
        let count: usize = args[3].parse()?;
        if !(2..=100).contains(&count) {
            return Err("quality band count must be in 2..=100".into());
        }
        let rows = parse(&std::fs::read_to_string(&args[1])?)?;
        let mut groups: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
        for row in &rows {
            groups.entry(&row.dataset).or_default().push(row);
        }
        let mut out = BufWriter::new(std::fs::File::create_new(&args[2])?);
        writeln!(
            out,
            "scope\tdataset\tkey\tarm\tn\tstatus\tsigned_srocc\tsrocc\tplcc\tkrocc\tor\tpwrc\tz_rmse\tgeomean3\tharmean3\tmin3"
        )?;
        for (dataset, rows) in groups {
            for (i, band) in quality_bands(&rows, count).iter().enumerate() {
                let key = format!("{}/{count}", i + 1);
                eprintln!("quality band {dataset}/{key}: {} pairs", band.len());
                panel(&mut out, "quality_band", dataset, &key, band)?;
                out.flush()?;
            }
        }
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "--participant-pairs") {
        if args.len() != 6 {
            return Err("usage: margarine-eval --participant-pairs SCORED_DIR OPINIONS.tsv NEW_OUTPUT_DIR DRAWS SEED".into());
        }
        return participants::run(
            &args[1],
            &args[2],
            &args[3],
            args[4].parse()?,
            args[5].parse()?,
        );
    }
    if args.first().is_some_and(|a| a == "--published-sigma") {
        if args.len() != 3 {
            return Err(
                "usage: margarine-eval --published-sigma SCORES_WITH_SIGMA.tsv NEW_OUTPUT.tsv"
                    .into(),
            );
        }
        let rows = parse(&std::fs::read_to_string(&args[1])?)?;
        let mut out = BufWriter::new(std::fs::File::create_new(&args[2])?);
        published_sigma_panel(&mut out, &rows)?;
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "--bootstrap-all") {
        if args.len() != 5 {
            return Err(
                "usage: margarine-eval --bootstrap-all SCORED_DIR NEW_OUTPUT_DIR DRAWS SEED".into(),
            );
        }
        let draws = args[3].parse()?;
        let seed = args[4].parse()?;
        let output = std::path::Path::new(&args[2]);
        std::fs::create_dir(output)?;
        for norm in ["max", "p1", "p2", "p3", "p6"] {
            let input = std::path::Path::new(&args[1]).join(format!("scores-{norm}.tsv"));
            let rows = parse(&std::fs::read_to_string(input)?)?;
            eprintln!("Starting clustered panel: {norm}");
            let target = output.join(format!("{norm}.tsv"));
            uncertainty::run(
                &rows,
                target.to_str().ok_or("non-UTF8 output path")?,
                draws,
                seed,
            )?;
            eprintln!("Completed clustered panel: {norm}");
        }
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "--bootstrap") {
        if args.len() != 5 {
            return Err("usage: margarine-eval --bootstrap SCORES.tsv OUT.tsv DRAWS SEED".into());
        }
        let rows = parse(&std::fs::read_to_string(&args[1])?)?;
        return uncertainty::run(&rows, &args[2], args[3].parse()?, args[4].parse()?);
    }
    if args.len() != 3 {
        return Err("usage: margarine-eval SCORES.tsv OUTPUT.tsv TEACHER_TIE_EPSILON\nDistances must be lower-is-better. No acceptance thresholds are implicit.".into());
    }
    let epsilon: f64 = args[2].parse()?;
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("invalid epsilon".into());
    }
    let rows = parse(&std::fs::read_to_string(&args[0])?)?;
    let mut out = BufWriter::new(std::fs::File::create_new(&args[1])?);
    writeln!(
        out,
        "scope\tdataset\tkey\tarm\tn\tstatus\tsigned_srocc\tsrocc\tplcc\tkrocc\tor\tpwrc\tz_rmse\tgeomean3\tharmean3\tmin3"
    )?;
    let mut groups: BTreeMap<(&str, &str, String), Vec<&Row>> = BTreeMap::new();
    for row in &rows {
        for (scope, key) in [
            ("corpus", "all".to_string()),
            ("codec", row.codec.clone()),
            ("source", row.source.clone()),
        ] {
            groups
                .entry((scope, &row.dataset, key))
                .or_default()
                .push(row);
        }
    }
    for ((scope, dataset, key), group) in &groups {
        eprintln!("panel {scope}/{dataset}/{key}: {} pairs", group.len());
        panel(&mut out, scope, dataset, key, group)?;
        out.flush()?;
    }
    writeln!(
        out,
        "\n# Within-source order counts; epsilon={epsilon}; no byte-budget or JND gate"
    )?;
    writeln!(
        out,
        "dataset\tsource\tdecisive\treversed\tcandidate_ties\tteacher_ties"
    )?;
    for ((scope, dataset, key), group) in &groups {
        if *scope != "source" {
            continue;
        }
        let o = orders(group, epsilon);
        writeln!(
            out,
            "{dataset}\t{key}\t{}\t{}\t{}\t{}",
            o.decisive, o.reversed, o.collapsed, o.teacher_ties
        )?;
    }
    writeln!(
        out,
        "# NOT MEASURED: clustered uncertainty, bands, matched-byte/target regret, corruption, local-edit coherence, time, peak RAM"
    )?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_bands_keep_ties_and_every_row_with_correct_polarity() {
        let rows = parse(&format!(
            "{HEADER}\n{}",
            [5, 4, 4, 3, 2, 1]
                .iter()
                .enumerate()
                .map(|(i, target)| format!("d\ts\tc\tp{i}\t{target}\tdistortion\t1\t2"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
        .unwrap();
        let refs: Vec<_> = rows.iter().collect();
        let bands = quality_bands(&refs, 3);
        let identities: Vec<Vec<_>> = bands
            .iter()
            .map(|band| band.iter().map(|r| r.pair.as_str()).collect())
            .collect();
        assert_eq!(
            identities,
            [vec!["p0", "p1", "p2"], vec!["p3"], vec!["p4", "p5"]]
        );
        assert_eq!(bands.iter().map(Vec::len).sum::<usize>(), rows.len());
        let mut out = Vec::new();
        panel(&mut out, "quality_band", "d", "empty", &[]).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("\t0\tunavailable"));
    }

    fn input(body: &str) -> String {
        format!("{HEADER}\n{body}")
    }

    #[test]
    fn supplied_sigma_is_explicit_and_missing_values_are_not_dropped() {
        let header = format!("{HEADER}\tsigma\n");
        for invalid in ["0", "-1", "NaN", "inf"] {
            assert!(parse(&format!("{header}d\ts\tc\tp\t1\tquality\t2\t3\t{invalid}")).is_err());
        }
        let body = (0..6)
            .map(|i| {
                format!(
                    "d\ts\tc\tp{i}\t{}\tquality\t{}\t{}\t1",
                    i + 1,
                    6 - i,
                    [6, 4, 5, 2, 3, 1][i]
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut rows = parse(&(header + &body)).unwrap();
        let mut first = Vec::new();
        published_sigma_panel(&mut first, &rows).unwrap();
        let first = String::from_utf8(first).unwrap();
        let z: f64 = first
            .lines()
            .last()
            .unwrap()
            .split('\t')
            .next_back()
            .unwrap()
            .parse()
            .unwrap();
        assert!(z > 0.0);
        for row in &mut rows {
            row.sigma = Some(2.0);
        }
        let mut twice = Vec::new();
        published_sigma_panel(&mut twice, &rows).unwrap();
        let twice = String::from_utf8(twice).unwrap();
        let z_twice: f64 = twice
            .lines()
            .last()
            .unwrap()
            .split('\t')
            .next_back()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(z, 2.0 * z_twice);
        rows[0].sigma = None;
        let mut missing = Vec::new();
        published_sigma_panel(&mut missing, &rows).unwrap();
        assert!(
            String::from_utf8(missing)
                .unwrap()
                .contains("\t6\t5\tunavailable\tNA\tNA")
        );
    }

    #[test]
    fn rejects_bad_alignment_and_nonfinite_values() {
        for body in [
            "a\ts\tc\tp\t1\tquality\t2\t3\na\ts\tc\tp\t2\tquality\t3\t4",
            "a\ts\tc\tp\tNaN\tquality\t2\t3",
            "a\ts\tc\tp\t1\tquality\t2\t-3",
            "a\ts\tc\tp\t1\tquality\t2\t3\na\ts\tc\tq\t2\tdistortion\t3\t4",
        ] {
            assert!(parse(&input(body)).is_err());
        }
    }

    #[test]
    fn does_not_hide_reversed_polarity() {
        let rows=parse(&input("a\ts\tc\tp1\t1\tquality\t4\t1\na\ts\tc\tp2\t2\tquality\t3\t2\na\ts\tc\tp3\t3\tquality\t2\t3\na\ts\tc\tp4\t4\tquality\t1\t4")).unwrap();
        let refs: Vec<_> = rows.iter().collect();
        assert_eq!(
            orders(&refs, 0.0),
            Orders {
                decisive: 6,
                reversed: 6,
                collapsed: 0,
                teacher_ties: 0
            }
        );
        let mut out = Vec::new();
        panel(&mut out, "corpus", "a", "all", &refs).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("candidate\t4\tmeasured\t-1\t1\t"));
    }

    #[test]
    fn constants_are_unavailable_and_ties_stay_visible() {
        let rows = parse(&input(
            "a\ts\tc\tp1\t1\tdistortion\t1\t2\na\ts\tc\tp2\t2\tdistortion\t2\t2",
        ))
        .unwrap();
        assert_eq!(rows[0].target, -1.0);
        let refs: Vec<_> = rows.iter().collect();
        assert_eq!(
            orders(&refs, 0.0),
            Orders {
                decisive: 1,
                reversed: 0,
                collapsed: 1,
                teacher_ties: 0
            }
        );
        assert_eq!(orders(&refs, 1.0).teacher_ties, 1);
        let mut out = Vec::new();
        panel(&mut out, "corpus", "a", "all", &refs).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("unavailable"));
    }
}
