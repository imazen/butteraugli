#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::{BufWriter, Write};
use zenstats::{LightPanel, ValAggregate, compute_panel, spearman};

const HEADER: &str = "dataset\tsource\tcodec\tpair\ttarget\tdirection\tteacher\tcandidate";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct Row {
    dataset: String,
    source: String,
    codec: String,
    target: f64,    // normalized polarity only; larger = better
    teacher: f64,   // raw distance; smaller = better
    candidate: f64, // raw distance; smaller = better
}

fn parse(input: &str) -> Result<Vec<Row>> {
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return Err(format!("expected header: {HEADER}").into());
    }
    let mut seen = BTreeSet::new();
    let mut orientations = BTreeMap::new();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 8 || fields.iter().any(|f| f.is_empty()) {
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
        rows.push(Row {
            dataset: fields[0].into(),
            source: fields[1].into(),
            codec: fields[2].into(),
            target: sign * target,
            teacher,
            candidate,
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

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
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

    fn input(body: &str) -> String {
        format!("{HEADER}\n{body}")
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
