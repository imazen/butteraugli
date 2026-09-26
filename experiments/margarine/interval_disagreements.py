#!/usr/bin/env python3
"""Audit AIC4 metric reversals against the published marginal JND intervals.

Non-overlap is reported literally. Marginal intervals alone do not supply a
paired-difference interval, simultaneous coverage, or encoded byte budgets.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from aic4_sample_manifest import LABEL_SHA, LABEL_URL
from evaluate_manifest import load_scores
from score_manifest import NORMS, digest


def compare(rows, labels, candidate, norm):
    summary = dict(pairs=0, cross_codec_pairs=0, teacher_ties=0,
                   candidate_ties=0, reversals=0, cross_codec_reversals=0,
                   point_harm=0, separated_harm=0, separated_benefit=0,
                   cross_codec_separated_harm=0)
    disagreements = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a['source'] != b['source']:
                continue
            cross = a['codec'] != b['codec']
            summary['pairs'] += 1
            summary['cross_codec_pairs'] += cross
            td = a['scores']['teacher'][norm] - b['scores']['teacher'][norm]
            cd = a['scores'][candidate][norm] - b['scores'][candidate][norm]
            summary['teacher_ties'] += td == 0
            summary['candidate_ties'] += cd == 0
            if td == 0 or cd == 0 or (td > 0) == (cd > 0):
                continue
            teacher, chosen = (a, b) if td < 0 else (b, a)
            ta, ca = labels[teacher['pair']], labels[chosen['pair']]
            loss = ca['target'] - ta['target']
            lower = ca['lower'] - ta['upper']
            upper = ca['upper'] - ta['lower']
            summary['reversals'] += 1
            summary['cross_codec_reversals'] += cross
            summary['point_harm'] += loss > 0
            summary['separated_harm'] += lower > 0
            summary['separated_benefit'] += upper < 0
            summary['cross_codec_separated_harm'] += cross and lower > 0
            disagreements.append(dict(norm=norm, source=a['source'],
                teacher_preferred=teacher['pair'], candidate_preferred=chosen['pair'],
                cross_codec=cross, human_loss=loss,
                candidate_lower_minus_teacher_upper=lower,
                candidate_upper_minus_teacher_lower=upper))
    return summary, disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scored', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--build-commit', required=True)
    args = parser.parse_args()
    if digest(args.labels) != LABEL_SHA:
        raise ValueError('requires the pinned original AIC4 JND/interval CSV')
    with args.labels.open() as file:
        labels = {r['img_distorted']: dict(target=float(r['distortion']),
            lower=float(r['CI_min']), upper=float(r['CI_max']),
            source=r['img_num'].zfill(5)) for r in csv.DictReader(file)}
    rows, source = load_scores(args.scored, args.candidate)
    if len(rows) != 300 or len(labels) != 300 or {r['pair'] for r in rows} != labels.keys():
        raise ValueError('requires all 300 AIC4 pairs with exact label identities')
    for row in rows:
        label = labels[row['pair']]
        if (row['dataset'], row['direction'], row['source'], float(row['target'])) != (
                'aic4_sample', 'distortion', label['source'], label['target']):
            raise ValueError('scored label identity or polarity differs')
        if not all(math.isfinite(label[k]) for k in ['lower', 'upper', 'target']) or label['lower'] > label['upper']:
            raise ValueError('invalid published interval')
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, records = [], []
    with (args.output / 'progress.log').open('x', buffering=1) as progress:
        for norm in NORMS:
            summary, pairs = compare(rows, labels, args.candidate, norm)
            summaries.append(dict(norm=norm, **summary))
            records.extend(pairs)
            print(f'{norm}: {summary}', file=progress, flush=True)
            print(f'{norm}: {summary}', flush=True)
    for name, data, fields in [
        ('summary.tsv', summaries, list(summaries[0])),
        ('disagreements.tsv', records, ['norm', 'source', 'teacher_preferred',
         'candidate_preferred', 'cross_codec', 'human_loss',
         'candidate_lower_minus_teacher_upper', 'candidate_upper_minus_teacher_lower'])]:
        with (args.output / name).open('x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(data)
    manifest = dict(build_commit=args.build_commit, candidate=args.candidate,
        scored_build_commit=source['build_commit'], cells_sha256=source['cells_sha256'],
        labels_sha256=LABEL_SHA, labels_url=LABEL_URL, status='complete',
        criterion='candidate JND lower bound exceeds teacher JND upper bound',
        limitation='published marginal 95% intervals; no paired interval or multiplicity claim; no matched byte budgets',
        outputs={p.name:digest(p) for p in args.output.glob('*.tsv')})
    (args.output / '_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
