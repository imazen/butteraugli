#!/usr/bin/env python3
"""Locate encoder-choice disagreements without treating point labels as significance.

Collapse repeated budgets choosing the same pair. Optional native-map probes
separate a global score shift from underestimation at Butteraugli's peak pixel.
"""
import argparse
import array
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from choice_eval import NORMS, choices


def unique_choices(rows, candidate, norm):
    grouped = {}
    for result in choices(rows, candidate, norm, human=True):
        if not result['different_pair']:
            continue
        key = result['teacher_pair'], result['candidate_pair']
        if key not in grouped:
            grouped[key] = dict(result, first_budget=result['budget_bpp'],
                                last_budget=result['budget_bpp'], budget_count=0)
        record = grouped[key]
        record['last_budget'] = result['budget_bpp']
        record['budget_count'] += 1
    return list(grouped.values())


def read_map(root, score):
    data = (root / (score['diffmap_sha256'] + '.f32le')).read_bytes()
    if hashlib.sha256(data).hexdigest() != score['diffmap_sha256']:
        raise ValueError('map hash mismatch')
    if len(data) != 4 * score['width'] * score['height']:
        raise ValueError('map geometry mismatch')
    values = array.array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('invalid native map')
    return values


def peak_probe(row, candidate, teacher_root, candidate_root):
    teacher = row['scores']['teacher']
    student = row['scores'][candidate]
    if (teacher['width'], teacher['height']) != (student['width'], student['height']):
        raise ValueError('teacher/candidate map geometry differs')
    a, b = read_map(teacher_root, teacher), read_map(candidate_root, student)
    peak = max(range(len(a)), key=a.__getitem__)
    other = max(range(len(b)), key=b.__getitem__)
    scale = student['p1'] / teacher['p1'] if teacher['p1'] else None
    return dict(dataset=row['dataset'], source=row['source'], pair=row['pair'],
                codec=row['codec'], teacher_peak=a[peak], candidate_at_teacher_peak=b[peak],
                candidate_peak=b[other], p1_scale=scale,
                peak_after_global_scale=(b[peak] / (a[peak] * scale)
                                         if a[peak] and scale else None),
                teacher_peak_x=peak % teacher['width'], teacher_peak_y=peak // teacher['width'],
                candidate_peak_x=other % teacher['width'], candidate_peak_y=other // teacher['width'],
                teacher_map_sha256=teacher['diffmap_sha256'],
                candidate_map_sha256=student['diffmap_sha256'])


def write_tsv(path, rows, fields):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledger', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--build-commit', required=True)
    parser.add_argument('--teacher-maps', type=Path)
    parser.add_argument('--candidate-maps', type=Path)
    parser.add_argument('--map-limit-per-tail', type=int, default=5)
    args = parser.parse_args()
    if bool(args.teacher_maps) != bool(args.candidate_maps) or args.map_limit_per_tail < 1:
        parser.error('provide both map roots and a positive tail limit')
    groups, lookup = defaultdict(list), {}
    for line in args.ledger.open():
        row = json.loads(line)
        key = row['dataset'], row['source'], row['pair']
        if key in lookup:
            raise ValueError('duplicate dataset/source/pair')
        lookup[key] = row
        groups[key[:2]].append(row)
    if not lookup:
        raise ValueError('empty ledger')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'progress.log').open('x', buffering=1) as log:
        def report(message):
            print(message, flush=True)
            print(message, file=log, flush=True)
        records, selected = [], set()
        for norm in NORMS:
            current = []
            for (dataset, source), rows in sorted(groups.items()):
                current.extend(dict(dataset=dataset, source=source, norm=norm, **r)
                               for r in unique_choices(rows, args.candidate, norm))
            current.sort(key=lambda r: r['human_quality_loss'], reverse=True)
            records.extend(current)
            for record in current[:args.map_limit_per_tail] + current[-args.map_limit_per_tail:]:
                for key in ['teacher_pair', 'candidate_pair']:
                    selected.add((record['dataset'], record['source'], record[key]))
            report(f'{norm}: {len(current)} distinct changed-choice pairs')
        fields = (list(records[0]) if records else
                  ['dataset', 'source', 'norm', 'teacher_pair', 'candidate_pair', 'budget_count'])
        write_tsv(args.output / 'choices.tsv', records, fields)
        probes = []
        if args.teacher_maps:
            for index, key in enumerate(sorted(selected)):
                probes.append(peak_probe(lookup[key], args.candidate,
                                         args.teacher_maps, args.candidate_maps))
                report(f'map {index + 1}/{len(selected)}: {key[2]}')
            write_tsv(args.output / 'peak-probes.tsv', probes,
                      list(probes[0]) if probes else ['dataset', 'source', 'pair'])
        provenance = dict(build_commit=args.build_commit, candidate=args.candidate,
                          ledger=str(args.ledger),
                          ledger_sha256=hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
                          distinct_choices=len(records), map_probes=len(probes),
                          tail_selection='largest and smallest signed point-label loss per norm',
                          map_limit_per_tail=args.map_limit_per_tail,
                          significance='not tested; participant uncertainty is not in this ledger')
        (args.output / '_MANIFEST.json').write_text(json.dumps(provenance, indent=2) + '\n')
        report('Complete; diagnostic disagreements, no significance verdict')


if __name__ == '__main__':
    main()
