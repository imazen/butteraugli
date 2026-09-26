#!/usr/bin/env python3
"""Join audited KADID opinions to frozen scores for participant uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from evaluate_manifest import load_scores
from score_manifest import FIELDS, NORMS, digest, evaluate_panels


def live1(args, cells, scored):
    opinion_manifest = args.opinions / '_MANIFEST.json'
    opinions = json.loads(opinion_manifest.read_text())
    audit_path = Path(scored['input_audit'])
    if digest(audit_path) != scored['input_audit_sha256']:
        raise ValueError('scored input audit has changed')
    audit = json.loads(audit_path.read_text())
    if (opinions['status'] != 'images-audited' or opinions['labels'] != audit['labels']
            or opinions['pairs_sha256'] != scored['pairs_sha256']
            or opinions['n_pairs'] != len(cells)):
        raise ValueError('LIVE opinion and score provenance differ')
    cohorts = opinions['cohorts']
    if len(cohorts) != 4 or set(cohorts) != {r['dataset'] for r in cells}:
        raise ValueError('requires all four declared LIVE Release 1 cohorts')
    args.output.mkdir(parents=True, exist_ok=False)
    participants = args.output / 'participants'
    participants.mkdir()
    manifest = dict(build_commit=args.build_commit, candidate=args.candidate,
                    scored_build_commit=scored['build_commit'], cells_sha256=scored['cells_sha256'],
                    opinions_manifest_sha256=digest(opinion_manifest),
                    evaluator_sha256=digest(args.evaluator), draws=args.draws, seed=args.seed,
                    scope='within each codec/session; no cross-cohort calibration',
                    uncertainty='conditional on published normalization and outlier selection',
                    status='running')
    path = args.output / '_MANIFEST.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'progress.log').open('x', buffering=1) as progress:
        def report(message):
            print(message, flush=True)
            print(message, file=progress, flush=True)
        for dataset, metadata in cohorts.items():
            rows = [r for r in cells if r['dataset'] == dataset]
            opinion_path = args.opinions / f'opinions-{dataset}.tsv'
            if len(rows) != metadata['pairs'] or digest(opinion_path) != metadata['opinions_sha256']:
                raise ValueError('LIVE cohort coverage or opinion hash differs')
            source = args.output / dataset
            source.mkdir()
            evaluate_panels(rows, args.candidate, args.evaluator.resolve(), source, report)
            report(f'{dataset}: starting {args.draws} observer-column draws')
            with (args.output / f'{dataset}-bootstrap.log').open('x') as log:
                subprocess.run([str(args.evaluator.resolve()), '--live1-participant-pairs',
                                str(source), str(opinion_path), str(participants / dataset),
                                str(args.draws), str(args.seed)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            report(f'{dataset}: participant bounds persisted')
        report('Evaluating fixed choices at every observed bitrate budget')
        with (args.output / 'choices.log').open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('choice_eval.py')),
                            str(args.scored / 'cells.jsonl'), str(args.output / 'choices'),
                            '--candidate', args.candidate, '--build-commit', args.build_commit,
                            '--participant-panels', str(participants), '--human-loss-threshold', '0'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest['status'] = 'complete'
        manifest['outputs'] = {str(p.relative_to(args.output)): digest(p)
                               for p in sorted(args.output.rglob('*'))
                               if p.is_file() and p.name != '_MANIFEST.json' and p.name != 'progress.log'}
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        report('Complete: cohort-scoped quality and participant-supported bitrate-choice panels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scored', type=Path)
    parser.add_argument('opinions', type=Path)
    parser.add_argument('evaluator', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--build-commit', required=True)
    parser.add_argument('--draws', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--live1', action='store_true', help='separate published Release 1 observer cohorts')
    args = parser.parse_args()
    if args.draws < 100 or not 0 <= args.seed < 2**64:
        parser.error('at least 100 draws and an unsigned 64-bit seed are required')
    cells, scored = load_scores(args.scored, args.candidate)
    if args.live1:
        return live1(args, cells, scored)
    opinions = json.loads((args.opinions / '_MANIFEST.json').read_text())
    opinion_path = args.opinions / 'opinions.tsv'
    if not opinions['all_label_means_and_std_match'] or digest(opinion_path) != opinions['opinions_sha256']:
        raise ValueError('raw opinions are not reconciled to published labels')
    if opinions['n_images'] != len(cells) or any(r['dataset'] != 'kadid' or r['direction'] != 'quality' for r in cells):
        raise ValueError('requires the complete quality-oriented KADID score set')
    audit_path = Path(scored['input_audit'])
    if digest(audit_path) != scored['input_audit_sha256']:
        raise ValueError('scored input audit has changed')
    audit = json.loads(audit_path.read_text())
    if audit['labels']['dmos.csv'] != opinions['dmos_sha256']:
        raise ValueError('scores and raw opinions use different published labels')
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / 'scores'
    source.mkdir()
    for norm in NORMS:
        with (source / f'scores-{norm}.tsv').open('x', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=FIELDS[:6] + ['teacher', 'candidate'], delimiter='\t')
            writer.writeheader()
            for cell in cells:
                writer.writerow(dict(**{k:cell[k] for k in FIELDS[:6]},
                                     teacher=cell['scores']['teacher'][norm],
                                     candidate=cell['scores'][args.candidate][norm]))
    manifest = dict(build_commit=args.build_commit, candidate=args.candidate,
                    scored_build_commit=scored['build_commit'], cells_sha256=scored['cells_sha256'],
                    opinions_sha256=opinions['opinions_sha256'],
                    opinions_manifest_sha256=digest(args.opinions / '_MANIFEST.json'),
                    evaluator_sha256=digest(args.evaluator), draws=args.draws, seed=args.seed,
                    status='running', bootstrap='resample workers with replacement; retain all their observations',
                    scope='fixed images; all within-source pairs; not matched-rate encoder choices',
                    pointwise='central 95% percentile intervals, no multiplicity adjustment',
                    simultaneous='95th percentile of maximum absolute centered within-source pair error',
                    sample_layout='participant-means.f64le: image-major then draw; order in images.tsv',
                    methodology_url='https://doi.org/10.1111/j.1467-9868.2007.00593.x')
    path = args.output / '_MANIFEST.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'run.log').open('x') as log:
        subprocess.run([str(args.evaluator.resolve()), '--participant-pairs', str(source),
                        str(opinion_path), str(args.output / 'bootstrap'), str(args.draws), str(args.seed)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest['status'] = 'complete'
    manifest['outputs'] = {str(p.relative_to(args.output)):digest(p)
                           for p in sorted((args.output / 'bootstrap').iterdir()) if p.is_file()}
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Participant disagreement intervals complete; encoder-choice gate remains separate', flush=True)


if __name__ == '__main__':
    main()
