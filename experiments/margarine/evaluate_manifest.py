#!/usr/bin/env python3
"""Evaluate persisted scores without decoding images or regenerating diffmaps."""
import argparse
import json
from pathlib import Path

from score_manifest import digest, evaluate_panels


def load_scores(directory, candidate):
    manifest = json.loads((directory / '_MANIFEST.json').read_text())
    if manifest['status'] not in ('scores-complete', 'complete'):
        raise ValueError('score generation is incomplete')
    path = directory / 'cells.jsonl'
    if digest(path) != manifest['cells_sha256']:
        raise ValueError('score ledger hash mismatch')
    with path.open() as ledger:
        cells = [json.loads(line) for line in ledger]
    if len(cells) != manifest['n_pairs'] or len({(r['dataset'], r['pair']) for r in cells}) != len(cells):
        raise ValueError('score ledger count or identities differ')
    if not cells or any(candidate not in r['scores'] or 'teacher' not in r['scores'] for r in cells):
        raise ValueError('missing scoring arm')
    return cells, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scored', type=Path)
    parser.add_argument('evaluator', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--build-commit', required=True)
    parser.add_argument('--dataset', help='explicitly select one dataset/cohort after checking the full ledger')
    parser.add_argument('--ordinary-only', action='store_true',
                        help='evaluate ordinary panels for every row; leave supplied-sigma panels explicitly unevaluated')
    args = parser.parse_args()
    cells, source = load_scores(args.scored, args.candidate)
    if args.dataset:
        cells = [row for row in cells if row['dataset'] == args.dataset]
        if not cells:
            raise ValueError('requested dataset is absent from scored ledger')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(build_commit=args.build_commit, candidate=args.candidate,
                    source=str(args.scored.resolve()), source_build_commit=source['build_commit'],
                    selected_dataset=args.dataset, source_n_pairs=source['n_pairs'],
                    cells_sha256=source['cells_sha256'],
                    evaluator_sha256=digest(args.evaluator), n_pairs=len(cells), status='running',
                    published_sigma='not requested' if args.ordinary_only else 'requested when supplied')
    path = args.output / '_MANIFEST.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'progress.log').open('x', buffering=1) as log:
        def report(message):
            print(message, flush=True)
            print(message, file=log, flush=True)
        evaluate_panels(cells, args.candidate, args.evaluator.resolve(), args.output, report,
                        published_sigma=not args.ordinary_only)
        manifest['status'] = 'complete'
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        report('Statistical panels complete; original scores and maps unchanged')


if __name__ == '__main__':
    main()
