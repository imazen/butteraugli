#!/usr/bin/env python3
"""Require the named command to reproduce a frozen scalar/map ledger exactly."""
import argparse
import json
from pathlib import Path
import subprocess

from evaluate_manifest import load_scores
from score_manifest import NORMS, digest


def verify(expected, actual, map_path):
    if actual.get('pooling') != 'max':
        raise ValueError('primary pooling differs')
    for key in ['width', 'height', *NORMS]:
        value = actual.get('score' if key == 'max' else key)
        if value != expected[key]:
            raise ValueError(f'CLI {key} differs: {value} != {expected[key]}')
    if map_path.stat().st_size != expected['width'] * expected['height'] * 4:
        raise ValueError('CLI map byte count differs')
    if digest(map_path) != expected['diffmap_sha256']:
        raise ValueError('CLI native map differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scored', type=Path)
    p.add_argument('binary', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--candidate', required=True)
    p.add_argument('--build-commit', required=True)
    args = p.parse_args()
    cells, source = load_scores(args.scored, args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    maps = args.output / 'maps'
    maps.mkdir()
    manifest = dict(build_commit=args.build_commit, frozen_build_commit=source['build_commit'],
                    candidate=args.candidate, frozen_cells_sha256=digest(args.scored/'cells.jsonl'),
                    binary_sha256=digest(args.binary), pairs=len(cells), verified=0, status='running')
    target = args.output / '_MANIFEST.json'
    target.write_text(json.dumps(manifest, indent=2)+'\n')
    checked = {}
    with (args.output/'progress.log').open('x', buffering=1) as log:
        for i, cell in enumerate(cells):
            for key in ['reference', 'distorted']:
                path = cell[key]
                if path not in checked:
                    checked[path] = digest(Path(path))
                if checked[path] != cell[key+'_sha256']:
                    raise ValueError(f'input changed: {path}')
            # Keep every produced map, including a failing one, for diagnosis.
            map_path = maps/f'{i}.f32le'
            command = [str(args.binary.resolve()), '--all-scores', '--diffmap', str(map_path),
                       cell['reference'], cell['distorted']]
            with (args.output/f'cell-{i}.json').open('x') as out, \
                    (args.output/f'cell-{i}.log').open('x') as errors:
                subprocess.run(command, stdout=out, stderr=errors, check=True)
            actual = json.loads((args.output/f'cell-{i}.json').read_text())
            verify(cell['scores'][args.candidate], actual, map_path)
            manifest['verified'] = i+1
            target.write_text(json.dumps(manifest, indent=2)+'\n')
            message = f'Exact CLI replay {i+1}/{len(cells)}: {cell["dataset"]} {cell["pair"]}'
            print(message, flush=True)
            print(message, file=log, flush=True)
    manifest['status'] = 'complete'
    target.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
