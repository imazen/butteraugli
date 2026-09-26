#!/usr/bin/env python3
"""Audit LIVE Release 1's four separately normalized observer cohorts.

The published processed opinions retain their original outlier exclusions and
normalization. They support uncertainty conditional on that processing, not
cross-session calibration or Release 2 DMOS uncertainty.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

from cid22_manifest import audit
from score_manifest import FIELDS, digest


def checked_opinions(values, target, sigma):
    values = [float(v) for v in values]
    if any(not math.isfinite(v) or v < 0 or v > 100 for v in values):
        raise ValueError('invalid published processed opinion')
    # The release explicitly uses zero for excluded/skipped observations.
    kept = [(i, v) for i, v in enumerate(values) if v != 0]
    if len(kept) < 2:
        raise ValueError('insufficient published observer coverage')
    ratings = [v for _, v in kept]
    if not math.isclose(statistics.mean(ratings), target, abs_tol=1e-10, rel_tol=0):
        raise ValueError('published opinion mean differs from mmt')
    if not math.isclose(statistics.stdev(ratings), sigma, abs_tol=1e-10, rel_tol=0):
        raise ValueError('published opinion sample deviation differs from mst')
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path, help='contains jpeg/ and jpeg2000/ archive contents')
    p.add_argument('output', type=Path)
    p.add_argument('--destination-root', type=Path, required=True)
    p.add_argument('--build-commit', required=True)
    args = p.parse_args()
    from scipy.io import loadmat
    args.output.mkdir(parents=True, exist_ok=False)
    labels = args.output / 'labels'
    labels.mkdir()
    rows, images, sources, cohorts = [], {}, {}, {}
    with (args.output / 'progress.log').open('x', buffering=1) as log:
        def report(message):
            print(message, flush=True)
            print(message, file=log, flush=True)
        for codec, count, info_name in [('jpeg', 233, 'jpeginfo.txt'),
                                         ('jpeg2000', 227, 'jp2info.txt')]:
            root = args.root / codec
            info = [line.split() for line in (root / info_name).read_text().splitlines()]
            if len(info) != count or any(len(v) != 3 for v in info):
                raise ValueError('unexpected bitrate table geometry')
            for i, (_, name, rate) in enumerate(info, 1):
                if name != f'img{i}.bmp' or not math.isfinite(float(rate)) or float(rate) < 0:
                    raise ValueError('invalid image order or bitrate')
            for path in root.iterdir():
                if path.suffix.lower() in ('.txt', '.mat', '.m'):
                    dst = labels / codec / path.name
                    dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(path, dst)
                    sources[f'{codec}/{path.name}'] = digest(path)
            offset = 0
            for session, length in [(1, 116), (2, count - 116)]:
                dataset = f'live_r1_{codec}_s{session}'
                m = loadmat(root / f'scores{session}.mat')
                if any(m[k].shape != (1, length) for k in ['mmt', 'mst', 'br']):
                    raise ValueError('unexpected score vector geometry')
                if m['scores'].ndim != 2 or m['scores'].shape[0] != length:
                    raise ValueError('unexpected participant matrix geometry')
                opinions = []
                included = lossless = observations = 0
                for i in range(length):
                    ref, dist, rate = info[offset + i]
                    rate, target, sigma = float(rate), float(m['mmt'][0, i]), float(m['mst'][0, i])
                    if abs(rate - float(m['br'][0, i])) > 1e-12:
                        raise ValueError('bitrate table differs from published score metadata')
                    ratings = checked_opinions(m['scores'][i], target, sigma)
                    if rate == 0:
                        lossless += 1
                        continue  # documented lossless controls, not compressed candidates
                    pair = f'{codec}-s{session}-{dist}'
                    paths = []
                    for name in [ref, dist]:
                        key = f'{codec}/{name}'
                        path = root / name
                        if key not in images:
                            images[key] = dict(sha256=digest(path), bytes=path.stat().st_size,
                                               dimensions=list(audit(path)))
                        paths.append(str(args.destination_root / key))
                    if images[f'{codec}/{ref}']['dimensions'] != images[f'{codec}/{dist}']['dimensions']:
                        raise ValueError('reference and distortion dimensions differ')
                    rows.append(dict(zip(FIELDS, (dataset, ref, codec, pair, target, 'quality', *paths)),
                                     bpp=rate, setting='published bitrate'))
                    for column, rating in ratings:
                        worker = hashlib.sha256(f'{dataset}:processed-column:{column}'.encode()).hexdigest()
                        opinions.append((pair, worker, rating, f'{codec}/{dist}', f'{codec}/{ref}'))
                    included += 1
                    observations += len(ratings)
                with (args.output / f'opinions-{dataset}.tsv').open('x') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['image', 'worker', 'rating', 'dist_url', 'ref_url'])
                    writer.writerows(opinions)
                cohorts[dataset] = dict(pairs=included, lossless_controls=lossless,
                                        processed_workers=m['scores'].shape[1], observations=observations)
                report(f'{dataset}: {included} pairs, {observations} retained opinions; {lossless} lossless controls')
                offset += length
        if len(rows) != 344:
            raise ValueError('expected 169 JPEG and 175 JPEG2000 compressed stimuli')
        pairs = args.output / 'pairs.tsv'
        with pairs.open('x') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS + ['bpp', 'setting'], delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
        manifest = dict(build_commit=args.build_commit, status='images-audited', n_pairs=len(rows),
                        root=str(args.root.resolve()), destination_root=str(args.destination_root),
                        images=images, labels=sources, cohorts=cohorts, pairs_sha256=digest(pairs),
                        uncertainty='conditional on published normalization and outlier selection',
                        choice_scope='within each codec/session; no cross-cohort score alignment',
                        source='https://live.ece.utexas.edu/research/quality/subjective.htm')
        (args.output / '_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        report('Complete: all means, sample deviations, bitrates and image identities checked')


if __name__ == '__main__':
    main()
