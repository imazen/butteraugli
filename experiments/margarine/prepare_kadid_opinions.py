#!/usr/bin/env python3
"""Audit KADID raw ratings against published rounded DMOS and variance.

Retain only image/rating/pseudonymous-worker fields and original image links.
Location, IP address, and other crowd-platform fields are not copied.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


# Verified across 81 references × 5 levels for every distortion: raw export
# identifiers predate the published distortion ordering. Every mapped mean and
# population standard deviation agrees with the rounded published table.
RAW_TO_PUBLISHED = (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17, 8,
                    19, 21, 24, 22, 15, 23, 20, 7, 18, 25)


def image_name(url):
    match = re.fullmatch(r'i(\d+)_(\d+)_(\d+)\.png', url.rsplit('/', 1)[-1], re.I)
    if not match:
        raise ValueError(f'unrecognized KADID image: {url}')
    image, distortion, level = map(int, match.groups())
    if not 1 <= distortion <= len(RAW_TO_PUBLISHED):
        raise ValueError('raw distortion identifier out of range')
    return 'I%02d_%02d_%02d.png' % (image, RAW_TO_PUBLISHED[distortion - 1], level)


def agrees_with_rounding(value, published):
    rounded = Decimal(published)
    half_unit = Decimal(10) ** rounded.as_tuple().exponent / 2
    return abs(value - rounded) <= half_unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('dmos', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--build-commit', required=True)
    args = parser.parse_args()
    labels = {r['dist_img']: r for r in csv.DictReader(args.dmos.open())}
    if len(labels) != 10125:
        raise ValueError('expected all 10125 published KADID labels')
    args.output.mkdir(parents=True, exist_ok=False)
    ratings = defaultdict(list)
    counts, identities = Counter(), set()
    duplicate_worker_image = 0
    with (args.output / 'progress.log').open('x', buffering=1) as log:
        def report(message):
            print(message, file=log, flush=True)
            print(message, flush=True)
        with (args.output / 'opinions.tsv').open('x', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['image', 'worker', 'rating', 'dist_url', 'ref_url'], delimiter='\t')
            writer.writeheader()
            for index, row in enumerate(csv.DictReader(args.raw.open())):
                prefix = row['dist_url'].rsplit('/', 1)[0]
                counts['raw_rows'] += 1
                if prefix == 'tid_2013_png':
                    counts['tid_control_rows'] += 1
                    continue
                if prefix != 'kon10k_png':
                    raise ValueError(f'unknown raw-rating dataset prefix: {prefix}')
                if row['_golden'] not in ('true', 'false') or row['_tainted'] not in ('true', 'false'):
                    raise ValueError('unknown crowd-platform eligibility flag')
                if row['_golden'] == 'true' or row['_tainted'] == 'true':
                    counts['excluded_gold_or_tainted'] += 1
                    continue
                name = image_name(row['dist_url'])
                if name not in labels:
                    raise ValueError('opinion image not in published labels')
                rating = int(row['dcr'])
                if not 1 <= rating <= 5:
                    raise ValueError('DCR outside [1,5]')
                worker = hashlib.sha256(('KADID-10k/worker/' + row['_worker_id']).encode()).hexdigest()
                duplicate_worker_image += (name, worker) in identities
                identities.add((name, worker))
                ratings[name].append(rating)
                writer.writerow(dict(image=name, worker=worker, rating=rating,
                                     dist_url=row['dist_url'], ref_url=row['ref_url']))
                if (index + 1) % 50000 == 0:
                    report(f'audited {index + 1} raw rows')
        audits = []
        for name, published in sorted(labels.items()):
            values = ratings[name]
            if not values:
                raise ValueError(f'no eligible opinions for {name}')
            n = len(values)
            mean = Decimal(sum(values)) / n
            squared = sum((Decimal(v) - mean) ** 2 for v in values)
            population = squared / n
            sample = squared / (n - 1) if n > 1 else Decimal('NaN')
            audits.append(dict(image=name, n=n, mean=float(mean),
                               published_dmos=published['dmos'], published_variance=published['var'],
                               population_variance=float(population), sample_variance=float(sample),
                               population_std=float(population.sqrt()),
                               population_std_matches=agrees_with_rounding(population.sqrt(), published['var']),
                               mean_matches=agrees_with_rounding(mean, published['dmos']),
                               population_variance_matches=agrees_with_rounding(population, published['var']),
                               sample_variance_matches=agrees_with_rounding(sample, published['var'])))
        with (args.output / 'label-audit.tsv').open('x', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(audits[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(audits)
        mean_misses = sum(not r['mean_matches'] for r in audits)
        sample_misses = sum(not r['sample_variance_matches'] for r in audits)
        population_misses = sum(not r['population_variance_matches'] for r in audits)
        std_misses = sum(not r['population_std_matches'] for r in audits)
        qualified = (not mean_misses and not duplicate_worker_image and
                     all(r['n'] == 30 for r in audits) and not min(sample_misses, population_misses, std_misses))
        manifest = dict(build_commit=args.build_commit, source=str(args.raw),
                        raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
                        dmos_sha256=hashlib.sha256(args.dmos.read_bytes()).hexdigest(),
                        opinions_sha256=hashlib.sha256((args.output / 'opinions.tsv').read_bytes()).hexdigest(),
                        status='matches-published-labels' if qualified else 'requires-reconciliation',
                        counts=dict(counts), n_images=len(ratings),
                        ratings_per_image=dict(Counter(r['n'] for r in audits)),
                        mean_mismatches=mean_misses, sample_variance_mismatches=sample_misses,
                        population_variance_mismatches=population_misses,
                        population_std_mismatches=std_misses,
                        all_label_means_and_std_match=not mean_misses and not std_misses,
                        raw_to_published_distortion_order=RAW_TO_PUBLISHED,
                        duplicate_worker_image=duplicate_worker_image,
                        eligibility='kon10k_png rows; exclude golden or tainted observations',
                        rounding='half the last decimal place printed in each published label',
                        source_url='https://database.mmsp-kn.de/kadid-10k-database.html')
        (args.output / '_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        report(json.dumps({k: manifest[k] for k in ['status', 'n_images', 'ratings_per_image',
                                                   'mean_mismatches', 'sample_variance_mismatches',
                                                   'population_variance_mismatches', 'population_std_mismatches', 'duplicate_worker_image']}))


if __name__ == '__main__':
    main()
