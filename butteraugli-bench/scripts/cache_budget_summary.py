#!/usr/bin/env python3
"""Summarize completed cache-budget timings and separate-process RSS logs."""
import csv
import io
import re
import sys
from pathlib import Path
from extract_version_audit import extract

BUDGETS = ['default', '0', '32', '64', '128', '256', '512', '1024']
SIZES = ['512x512', '1920x1080', '3840x2160']


def write_csv(path, fields, rows):
    data = io.StringIO()
    writer = csv.DictWriter(data, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    assert len(data.getvalue().encode()) <= 30_000, 'store larger generated data outside git'
    path.write_text(data.getvalue())


def main():
    directory, prefix = map(Path, sys.argv[1:3])
    paired = []
    for budget in BUDGETS:
        rows = list(csv.DictReader(io.StringIO(extract(directory / f'butteraugli-cache-budget-{budget}.log'))))
        old = {r['group']: r for r in rows if r['benchmark'] == '0.9.3'}
        current = [r for r in rows if r['benchmark'] == 'main']
        assert len(current) == 6, f'incomplete paired run for {budget}'
        for r in current:
            paired.append(dict(budget_mib=budget, case=r['group'], published_mean_ns=old[r['group']]['mean_ns'],
                               current_mean_ns=r['mean_ns'], change_pct=r['vs_base_pct'],
                               ci_lo_pct=r['vs_base_ci_lo_pct'], ci_hi_pct=r['vs_base_ci_hi_pct']))
    write_csv(Path(f'{prefix}_paired.csv'), list(paired[0]), paired)
    if "--paired-only" in sys.argv[3:]:
        return
    profiles = []
    for budget in ['published', *BUDGETS]:
        for size in SIZES:
            log = (directory / f'butteraugli-cache-rss-{budget}-{size}.log').read_text()
            metrics = re.search(r'PROFILE version=(\w+) size=(\S+) threads=(\d+) calls=(\d+) wall_ms=([\d.]+) score=([\d.]+) pnorm_3=([\d.]+)', log)
            assert metrics, f'missing profile for {budget}/{size}'
            version, actual_size, threads, calls, wall, score, pnorm = metrics.groups()
            assert actual_size == size
            rss = re.search(r'Maximum resident set size \(kbytes\): (\d+)', log)
            assert rss, 'RSS not measured'
            retained = re.search(r'RETAINED bytes=(\d+) precompute_bytes=(\d+)', log)
            profiles.append(dict(budget_mib=budget, version=version, size=size, threads=threads, calls=calls,
                                 batch_wall_ms=wall, peak_rss_kib=rss[1], retained_bytes=retained[1] if retained else '',
                                 precompute_bytes=retained[2] if retained else '', score=score, pnorm_3=pnorm))
    for size in SIZES:
        matching = [r for r in profiles if r['size'] == size]
        assert len({(r['score'], r['pnorm_3']) for r in matching}) == 1, f'profile scores changed at {size}'
    write_csv(Path(f'{prefix}_profiles.csv'), list(profiles[0]), profiles)


if __name__ == '__main__':
    main()
