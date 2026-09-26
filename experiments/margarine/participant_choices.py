"""Join fixed encoder choices to saved participant bootstrap draws."""
import csv
import hashlib
import math
from pathlib import Path
import struct


def load(directory, rows):
    directory = Path(directory)
    with (directory / 'bootstrap.tsv').open() as f:
        meta = list(csv.DictReader(f, delimiter='\t'))
    with (directory / 'images.tsv').open() as f:
        images = list(csv.DictReader(f, delimiter='\t'))
    if len(meta) != 1 or len(images) != len(rows) or int(meta[0]['images']) != len(rows):
        raise ValueError('participant panel coverage differs from choice cohort')
    meta = meta[0]
    draws = int(meta['draws'])
    radius = float(meta['simultaneous_95_radius'])
    if draws < 100 or not math.isfinite(radius) or radius < 0:
        raise ValueError('invalid participant uncertainty metadata')
    indexed = {row['pair']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate choice identities')
    raw = (directory / 'participant-means.f64le').read_bytes()
    if len(raw) != len(rows) * draws * 8:
        raise ValueError('participant draw payload size differs')
    result = {}
    for i, image in enumerate(images):
        row = indexed.get(image['pair'])
        if (row is None or int(image['index']) != i or image['pair'] in result
                or image['source'] != row['source'] or row['direction'] != 'quality'
                or not math.isclose(float(image['mean']), float(row['target']), abs_tol=1e-9, rel_tol=0)):
            raise ValueError('participant identity or mean differs from scored choice')
        values = struct.unpack_from(f'<{draws}d', raw, i * draws * 8)
        if any(not math.isfinite(v) for v in values):
            raise ValueError('nonfinite participant bootstrap mean')
        result[image['pair']] = values
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ['bootstrap.tsv', 'images.tsv', 'participant-means.f64le', 'method.txt']}
    return dict(values=result, radius=radius, hashes=hashes,
                method=(directory / 'method.txt').read_text().strip())


def bounds(choice, panel):
    a, b = choice['teacher_pair'], choice['candidate_pair']
    values = sorted(x - y for x, y in zip(panel['values'][a], panel['values'][b]))
    n = len(values) - 1
    low, high = values[math.floor(0.025 * n)], values[math.ceil(0.975 * n)]
    # Identical choices have exactly zero loss; the common-radius envelope is
    # needed only for distinct images from the declared within-source family.
    radius = panel['radius'] if a != b else 0.0
    loss = choice['human_quality_loss']
    return dict(pointwise_p025=low, pointwise_p975=high,
                simultaneous_lower=loss - radius, simultaneous_upper=loss + radius,
                pointwise_95_harm=int(low > 0), simultaneous_95_harm=int(loss - radius > 0))
