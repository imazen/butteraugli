import json
from pathlib import Path
import tempfile
import unittest

from evaluate_manifest import load_scores
from score_manifest import digest


class PersistedScoreTests(unittest.TestCase):
    def test_replay_requires_complete_unchanged_aligned_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / 'cells.jsonl'
            ledger.write_text(json.dumps(dict(dataset='d', pair='p', scores=dict(teacher={}, lattice={}))) + '\n')
            manifest = dict(status='scores-complete', n_pairs=1, cells_sha256=digest(ledger))
            path = root / '_MANIFEST.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(load_scores(root, 'lattice')[0]), 1)
            with self.assertRaisesRegex(ValueError, 'missing scoring arm'):
                load_scores(root, 'other')
            manifest['n_pairs'] = 2
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'count or identities'):
                load_scores(root, 'lattice')
            ledger.write_text(ledger.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_scores(root, 'lattice')
            manifest['status'] = 'running'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_scores(root, 'lattice')
