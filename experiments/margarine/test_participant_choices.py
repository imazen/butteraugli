import unittest
from pathlib import Path
import tempfile
import struct

from participant_choices import bounds, load


class ChoiceUncertainty(unittest.TestCase):
    def test_same_choice_is_exactly_zero_even_with_wide_family_bounds(self):
        panel = dict(values={'a': [1., 2., 3., 4.] * 25}, radius=10.)
        result = bounds(dict(teacher_pair='a', candidate_pair='a', human_quality_loss=0.), panel)
        self.assertEqual(result['pointwise_p025'], 0.)
        self.assertEqual(result['simultaneous_lower'], 0.)
        self.assertEqual(result['pointwise_95_harm'], 0)

    def test_pairing_retains_shared_observer_fluctuations_and_loss_direction(self):
        panel = dict(values={'a': [float(i) + 2 for i in range(100)],
                             'b': [float(i) for i in range(100)]}, radius=3.)
        result = bounds(dict(teacher_pair='a', candidate_pair='b', human_quality_loss=2.), panel)
        self.assertEqual(result['pointwise_p025'], 2.)
        self.assertEqual(result['pointwise_p975'], 2.)
        self.assertEqual(result['pointwise_95_harm'], 1)
        self.assertEqual(result['simultaneous_95_harm'], 0)
        reverse = bounds(dict(teacher_pair='b', candidate_pair='a', human_quality_loss=-2.), panel)
        self.assertEqual(reverse['pointwise_p975'], -2.)
        self.assertEqual(reverse['pointwise_95_harm'], 0)

    def test_loader_requires_complete_matching_finite_draws(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            p = Path(directory)
            (p / 'bootstrap.tsv').write_text('images\tdraws\tsimultaneous_95_radius\n2\t100\t1\n')
            identities = 'index\tsource\tpair\tmean\n0\ts\ta\t10\n1\ts\tb\t20\n'
            (p / 'images.tsv').write_text(identities)
            (p / 'method.txt').write_text('conditional fixture')
            draws = struct.pack('<200d', *([10.] * 100 + [20.] * 100))
            (p / 'participant-means.f64le').write_bytes(draws)
            rows = [dict(pair=k, source='s', direction='quality', target=v) for k,v in [('a',10.), ('b',20.)]]
            panel = load(p, rows)
            self.assertEqual(panel['values']['a'], (10.,) * 100)
            (p / 'images.tsv').write_text(identities.replace('a\t10', 'a\tNaN'))
            with self.assertRaisesRegex(ValueError, 'identity or mean'):
                load(p, rows)
            (p / 'images.tsv').write_text(identities)
            (p / 'participant-means.f64le').write_bytes(draws[:-8])
            with self.assertRaisesRegex(ValueError, 'payload size'):
                load(p, rows)
