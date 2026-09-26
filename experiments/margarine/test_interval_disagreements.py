import unittest

from interval_disagreements import compare


class IntervalDisagreements(unittest.TestCase):
    def test_harm_benefit_overlap_and_source_boundaries(self):
        def row(pair, teacher, candidate, codec='a', source='s'):
            return dict(pair=pair, source=source, codec=codec,
                        scores=dict(teacher=dict(max=teacher), trial=dict(max=candidate)))
        labels = dict(a=dict(target=1., lower=.9, upper=1.1),
                      b=dict(target=2., lower=1.9, upper=2.1))
        a, b = row('a', 1, 2), row('b', 2, 1, 'b')
        result, records = compare([a, b], labels, 'trial', 'max')
        self.assertEqual(result['separated_harm'], 1)
        self.assertEqual(result['cross_codec_pairs'], 1)
        self.assertGreater(records[0]['human_loss'], 0)
        result, _ = compare([row('a', 2, 1), row('b', 1, 2)], labels, 'trial', 'max')
        self.assertEqual(result['separated_benefit'], 1)
        labels['b']['lower'] = 1.
        result, _ = compare([a, b], labels, 'trial', 'max')
        self.assertEqual(result['point_harm'], 1)
        self.assertEqual(result['separated_harm'], 0)
        b['source'] = 'other'
        result, records = compare([a, b], labels, 'trial', 'max')
        self.assertEqual(result['pairs'], 0)
        self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()
