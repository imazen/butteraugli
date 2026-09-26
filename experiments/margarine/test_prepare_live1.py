import math
import statistics
import unittest

from prepare_live1 import checked_opinions


class PublishedLiveOpinions(unittest.TestCase):
    def test_preserves_column_identity_and_fractional_processed_scores(self):
        values = [12.5, 0, 31.75, 88.125]
        kept = [v for v in values if v]
        self.assertEqual(checked_opinions(values, statistics.mean(kept), statistics.stdev(kept)),
                         [(0, 12.5), (2, 31.75), (3, 88.125)])

    def test_rejects_population_deviation_and_unreconciled_means(self):
        values = [10, 20, 30]
        with self.assertRaisesRegex(ValueError, 'sample deviation'):
            checked_opinions(values, 20, statistics.pstdev(values))
        with self.assertRaisesRegex(ValueError, 'mean differs'):
            checked_opinions(values, 21, 10)

    def test_invalid_or_insufficient_opinions_fail_loudly(self):
        for values in [[0, 10], [0, 0], [-1, 10], [101, 10], [math.nan, 10]]:
            with self.assertRaises(ValueError):
                checked_opinions(values, 10, 0)
