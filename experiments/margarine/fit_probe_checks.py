"""Explicit training-dependency checks: python -m unittest fit_probe_checks.

Run with the interpreter containing requirements-training.txt, not the stdlib-only
manifest-test interpreter. The justfile exposes this as margarine-fit-check.
"""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from fit_probe import fit_nonnegative, read_splits, source_weights, transform


class FitContract(unittest.TestCase):
    def test_source_balancing_and_exact_linear_recovery(self):
        x = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 2.], [2., 2.]])
        sources = ["a", "a", "a", "b", "b", "c"]
        weights = source_weights(sources)
        for source in set(sources):
            self.assertAlmostEqual(sum(w for s, w in zip(sources, weights) if s == source), 1 / 3)
        expected = np.array([2., 3.])
        actual = fit_nonnegative(x, x @ expected, sources, 1e-8)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        self.assertTrue((actual >= 0).all())
        self.assertEqual(float(np.expm1(transform([0., 0.], [1., 2.]) @ actual)), 0.)

    def test_bad_numeric_inputs_fail(self):
        for values, scales in [([-1.], [1.]), ([np.nan], [1.]), ([1.], [0.])]:
            with self.assertRaises(ValueError):
                transform(values, scales)

    def test_source_partition_is_explicit_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.tsv"
            path.write_text("source\tsplit\na\tfit\nb\ttune\nc\ttest\n")
            self.assertEqual(read_splits(path), dict(a="fit", b="tune", c="test"))
            path.write_text(path.read_text() + "a\ttest\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_splits(path)


if __name__ == "__main__":
    unittest.main()
