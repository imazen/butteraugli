import unittest
from resource_sweep import rss_bytes


class PeakTests(unittest.TestCase):
    def test_platform_units(self):
        self.assertEqual(rss_bytes("   16384  maximum resident set size\n", "Darwin"), 16384)
        self.assertEqual(rss_bytes("Maximum resident set size (kbytes): 16384\n", "Linux"), 16777216)

    def test_missing_or_multiple_peaks_fail(self):
        with self.assertRaises(ValueError): rss_bytes("nothing", "Darwin")
        with self.assertRaises(ValueError):
            rss_bytes("1 maximum resident set size\n2 maximum resident set size\n", "Darwin")
