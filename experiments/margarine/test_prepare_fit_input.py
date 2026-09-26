import unittest
from collections import Counter

from prepare_fit_input import partition_sources


class SourcePartitions(unittest.TestCase):
    def test_order_independence_and_class_holdouts(self):
        sources = {f"s{group}-{i}": content for group, content in enumerate([
            "1000-photos", "8100-screens", "7000-plots", "9094-illustrations"])
            for i in range(7)}
        a = partition_sources(sources, "fixed-seed")
        self.assertEqual(a, partition_sources(dict(reversed(list(sources.items()))), "fixed-seed"))
        self.assertEqual(set(a), set(sources))
        for content in set(sources.values()):
            counts = Counter(a[source] for source, c in sources.items() if c == content)
            self.assertEqual(counts, dict(fit=3, tune=2, test=2))

    def test_insufficient_class_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "not enough sources"):
            partition_sources(dict(a="7000-plots", b="8100-screens"), "seed")


if __name__ == "__main__":
    unittest.main()
