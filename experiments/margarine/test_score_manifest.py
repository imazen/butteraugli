import struct
import tempfile
import unittest
from pathlib import Path

from score_manifest import audit_png, parse_score


class ScoringContract(unittest.TestCase):
    def test_color_tags_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tagged.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4s", 4, b"gAMA"))
            with self.assertRaisesRegex(ValueError, "managed ingress"):
                audit_png(path)

    def test_nonfinite_scores_and_short_maps_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map"
            path.write_bytes(bytes(4))
            header = "mode\twidth\theight\tmax\tp1\tp2\tp3\tp6\tdiffmap\n"
            row = f"box3\t1\t1\t1\t1\t1\t1\t1\t{path}\n"
            self.assertEqual(parse_score(header + row, "box3", (1, 1), path)["p3"], 1)
            bad = row.split("\t")
            bad[3] = "nan"
            with self.assertRaisesRegex(ValueError, "invalid metric output"):
                parse_score(header + "\t".join(bad), "box3", (1, 1), path)
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "byte count"):
                parse_score(header + row, "box3", (1, 1), path)


if __name__ == "__main__":
    unittest.main()
