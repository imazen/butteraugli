import struct
import tempfile
import unittest
import zlib

from cid22_manifest import audit
from pathlib import Path

from score_manifest import audit_png, parse_score


class ScoringContract(unittest.TestCase):
    def test_color_tags_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tagged.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4s", 4, b"gAMA"))
            with self.assertRaisesRegex(ValueError, "managed ingress"):
                audit_png(path)

    def test_cid16_metadata_is_explicit_and_crc_checked(self):
        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb16.png"
            header = chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 3, 16, 2, 0, 0, 0))
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + header + chunk(b"IEND", b""))
            self.assertEqual(audit(path), (2, 3))
            bad_icc = chunk(b"iCCP", b"unknown\0\0" + zlib.compress(b"not an approved profile"))
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + header + bad_icc + chunk(b"IEND", b""))
            with self.assertRaisesRegex(ValueError, "unaudited ICC"):
                audit(path)
            data = bytearray(path.read_bytes())
            data[24] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "bad PNG chunk"):
                audit(path)

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
            path.write_bytes(struct.pack("<f", float("nan")))
            with self.assertRaisesRegex(ValueError, "diffmap sample"):
                parse_score(header + row, "box3", (1, 1), path)
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "byte count"):
                parse_score(header + row, "box3", (1, 1), path)


if __name__ == "__main__":
    unittest.main()
