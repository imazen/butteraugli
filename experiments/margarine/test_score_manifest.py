import struct
import tempfile
import unittest
import zlib

from cid22_manifest import audit
from pathlib import Path

from score_manifest import EDGE_COLUMNS, FIELDS, NORMS, aligned_teacher, audit_png, parse_features, parse_prediction, parse_score


class ScoringContract(unittest.TestCase):
    def test_cached_teacher_requires_identical_labels_pixels_and_dimensions(self):
        row = dict(zip(FIELDS, ("aic", "source", "codec", "pair", "2", "quality", "ref", "dist")))
        frozen = dict(row, reference_sha256="a", distorted_sha256="b",
                      scores={"teacher": dict.fromkeys(NORMS, 1.0) | dict(width=2, height=3)})
        images = dict(ref="a", dist="b")
        self.assertEqual(aligned_teacher(row, frozen, images, (2, 3))["p3"], 1.0)
        for altered in (dict(row, target="3"), dict(row, source="other")):
            with self.assertRaisesRegex(ValueError, "labels or identities"):
                aligned_teacher(altered, frozen, images, (2, 3))
        with self.assertRaisesRegex(ValueError, "input hash"):
            aligned_teacher(row, frozen, dict(ref="c", dist="b"), (2, 3))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            aligned_teacher(row, frozen, images, (3, 2))

    def test_student_scalar_output_is_checked_without_inventing_a_map(self):
        text = "mode\twidth\theight\tmax\tp1\tp2\tp3\tp6\nmargarine-probe\t2\t3\t1\t1\t1\t1\t1\n"
        _, scores = parse_prediction(text, "margarine-probe", (2, 3))
        self.assertEqual(scores, dict.fromkeys(NORMS, 1.0))
        with self.assertRaisesRegex(ValueError, "invalid metric output"):
            parse_prediction(text.replace("\t1\n", "\tnan\n"), "margarine-probe", (2, 3))

    def test_feature_order_and_nonfinite_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.tsv"
            header = "\t".join(["width", "height"] + EDGE_COLUMNS)
            values = ["1", "1"] + ["0"] * 168
            path.write_text(header + "\n" + "\t".join(values) + "\n")
            self.assertEqual(parse_features(path, (1, 1)), [0.0] * 168)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                parse_features(path, (2, 1))
            values[-1] = "nan"
            path.write_text(header + "\n" + "\t".join(values) + "\n")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                parse_features(path, (1, 1))
            path.write_text(header.replace("feature_003", "feature_004") + "\n")
            with self.assertRaisesRegex(ValueError, "ordering"):
                parse_features(path, (1, 1))

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
