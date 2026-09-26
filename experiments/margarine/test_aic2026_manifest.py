import unittest
from io import BytesIO
from zipfile import ZipFile, ZipInfo

from aic2026_manifest import audit


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.header = "distorted,source,codec_acronym,distortion_level,bpp,JND_CVVDP\n"
        self.row = "S01_JPG_01.png,S01_Ref_00.png,JPG,1,0.5,0.2\n"
        data = BytesIO()
        with ZipFile(data, "w") as archive:
            archive.writestr("sources/S01_Ref_00.png", b"reference fixture")
            archive.writestr("distorted/S01_JPG_01.png", b"distorted fixture")
        with ZipFile(data) as archive:
            self.members = archive.infolist()

    def test_metric_derived_jnd_is_not_a_human_label(self):
        rows, _ = audit(self.header + self.row, self.members, "fixture.zip")
        self.assertEqual(rows[0]["human_target_status"], "unavailable")
        self.assertEqual(rows[0]["split_status"], "unassigned")

    def test_missing_or_duplicate_images_and_rows_fail(self):
        for text, members in [
            (self.header + self.row, self.members[:1]),
            (self.header + self.row, self.members * 2),
            (self.header + self.row * 2, self.members),
            (self.header + self.row, self.members + [ZipInfo("distorted/extra.png")]),
        ]:
            with self.subTest(text=text, members=members), self.assertRaises(ValueError):
                audit(text, members, "fixture.zip")

    def test_invalid_metadata_fails(self):
        for row in [self.row.replace("0.5", "NaN"), self.row.replace(",1,", ",0,"),
                    self.row.replace("S01_JPG_01.png", "../S01_JPG_01.png")]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                audit(self.header + row, self.members, "fixture.zip")


if __name__ == "__main__":
    unittest.main()
