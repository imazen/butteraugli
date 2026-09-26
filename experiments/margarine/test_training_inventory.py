import unittest
from training_inventory import rendition_axes


class AxesTests(unittest.TestCase):
    def test_explicit_filename_and_mode(self):
        row = rendition_axes(dict(image_path="1234.scale256x128.png", q=15,
                                  knob_tuple_json='{"cell":"s2-420","fp":"abc"}'))
        self.assertEqual((row["origin_id"], row["width"], row["height"], row["mode"]),
                         ("1234", 256, 128, "s2-420"))
        self.assertEqual(row["q"], 15)

    def test_unrecognized_names_fail_instead_of_combining_sources(self):
        for name in ("image.png", "1234.crop256x128.png", "1234.scale0x128.png"):
            with self.assertRaises(ValueError): rendition_axes(dict(image_path=name))
