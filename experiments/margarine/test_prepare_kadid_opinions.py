from decimal import Decimal
import unittest
from prepare_kadid_opinions import agrees_with_rounding, image_name


class OpinionTests(unittest.TestCase):
    def test_native_image_identity_and_published_rounding(self):
        self.assertEqual(image_name('kon10k_png/i01_2_3.png'), 'I01_02_03.png')
        self.assertEqual(image_name('kon10k_png/i01_7_1.png'), 'I01_09_01.png')
        self.assertTrue(agrees_with_rounding(Decimal(137) / 30, '4.57'))
        self.assertFalse(agrees_with_rounding(Decimal('4.56'), '4.57'))
        self.assertFalse(agrees_with_rounding(Decimal('0.4966'), '0.496'))
        with self.assertRaises(ValueError):
            image_name('kon10k_png/unknown.png')


if __name__ == '__main__':
    unittest.main()
