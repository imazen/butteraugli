import csv
from pathlib import Path
import struct
import tempfile
import unittest

from prepare_human import records
from cid22_manifest import audit


class HumanInputs(unittest.TestCase):
    def setUp(self):
        scratch=Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_kadid_keeps_native_quality_and_sigma_units(self):
        (self.root/'dmos.csv').write_text('dist_img,ref_img,dmos,var\nI01_01_01.png,I01.png,4.57,0.49\n')
        rows,_=records('kadid',self.root)
        self.assertEqual(rows[0][3:5],(4.57,'quality'))
        self.assertAlmostEqual(rows[0][5],0.7)

    def test_tid_preserves_mixed_case_reference_and_label_alignment(self):
        (self.root/'reference_images_png').mkdir()
        (self.root/'distorted_images_png').mkdir()
        (self.root/'reference_images_png/i25.png').touch()
        (self.root/'distorted_images_png/I25_01_1.png').touch()
        (self.root/'mos_with_names.txt').write_text('5.5 i25_01_1.bmp\n')
        (self.root/'mos_std.txt').write_text('0.25\n')
        rows,_=records('tid',self.root)
        self.assertEqual(rows[0][0].name,'i25.png')
        self.assertEqual(rows[0][1].name,'I25_01_1.png')
        self.assertEqual(rows[0][3:],(5.5,'quality',0.25))

    def test_bmp_audit_rejects_truncation_and_unaudited_color_headers(self):
        pixels=bytes(16)
        file_header=struct.pack('<2sIHHI',b'BM',54+len(pixels),0,0,54)
        dib=struct.pack('<IiiHHIIiiII',40,2,-2,1,24,0,16,0,0,0,0)
        p=self.root/'fixture.bmp';p.write_bytes(file_header+dib+pixels)
        self.assertEqual(audit(p),(2,2))
        p.write_bytes((file_header+dib+pixels)[:-1])
        with self.assertRaisesRegex(ValueError,'truncated BMP pixels'):audit(p)
        bad=bytearray(file_header+dib+pixels);bad[14:18]=(124).to_bytes(4,'little');p.write_bytes(bad)
        with self.assertRaisesRegex(ValueError,'unaudited BMP layout'):audit(p)


if __name__=='__main__':
    unittest.main()
