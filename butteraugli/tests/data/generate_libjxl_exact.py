"""Capture procedural sRGB oracle scores from pinned libjxl v0.12 (Pillow required)."""
import argparse
import csv
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("libjxl", type=Path, help="libjxl v0.12 source/build root")
parser.add_argument("output", type=Path, help="new output TSV path")
args = parser.parse_args()
source = args.libjxl / "tools/butteraugli_main.cc"
if 'printf("%g-norm: %.17g\\n", p, pnorm);' not in source.read_text():
    raise RuntimeError("Missing libjxl %.17g p-norm instrumentation; patch and rebuild butteraugli_main")

scratch = Path.home() / "tmp"
scratch.mkdir(exist_ok=True)
inputs = Path(tempfile.mkdtemp(prefix="butteraugli-oracles-", dir=scratch))
rows = []
for width, height in [(16, 16), (17, 19), (65, 49), (128, 96), (257, 193), (512, 512)]:
    reference = bytes(
        (x * 13 + y * 17 + c * 71 + (x * y) % 29) % 256
        for y in range(height) for x in range(width) for c in range(3)
    )
    distorted = bytes(v // 8 * 8 for v in reference)
    paths = []
    for label, pixels in [("ref", reference), ("dist", distorted)]:
        path = inputs / f"{width}x{height}-{label}.png"
        Image.frombytes("RGB", (width, height), pixels).save(path)
        paths.append(str(path))
    output = subprocess.check_output(
        [str(args.libjxl / "build/tools/butteraugli_main"), *paths,
         "--intensity_target", "80", "--pnorm", "3"], text=True,
    ).splitlines()
    pnorm = output[1].split()[-1]
    if len(pnorm.replace(".", "").replace("-", "")) < 12:
        raise RuntimeError("Binary lacks %.17g p-norm precision; rebuild butteraugli_main")
    rows.append((width, height, float(output[0]), float(pnorm)))

with args.output.open("x", newline="") as output:
    writer = csv.writer(output, delimiter="\t")
    writer.writerow(["width", "height", "max_norm", "pnorm3"])
    writer.writerows(rows)
