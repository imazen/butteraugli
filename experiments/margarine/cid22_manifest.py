#!/usr/bin/env python3
"""Audit CID22 validation labels and its explicit encoded-sRGB input policy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import zlib

from score_manifest import FIELDS, digest

# Uncompressed ICC identities observed in the validation PNGs and JPEGs.
# All identify sRGB. This policy interprets encoded samples as standard sRGB,
# matching the common SDR convention, rather than applying the ICC LUTs.
SRGB_PROFILES = {
    "e24157b4e077058cf35fa11105e8a99f6d7198a1d6a71db1561253e74794b72d",
    "ec047e0e848704b5e24cea648995543b412520b90c9fa0eeb38d33c9514823c7",
    "2b3aa1645779a9e634744faf9b01e9102b0c9b88fd6deced7934df86b949af7e",
}


def audit(path):
    with path.open("rb") as f:
        signature = f.read(8)
        if signature == b"\x89PNG\r\n\x1a\n":
            dimensions = None
            while True:
                header = f.read(8)
                if len(header) != 8:
                    raise ValueError(f"truncated PNG: {path}")
                n, tag = struct.unpack(">I4s", header)
                data = f.read(n)
                crc = f.read(4)
                if len(data) != n or len(crc) != 4 or zlib.crc32(tag + data) != int.from_bytes(crc, "big"):
                    raise ValueError(f"bad PNG chunk: {path}")
                if tag == b"IHDR":
                    if n != 13 or data[8] not in (8, 16) or data[9] not in (2, 6):
                        raise ValueError(f"unsupported SDR PNG: {path}")
                    dimensions = struct.unpack(">II", data[:8])
                elif tag == b"iCCP":
                    _, compressed = data.split(b"\0", 1)
                    if compressed[0] != 0:
                        raise ValueError("unknown ICC compression")
                    if hashlib.sha256(zlib.decompress(compressed[1:])).hexdigest() not in SRGB_PROFILES:
                        raise ValueError(f"unaudited ICC profile: {path}")
                elif tag in (b"gAMA", b"cHRM", b"sRGB", b"cICP"):
                    raise ValueError(f"unaudited color tag {tag!r}: {path}")
                elif tag == b"IEND":
                    if dimensions is None:
                        raise ValueError("missing PNG dimensions")
                    return dimensions
        if signature[:2] != b"\xff\xd8":
            raise ValueError(f"unsupported image: {path}")
        f.seek(2)
        dimensions = None
        icc_parts, icc_count = {}, None
        while True:
            if f.read(1) != b"\xff":
                raise ValueError(f"invalid JPEG marker: {path}")
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker:
                raise ValueError("truncated JPEG")
            if marker in (b"\xda", b"\xd9"):
                if dimensions is None:
                    raise ValueError("missing JPEG dimensions")
                if icc_parts:
                    if set(icc_parts) != set(range(1, icc_count + 1)):
                        raise ValueError("incomplete JPEG ICC profile")
                    profile = b"".join(icc_parts[i] for i in range(1, icc_count + 1))
                    sha = hashlib.sha256(profile).hexdigest()
                    if sha not in SRGB_PROFILES:
                        raise ValueError(f"unaudited JPEG ICC {sha}: {path}")
                return dimensions
            length = f.read(2)
            if len(length) != 2 or int.from_bytes(length, "big") < 2:
                raise ValueError("invalid JPEG segment length")
            n = int.from_bytes(length, "big") - 2
            data = f.read(n)
            if len(data) != n:
                raise ValueError("truncated JPEG segment")
            if marker == b"\xe2" and data.startswith(b"ICC_PROFILE\0"):
                if len(data) < 14 or data[12] == 0 or data[13] == 0:
                    raise ValueError("invalid JPEG ICC segment")
                index, count = data[12:14]
                if index in icc_parts or (icc_count is not None and count != icc_count):
                    raise ValueError("inconsistent JPEG ICC segments")
                icc_parts[index], icc_count = data[14:], count
            if marker in (b"\xc0", b"\xc1", b"\xc2"):
                if len(data) < 6 or data[0] != 8 or data[5] != 3:
                    raise ValueError("expected three-channel 8-bit JPEG")
                h, w = struct.unpack(">HH", data[1:5])
                dimensions = w, h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    labels = root / "CID22_validation_set.csv"
    with labels.open() as f:
        original = list(csv.DictReader(f))
    if len(original) != 4341:
        raise ValueError("expected 4341 validation rows, including 49 references")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "progress.log").open("x", buffering=1) as log:
        def report(message):
            print(message, flush=True)
            print(message, file=log, flush=True)
        images, rows, identities, seen = {}, [], [], set()
        for i, row in enumerate(original):
            paths = []
            for key in ("reference_img", "distorted_img"):
                p = (root / row[key]).resolve()
                if not p.is_relative_to(root):
                    raise ValueError("path outside dataset root")
                if str(p) not in images:
                    if audit(p) != (512, 512):
                        raise ValueError("unexpected CID22 geometry")
                    images[str(p)] = digest(p)
                paths.append(str(p))
            if paths[0] == paths[1]:
                if row["encoder"] != "Reference" or int(row["nb_pc_opinions"]) != 0:
                    raise ValueError("unexpected identity label")
                identities.append(row)
                continue
            pair = row["distorted_img"]
            target, bpp = float(row["MCOS"]), float(row["bpp"])
            if pair in seen or not math.isfinite(target) or not math.isfinite(bpp) or bpp <= 0:
                raise ValueError("invalid or duplicated CID22 row")
            seen.add(pair)
            rows.append(dict(zip(FIELDS, ("cid22_validation", row["reference_img"],
                row["encoder"], pair, row["MCOS"], "quality", *paths)),
                bpp=row["bpp"], setting=row["setting"]))
            if i % 100 == 0:
                report(f"Audited {i + 1}/{len(original)} rows")
        if len(identities) != 49 or len(rows) != 4292:
            raise ValueError("unexpected identity/distortion counts")
        with (args.output / "pairs.tsv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS + ["bpp", "setting"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        shutil.copyfile(labels, args.output / "original_labels.csv")
        manifest = dict(build_commit=args.build_commit, labels_sha256=digest(labels),
            pairs_sha256=digest(args.output / "pairs.tsv"), n_pairs=len(rows),
            identity_rows_excluded=identities, images=images,
            ingress="encoded-sRGB; audited ICC identities; RGB16 preserved; alpha checked by decoder",
            icc_sha256=sorted(SRGB_PROFILES), target="MCOS, larger is better")
        (args.output / "_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report(f"Complete: {len(rows)} distorted pairs, {len(identities)} unscored identity rows")


if __name__ == "__main__":
    main()
