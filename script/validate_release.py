#!/usr/bin/env python3
"""Validate release completeness, closure, hashes, provenance, and hygiene."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import sys
import zlib

from jsonschema.validators import validator_for

from dataset import DataError, load_json, load_manifest, projected_criteria, sha256

EXPECTED = 2220
FORBIDDEN_TEXT = ("/apd" + "cephfs", "/ro" + "ot/", "share_" + "303970632",
                  "vista_" + "his", "poko" + "lv")
PRIVATE_IP = re.compile(r"(?<!\d)(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?!\d)")
SECRET = re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{16,}|bearer\s+[A-Za-z0-9._-]{16,}|AKIA[A-Z0-9]{16})")
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".txt", ".toml", ".csv"}
JPEG_SOI = b"\xff\xd8"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ALLOWED_ORIENTATIONS = {3, 6, 8}
NORMALIZED_MTIME = 946684800
METADATA_NEEDLES = (
    b"xmp", b"photoshop", b"iptc", b"aigc", b"contentauthenticity",
    b"artist", b"copyright", b"hostcomputer", b"creationtime",
)


def fail(message: str) -> None:
    raise DataError(message)


def json_assets(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.name != "manifest.json")


def minimal_orientation_tiff(orientation: int) -> bytes:
    return (b"II\x2a\x00\x08\x00\x00\x00\x01\x00" +
            struct.pack("<HHI", 274, 3, 1) + struct.pack("<H", orientation) +
            b"\x00\x00\x00\x00\x00\x00")


def minimal_orientation(payload: bytes, *, prefix: bytes = b"") -> int | None:
    for orientation in sorted(ALLOWED_ORIENTATIONS):
        if payload == prefix + minimal_orientation_tiff(orientation):
            return orientation
    return None


def scan_jpeg(data: bytes, relative: str) -> int | None:
    if not data.startswith(JPEG_SOI):
        fail(f"invalid JPEG signature: {relative}")
    pos = 0
    in_scan = False
    orientations = []
    saw_eoi = False
    while pos < len(data):
        if in_scan:
            marker_start = data.find(b"\xff", pos)
            if marker_start < 0:
                fail(f"unterminated JPEG scan: {relative}")
            q = marker_start + 1
            while q < len(data) and data[q] == 0xFF:
                q += 1
            if q >= len(data):
                fail(f"unterminated JPEG marker: {relative}")
            if data[q] == 0:
                pos = q + 1
                continue
            pos = marker_start
        if data[pos] != 0xFF:
            fail(f"unexpected JPEG data outside scan: {relative}")
        q = pos + 1
        while q < len(data) and data[q] == 0xFF:
            q += 1
        if q >= len(data):
            fail(f"unterminated JPEG marker: {relative}")
        marker = data[q]
        if marker == 0:
            fail(f"stuffed JPEG byte outside scan: {relative}")
        if marker in {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}:
            pos = q + 1
            if marker == 0xD9:
                saw_eoi = True
                in_scan = False
                if pos < len(data):
                    if data[pos:pos + 2] != JPEG_SOI:
                        fail(f"unexpected bytes after JPEG EOI: {relative}")
                    saw_eoi = False
            continue
        if q + 3 > len(data):
            fail(f"truncated JPEG segment: {relative}")
        length = int.from_bytes(data[q + 1:q + 3], "big")
        end = q + 1 + length
        if length < 2 or end > len(data):
            fail(f"invalid JPEG segment length: {relative}")
        payload = data[q + 3:end]
        lowered = payload.lower()
        if marker == 0xE1:
            orientation = minimal_orientation(payload, prefix=b"Exif\x00\x00")
            if orientation is None:
                fail(f"non-minimal EXIF/XMP segment: {relative}")
            orientations.append(orientation)
        elif marker in {*range(0xE3, 0xEE), 0xEF, 0xFE}:
            fail(f"JPEG metadata/comment segment 0x{marker:02x}: {relative}")
        elif marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
            fail(f"JPEG ICC profile: {relative}")
        elif any(needle in lowered for needle in METADATA_NEEDLES):
            fail(f"JPEG identity/AIGC metadata signature: {relative}")
        pos = end
        if marker == 0xDA:
            in_scan = True
    if not saw_eoi:
        fail(f"JPEG has no final EOI: {relative}")
    if len(orientations) > 1:
        fail(f"multiple retained EXIF orientation segments: {relative}")
    return orientations[0] if orientations else None


def scan_png(data: bytes, relative: str) -> int | None:
    if not data.startswith(PNG_SIGNATURE):
        fail(f"invalid PNG signature: {relative}")
    pos = len(PNG_SIGNATURE)
    orientation = None
    saw_end = False
    allowed_ancillary = {b"tRNS", b"acTL", b"fcTL", b"fdAT"}
    while pos < len(data):
        if pos + 12 > len(data):
            fail(f"truncated PNG chunk: {relative}")
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            fail(f"truncated PNG payload: {relative}")
        payload = data[pos + 8:pos + 8 + length]
        expected_crc = int.from_bytes(data[pos + 8 + length:end], "big")
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            fail(f"PNG CRC mismatch: {relative}")
        if kind == b"eXIf":
            if orientation is not None:
                fail(f"multiple PNG EXIF chunks: {relative}")
            orientation = minimal_orientation(payload)
            if orientation is None:
                fail(f"non-minimal PNG EXIF: {relative}")
        elif kind[0] & 0x20 and kind not in allowed_ancillary:
            fail(f"PNG metadata chunk {kind!r}: {relative}")
        pos = end
        if kind == b"IEND":
            saw_end = True
            if pos != len(data):
                fail(f"unexpected bytes after PNG IEND: {relative}")
    if not saw_end:
        fail(f"PNG has no IEND: {relative}")
    return orientation


def scan_webp(data: bytes, relative: str) -> None:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        fail(f"invalid WebP signature: {relative}")
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        fail(f"invalid WebP RIFF size: {relative}")
    pos = 12
    allowed = {b"VP8 ", b"VP8L", b"VP8X", b"ALPH", b"ANIM", b"ANMF"}
    while pos < len(data):
        if pos + 8 > len(data):
            fail(f"truncated WebP chunk: {relative}")
        kind = data[pos:pos + 4]
        length = int.from_bytes(data[pos + 4:pos + 8], "little")
        end = pos + 8 + length
        padded = end + (length & 1)
        if padded > len(data):
            fail(f"truncated WebP payload: {relative}")
        payload = data[pos + 8:end]
        if kind not in allowed:
            fail(f"WebP metadata/unknown chunk {kind!r}: {relative}")
        if kind == b"VP8X" and (len(payload) != 10 or payload[0] & 0x2C):
            fail(f"WebP metadata flag retained: {relative}")
        pos = padded


def scan_image(path: Path, root: Path) -> int | None:
    data = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    if data.startswith(JPEG_SOI):
        return scan_jpeg(data, relative)
    if data.startswith(PNG_SIGNATURE):
        return scan_png(data, relative)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        scan_webp(data, relative)
        return None
    fail(f"unsupported image format: {relative}")


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def validate(root: Path, *, check_hygiene: bool = False) -> dict:
    required = ["README.md", "LICENSE", "requirements.txt", "config.example.toml",
                "data/manifest.jsonl", "data/release_manifest.json", "data/images", "data/ocr",
                "data/verified_translations", "data/simple_rubric", "data/rubric", "data/results",
                "script/api.py", "script/dataset.py", "script/generate_image_candidates.py",
                "script/generate_text_candidates.py", "script/judge.py", "script/scoring.py",
                "script/summarize.py", "script/validate_release.py", "script/prompts", "script/schemas",
                "script/tests"]
    for relative in required:
        if not (root / relative).exists():
            fail(f"missing required release entry: {relative}")
    for path in root.rglob("*"):
        if path.is_symlink():
            fail(f"symlinks are forbidden: {path.relative_to(root)}")
        if check_hygiene:
            if path.name in {"__pycache__", ".pytest_cache", ".cache"}:
                fail(f"cache artifact is forbidden: {path.relative_to(root)}")
            expected_mode = 0o755 if path.is_dir() else 0o644
            metadata = path.stat()
            actual_mode = stat.S_IMODE(metadata.st_mode)
            if actual_mode != expected_mode:
                fail(f"non-normalized permissions {actual_mode:o}: {path.relative_to(root)}")
            if int(metadata.st_mtime) != NORMALIZED_MTIME:
                fail(f"non-normalized filesystem timestamp: {path.relative_to(root)}")
    schemas = sorted((root / "script/schemas").glob("*.json"))
    if not schemas:
        fail("no JSON schemas found")
    for path in schemas:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"invalid JSON schema {path.relative_to(root)}: {exc}")
        if not isinstance(schema, dict):
            fail(f"JSON schema root is not an object: {path.relative_to(root)}")
        try:
            validator_for(schema).check_schema(schema)
        except Exception as exc:
            fail(f"invalid JSON schema {path.relative_to(root)}: {exc}")
    records = load_manifest(root, verify_hashes=True)
    if len(records) != EXPECTED:
        fail(f"manifest count is {len(records)}, expected {EXPECTED}")
    excluded = {4499, 4851, 5801, 5805, 8574, 9477, 10066, 10960}
    if any(row["seq"] in excluded for row in records):
        fail("privacy-excluded sample remains in manifest")
    counts = {
        "images": sum(path.is_file() for path in (root / "data/images").rglob("*")),
        "ocr": len(json_assets(root / "data/ocr")),
        "verified_translations": len(json_assets(root / "data/verified_translations")),
        "simple_rubric": len(json_assets(root / "data/simple_rubric")),
        "rubric": len(json_assets(root / "data/rubric")),
    }
    if any(value != EXPECTED for value in counts.values()):
        fail(f"asset count mismatch: {counts}")
    retained_orientations = 0
    for path in sorted(item for item in (root / "data/images").rglob("*") if item.is_file()):
        retained_orientations += scan_image(path, root) is not None
    referenced = {asset["path"] for row in records for asset in row["assets"].values()}
    actual = set()
    for directory in ("images", "ocr", "verified_translations", "simple_rubric", "rubric"):
        for path in (root / "data" / directory).rglob("*"):
            if path.is_file() and path.name not in {"manifest.json", "zero_criteria_tasks.jsonl"}:
                actual.add(path.relative_to(root).as_posix())
    if referenced != actual:
        fail(f"manifest closure mismatch: unreferenced={len(actual-referenced)}, missing={len(referenced-actual)}")
    simple_manifest = load_json(root / "data/simple_rubric/manifest.json")
    if simple_manifest.get("source_kind_counts") != {"historical_original_fallback": 2, "rubric_lite_v1": 2218}:
        fail("simple rubric provenance counts are invalid")
    fallback_count = native_count = 0
    textohner = process_keys = 0
    projections = retained = zero = 0
    for row in records:
        translation = load_json(root / row["assets"]["verified_translation"]["path"])
        def walk(value):
            nonlocal textohner, process_keys
            if isinstance(value, dict):
                textohner += int("textohner" in value)
                process_keys += sum(key in value for key in
                                    ("verified_by", "fixed_by", "text_before_verify", "text_before_fix"))
                for child in value.values(): walk(child)
            elif isinstance(value, list):
                for child in value: walk(child)
        walk(translation)
        simple = load_json(root / row["assets"]["simple_rubric"]["path"])
        kind = simple.get("provenance", {}).get("source_kind")
        native = simple.get("provenance", {}).get("native_lite_v1")
        fallback_count += kind == "historical_original_fallback" and native is False
        native_count += kind == "rubric_lite_v1" and native is True
        evaluation_rubric = load_json(root / row["assets"]["rubric"]["path"])
        provenance = evaluation_rubric.get("provenance", {})
        if (provenance.get("selection_leakage") is not True or
                provenance.get("analysis_status") != "exploratory"):
            fail("selected rubric must declare its interpretation limits")
        for lang in evaluation_rubric.get("target_langs", []):
            criteria = projected_criteria(evaluation_rubric, lang)
            projections += 1; retained += len(criteria); zero += not criteria
    if textohner != 0 or process_keys != 0 or fallback_count != 2 or native_count != 2218:
        fail("translation rename or simple-rubric provenance audit failed")
    rubric_manifest = load_json(root / "data/rubric/manifest.json")
    expected_projection_counts = (
        rubric_manifest.get("task_projections"),
        rubric_manifest.get("retained_criterion_projection_instances"),
        rubric_manifest.get("zero_criteria_tasks"),
    )
    if (projections, retained, zero) != expected_projection_counts:
        fail(f"rubric projection audit failed: {(projections, retained, zero)}")
    zero_lines = sum(1 for line in (root / "data/rubric/zero_criteria_tasks.jsonl").open() if line.strip())
    if zero_lines != zero:
        fail("zero-criteria list count mismatch")
    for path in (root / "data").rglob("*"):
        lowered = path.name.lower()
        if path.is_file() and (lowered in {"task_scores.csv", "raw_judgments.jsonl"} or
                               "prefinal" in lowered or "precoverage" in lowered or
                               "leaderboard_delta" in lowered):
            fail(f"forbidden process artifact: {path.relative_to(root)}")
    expected_results = {
        "data/results/image/simple_rubric": {"leaderboard.csv", "scores.csv", "coverage.csv", "summary.json"},
        "data/results/image/rubric": {"leaderboard.csv", "scores.csv", "scores_by_direction_level1.csv", "scores_by_level1.csv", "summary.json"},
        "data/results/text/rubric": {"leaderboard.csv", "leaderboard_task_micro.csv", "scores_by_direction.csv", "scores_by_direction_level1.csv", "scores_by_level1.csv", "coverage.csv", "summary.json"},
    }
    for relative, expected_files in expected_results.items():
        directory = root / relative
        actual_files = {path.name for path in directory.iterdir() if path.is_file()}
        if actual_files != expected_files:
            fail(f"unexpected aggregate result files in {relative}")
        summary = load_json(directory / "summary.json")
        if summary.get("cross_track_comparable") is not False:
            fail(f"result track must be non-comparable: {relative}")
        should_explore = relative != "data/results/image/simple_rubric"
        if summary.get("exploratory") is not should_explore:
            fail(f"incorrect exploratory flag: {relative}")
        if should_explore and summary.get("selection_leakage") is not True:
            fail(f"missing selection-leakage flag: {relative}")
        provenance = summary.get("release_dataset_coverage", {})
        if provenance.get("historical_full_dataset_2228") is not True or \
                provenance.get("excluded_from_release") != 8 or \
                provenance.get("not_exactly_reproducible_from_release_data") is not True:
            fail(f"missing historical aggregate disclosure: {relative}")
        for name, expected_hash in summary.get("files_sha256", {}).items():
            if sha256(directory / name) != expected_hash:
                fail(f"aggregate hash mismatch: {relative}/{name}")
    scanned = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        scanned += 1
        for token in FORBIDDEN_TEXT:
            if token in text:
                fail(f"internal path marker {token!r} in {path.relative_to(root)}")
        if PRIVATE_IP.search(text):
            fail(f"private-network address in {path.relative_to(root)}")
        if SECRET.search(text):
            fail(f"credential-like value in {path.relative_to(root)}")
    release = load_json(root / "data/release_manifest.json")
    if release.get("results", {}).get("raw_tasks_included") is not False:
        fail("release manifest must exclude raw task results")
    asset_trees = {}
    for asset_name in ("images", "ocr", "verified_translations", "simple_rubric", "rubric"):
        asset_manifest = release.get("assets", {}).get(asset_name, {})
        asset_root = root / "data" / asset_name
        asset_bytes = sum(path.stat().st_size for path in asset_root.rglob("*") if path.is_file())
        asset_tree = tree_sha256(asset_root)
        if (asset_manifest.get("bytes") != asset_bytes or
                asset_manifest.get("tree_sha256") != asset_tree):
            fail(f"release {asset_name} bytes/tree hash mismatch")
        asset_trees[asset_name] = asset_tree
    image_tree = asset_trees["images"]
    audit = release.get("sanitization_audit", {})
    image_audit = audit.get("image_metadata", {})
    required_audit_counts = {
        "metadata_stripped_files", "exif_files", "gps_files", "gps_coordinate_files",
        "xmp_files", "author_files", "host_files", "comment_files", "photoshop_files",
        "icc_files",
    }
    if not required_audit_counts <= image_audit.keys():
        fail("release manifest image anonymization counts are incomplete")
    if not all(isinstance(image_audit[key], int) and image_audit[key] >= 0
               for key in required_audit_counts):
        fail("release manifest image anonymization counts are invalid")
    if image_audit.get("pixel_content_preserved_unmodified_samples") != 2103:
        fail("release manifest unmodified-image count is invalid")
    if image_audit.get("face_redacted_images") != 121 or image_audit.get("face_regions") != 258:
        fail("release manifest face-redaction counts are invalid")
    if image_audit.get("orientation_preserved") is not True:
        fail("release manifest must attest orientation preservation")
    historical_release_tree = image_audit.get("release_tree_sha256")
    upstream_tree = image_audit.get("upstream_tree_sha256")
    if (not isinstance(historical_release_tree, str) or len(historical_release_tree) != 64 or
            not isinstance(upstream_tree, str) or len(upstream_tree) != 64 or
            upstream_tree == historical_release_tree):
        fail("historical image provenance hashes are invalid")
    if audit.get("filesystem_timestamps_normalized") is not True:
        fail("release manifest must record normalized filesystem timestamps")
    return {"status": "ok", "records": len(records), "counts": counts,
            "task_projections": projections, "retained_criteria": retained,
            "zero_criteria_tasks": zero, "simple_native": native_count,
            "simple_fallbacks": fallback_count, "text_files_scanned": scanned,
            "schemas_parsed": len(schemas), "images_metadata_scanned": counts["images"],
            "retained_orientation_files": retained_orientations,
            "images_tree_sha256": image_tree}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-hygiene", action="store_true",
                        help="also require normalized archive timestamps, modes, and no caches")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.root.resolve(), check_hygiene=args.check_hygiene), indent=2))
    except DataError as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
