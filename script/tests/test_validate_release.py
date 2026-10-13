import json
from pathlib import Path
import struct
import sys
import zlib

import pytest
from jsonschema.validators import validator_for

sys.path.insert(0, str(Path(__file__).parents[1]))
from dataset import DataError
from validate_release import (
    minimal_orientation_tiff,
    scan_jpeg,
    scan_png,
    scan_webp,
)


def jpeg_segment(marker, payload):
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


def png_chunk(kind, payload):
    return (len(payload).to_bytes(4, "big") + kind + payload +
            (zlib.crc32(kind + payload) & 0xFFFFFFFF).to_bytes(4, "big"))


def webp(*chunks):
    body = bytearray(b"WEBP")
    for kind, payload in chunks:
        body.extend(kind + len(payload).to_bytes(4, "little") + payload)
        if len(payload) & 1:
            body.append(0)
    return b"RIFF" + len(body).to_bytes(4, "little") + bytes(body)


def test_all_release_schemas_are_strict_json():
    schema_dir = Path(__file__).parents[1] / "schemas"
    schemas = list(schema_dir.glob("*.json"))
    assert schemas
    for path in schemas:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(schema, dict)
        validator_for(schema).check_schema(schema)


def test_jpeg_allows_only_minimal_orientation_exif():
    payload = b"Exif\x00\x00" + minimal_orientation_tiff(6)
    clean = b"\xff\xd8" + jpeg_segment(0xE1, payload) + b"\xff\xd9"
    assert scan_jpeg(clean, "clean.jpg") == 6
    dirty = b"\xff\xd8" + jpeg_segment(0xE1, payload + b"private") + b"\xff\xd9"
    with pytest.raises(DataError):
        scan_jpeg(dirty, "dirty.jpg")


def test_jpeg_rejects_comment_icc_and_c2pa():
    for marker, payload in ((0xFE, b"comment"),
                            (0xE2, b"ICC_PROFILE\x00\x01\x01x"),
                            (0xEB, b"JUMBF c2pa")):
        data = b"\xff\xd8" + jpeg_segment(marker, payload) + b"\xff\xd9"
        with pytest.raises(DataError):
            scan_jpeg(data, "dirty.jpg")


def test_png_allows_only_minimal_orientation_metadata():
    signature = b"\x89PNG\r\n\x1a\n"
    clean = signature + png_chunk(b"eXIf", minimal_orientation_tiff(3)) + png_chunk(b"IEND", b"")
    assert scan_png(clean, "clean.png") == 3
    dirty = signature + png_chunk(b"tEXt", b"Author\x00private") + png_chunk(b"IEND", b"")
    with pytest.raises(DataError):
        scan_png(dirty, "dirty.png")


def test_webp_rejects_metadata_chunks_and_flags():
    clean = webp((b"VP8X", b"\x00" * 10))
    scan_webp(clean, "clean.webp")
    with pytest.raises(DataError):
        scan_webp(webp((b"XMP ", b"private")), "dirty.webp")
    with pytest.raises(DataError):
        scan_webp(webp((b"VP8X", b"\x08" + b"\x00" * 9)), "dirty.webp")
