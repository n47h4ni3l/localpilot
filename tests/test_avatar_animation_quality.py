from __future__ import annotations

import hashlib
import zlib

from localpilot import native_avatar


def _decode_rgba8_png(raw: bytes) -> tuple[int, int, bytes]:
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    position = 8
    width = height = 0
    idat: list[bytes] = []
    while position + 12 <= len(raw):
        length = int.from_bytes(raw[position : position + 4], "big")
        kind = raw[position + 4 : position + 8]
        chunk = raw[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width = int.from_bytes(chunk[0:4], "big")
            height = int.from_bytes(chunk[4:8], "big")
            assert chunk[8:13] == bytes((8, 6, 0, 0, 0))
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break

    packed = zlib.decompress(b"".join(idat))
    stride = width * 4
    assert len(packed) == height * (stride + 1)
    decoded = bytearray(width * height * 4)
    previous = bytearray(stride)
    source = destination = 0

    def paeth(left: int, up: int, upper_left: int) -> int:
        prediction = left + up - upper_left
        distances = (
            abs(prediction - left),
            abs(prediction - up),
            abs(prediction - upper_left),
        )
        return (left, up, upper_left)[distances.index(min(distances))]

    for _ in range(height):
        filter_type = packed[source]
        source += 1
        row = bytearray(packed[source : source + stride])
        source += stride
        for index in range(stride):
            left = row[index - 4] if index >= 4 else 0
            up = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + up) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[index] = (row[index] + paeth(left, up, upper_left)) & 0xFF
            else:
                assert filter_type == 0
        decoded[destination : destination + stride] = row
        destination += stride
        previous = row
    return width, height, bytes(decoded)


def _frame_digest(pixels: bytes, width: int, row: int, frame: int, size: int) -> str:
    digest = hashlib.sha256()
    x0 = frame * size
    y0 = row * size
    for y in range(y0, y0 + size):
        start = (y * width + x0) * 4
        digest.update(pixels[start : start + size * 4])
    return digest.hexdigest()


def test_avatar_frames_have_true_transparent_corners():
    manifest = native_avatar._load_animation_manifest()
    assert manifest is not None
    raw = native_avatar._read_animation_atlas_data(manifest)
    assert raw is not None
    width, height, pixels = _decode_rgba8_png(raw)
    size = native_avatar.AVATAR_SIZE
    assert (width, height) == (size * manifest["columns"], size * manifest["rows"])

    for state, spec in manifest["states"].items():
        row = int(spec["row"])
        for frame in range(int(spec["frames"])):
            x0 = frame * size
            y0 = row * size
            for x, y in (
                (x0, y0),
                (x0 + size - 1, y0),
                (x0, y0 + size - 1),
                (x0 + size - 1, y0 + size - 1),
            ):
                assert pixels[(y * width + x) * 4 + 3] == 0, (state, frame, x, y)


def test_live_avatar_states_use_distinct_drawings_not_static_image_transforms():
    manifest = native_avatar._load_animation_manifest()
    assert manifest is not None
    raw = native_avatar._read_animation_atlas_data(manifest)
    assert raw is not None
    width, _height, pixels = _decode_rgba8_png(raw)
    size = native_avatar.AVATAR_SIZE

    for state, spec in manifest["states"].items():
        if state == "offline":
            continue
        digests = {
            _frame_digest(pixels, width, int(spec["row"]), frame, size)
            for frame in range(int(spec["loop_start"]), int(spec["loop_end"]) + 1)
        }
        assert len(digests) >= 2, state

    minimums = {"idle": 3, "speaking": 4, "restarting": 4, "sleeping": 4}
    for state, minimum in minimums.items():
        spec = manifest["states"][state]
        digests = {
            _frame_digest(pixels, width, int(spec["row"]), frame, size)
            for frame in range(int(spec["loop_start"]), int(spec["loop_end"]) + 1)
        }
        assert len(digests) >= minimum, state
