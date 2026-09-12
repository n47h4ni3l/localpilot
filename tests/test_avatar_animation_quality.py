from __future__ import annotations

import hashlib

from PIL import Image

from localpilot import native_avatar


def _frame_digest(frame: Image.Image) -> str:
    rgba = frame.convert("RGBA")
    return hashlib.sha256(rgba.tobytes()).hexdigest()


def test_avatar_sheet_frames_render_with_true_transparent_corners():
    manifest = native_avatar._load_animation_manifest()
    assert manifest is not None

    for asset_name, asset in manifest["assets"].items():
        path = native_avatar._sheet_asset_path(asset)
        assert path is not None
        with Image.open(path) as opened:
            sheet = opened.convert("RGBA")
            for index in range(int(asset["frames"])):
                frame = native_avatar._crop_sheet_frame(sheet, asset, index)
                alpha = frame.getchannel("A")
                corners = (
                    alpha.getpixel((0, 0)),
                    alpha.getpixel((frame.width - 1, 0)),
                    alpha.getpixel((0, frame.height - 1)),
                    alpha.getpixel((frame.width - 1, frame.height - 1)),
                )
                assert corners == (0, 0, 0, 0), (asset_name, index, corners)


def test_each_primary_state_sheet_contains_multiple_real_drawings():
    manifest = native_avatar._load_animation_manifest()
    assert manifest is not None

    minimums = {
        "idle": 4,
        "listening": 4,
        "thinking": 5,
        "researching": 5,
        "working": 6,
        "speaking": 5,
        "success": 5,
        "uncertain": 5,
        "sleeping": 4,
        "offline": 5,
    }

    for asset_name, minimum in minimums.items():
        asset = manifest["assets"][asset_name]
        path = native_avatar._sheet_asset_path(asset)
        assert path is not None
        with Image.open(path) as opened:
            sheet = opened.convert("RGBA")
            digests = {
                _frame_digest(native_avatar._crop_sheet_frame(sheet, asset, index))
                for index in range(int(asset["frames"]))
            }
        assert len(digests) >= minimum, (asset_name, len(digests), minimum)


def test_generated_sheet_geometry_is_trimmed_without_aspect_distortion():
    manifest = native_avatar._load_animation_manifest()
    assert manifest is not None

    for asset_name, asset in manifest["assets"].items():
        path = native_avatar._sheet_asset_path(asset)
        assert path is not None
        with Image.open(path) as opened:
            sheet = opened.convert("RGBA")
            for index in range(int(asset["frames"])):
                frame = native_avatar._crop_sheet_frame(sheet, asset, index)
                assert frame.size == (native_avatar.AVATAR_SIZE, native_avatar.AVATAR_SIZE)
                bbox = frame.getchannel("A").getbbox()
                assert bbox is not None, (asset_name, index)
                left, top, right, bottom = bbox
                assert 0 <= left < right <= native_avatar.AVATAR_SIZE
                assert 0 <= top < bottom <= native_avatar.AVATAR_SIZE
                # The fitted artwork keeps a small transparent safety margin.
                assert left > 0 or right < native_avatar.AVATAR_SIZE
                assert top > 0 or bottom < native_avatar.AVATAR_SIZE
