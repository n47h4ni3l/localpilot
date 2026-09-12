from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from localpilot import native_avatar


REQUIRED_STATES = {
    "idle",
    "listening",
    "thinking",
    "researching",
    "working",
    "speaking",
    "success",
    "uncertain",
    "error",
    "learning",
    "restarting",
    "sleeping",
    "offline",
}

EXPECTED_PRIMARY_ASSETS = {
    "idle.png",
    "listening.png",
    "thinking.png",
    "researching.png",
    "working.png",
    "speaking.png",
    "success.png",
    "error.png",
    "sleeping.png",
    "offline.png",
}


def test_per_state_animation_sheets_are_committed_verified_and_substantive():
    manifest = json.loads(native_avatar._animation_manifest_path().read_text(encoding="utf-8"))
    assert manifest["version"] == 6
    assert manifest["frame_size"] == native_avatar.AVATAR_SIZE
    assert 0 <= manifest["transition_ms"] <= 1000

    files = {asset["file"] for asset in manifest["assets"].values()}
    assert files == EXPECTED_PRIMARY_ASSETS

    for name, asset in manifest["assets"].items():
        raw = native_avatar._read_sheet_asset_data(asset)
        assert raw is not None, name
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(raw) == asset["bytes"]
        assert len(raw) > 1_000_000

        path = native_avatar._sheet_asset_path(asset)
        assert path is not None and path.is_file()
        with Image.open(path) as image:
            assert image.width >= asset["columns"] * 32
            assert image.height >= asset["rows"] * 32
            assert asset["frames"] == asset["columns"] * asset["rows"]

            sheet = image.convert("RGBA")
            registration = native_avatar._sheet_registration_box(sheet, asset)
            left, top, right, bottom = registration
            assert right > left
            assert bottom > top
            for index in range(asset["frames"]):
                frame = native_avatar._crop_sheet_frame(sheet, asset, index, registration)
                assert frame.size == (native_avatar.AVATAR_SIZE, native_avatar.AVATAR_SIZE)
                assert frame.getchannel("A").getbbox() is not None


def test_animation_manifest_covers_every_runtime_state_with_completed_loops():
    manifest = native_avatar._load_animation_manifest()
    assert manifest is not None
    assert REQUIRED_STATES <= set(manifest["states"])

    for state in REQUIRED_STATES:
        spec = manifest["states"][state]
        asset = manifest["assets"][spec["asset"]]
        assert asset["frames"] >= 8
        assert 50 <= spec["frame_ms"] <= 1000
        assert 0 <= spec["representative_frame"] < asset["frames"]

    assert manifest["states"]["working"]["asset"] == "working"
    assert manifest["assets"]["working"]["frames"] == 12
    assert manifest["states"]["learning"]["asset"] == "thinking"
    assert manifest["states"]["restarting"]["asset"] == "working"
    assert manifest["states"]["uncertain"]["asset"] == "uncertain"
    assert manifest["states"]["error"]["asset"] == "uncertain"


def test_native_renderer_registers_each_loop_before_scaling_frames():
    source = Path(native_avatar.__file__).read_text(encoding="utf-8")
    assert "_sheet_registration_box" in source
    assert "_visible_alpha_bbox" in source
    assert "_crop_sheet_frame" in source
    assert "Image.Resampling.LANCZOS" in source
    assert "alpha_composite" in source
    assert "_animation_frames" in source
    assert "_frame_cursor" in source
    assert "create_image" in source

    # Regression guard: no state-specific whole-image squeeze/nudge animation.
    assert "_loop_offset" not in source
    assert "_LINE_BOIL_TICKS" not in source
    assert "line_frame" not in source


def test_registration_box_is_stable_when_motion_changes_visible_bounds():
    sheet = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    # Frame 0: narrower pose lower in the cell.
    sheet.paste((255, 255, 255, 255), (10, 20, 50, 80))
    # Frame 1: wider pose, shifted upward, in the second 100px cell.
    sheet.paste((255, 255, 255, 255), (120, 10, 180, 90))
    asset = {"columns": 2, "rows": 1, "frames": 2}

    registration = native_avatar._sheet_registration_box(sheet, asset)
    assert registration == (10, 10, 80, 90)

    first = native_avatar._crop_sheet_frame(sheet, asset, 0, registration)
    second = native_avatar._crop_sheet_frame(sheet, asset, 1, registration)
    assert first.size == second.size == (native_avatar.AVATAR_SIZE, native_avatar.AVATAR_SIZE)


def test_native_state_change_starts_on_frame_zero_instead_of_skipping_it():
    app = object.__new__(native_avatar.NativeAvatarApp)
    app._animation_manifest = {"states": {}}
    app._animation_frames = {"sentinel": [object()]}
    app._display_state = "idle"
    app.runtime_state = "working"
    app._frame_cursor = 7

    assert app._request_runtime_state() is True
    assert app._display_state == "working"
    assert app._frame_cursor == 0
    assert app._request_runtime_state() is False

    source = Path(native_avatar.__file__).read_text(encoding="utf-8")
    assert "state_changed = self._request_runtime_state()" in source
    assert "if not state_changed:" in source


def test_webview_uses_registered_per_state_sheets_and_polished_transitions():
    webview_dir = Path(native_avatar.__file__).resolve().parent / "webview"
    index = (webview_dir / "index.html").read_text(encoding="utf-8")
    script = (webview_dir / "illustrated-avatar.js").read_text(encoding="utf-8")

    assert '<script src="app.js"></script>' in index
    assert '<script src="illustrated-avatar.js"></script>' in index
    assert "document.documentElement.dataset.state" in script
    assert 'const MANIFEST_URL = "avatar/anim/animation-manifest.json"' in script
    assert "alphaTrimRect" in script
    assert "registeredFrameRects" in script
    assert "drawFittedFrame" in script
    assert "smoothstep" in script
    assert "previousFrameIndex" in script
    assert "this.stateStart = now + this.transitionMs" in script
    assert "manifest.assets" in script
    assert 'loadImage("avatar/anim/" + spec.file)' in script
    assert "getImageData" in script
    assert "prefers-reduced-motion" in script
    assert 'this.canvas.style.opacity = "0"' in script
    assert "pixel fallback" in script

    # No transform-driven fake writing/typing/breathing animation remains.
    assert "function stateMotion" not in script
    assert "motionTransform" not in script
    assert "translate(" not in script
    assert "rotate(" not in script
    assert "scale(" not in script
    assert "LINE_BOIL_MS" not in script


def test_missing_or_tampered_animation_sheet_keeps_pixel_fallback(tmp_path, monkeypatch):
    manifest = json.loads(native_avatar._animation_manifest_path().read_text(encoding="utf-8"))
    asset = dict(manifest["assets"]["idle"])

    missing = tmp_path / "idle.png"
    monkeypatch.setattr(native_avatar, "_sheet_asset_path", lambda _asset: missing)
    assert native_avatar._read_sheet_asset_data(asset) is None

    bad = tmp_path / "idle.png"
    bad.write_bytes(b"not the approved animation sheet")
    monkeypatch.setattr(native_avatar, "_sheet_asset_path", lambda _asset: bad)
    assert native_avatar._read_sheet_asset_data(asset) is None


def test_researching_and_learning_are_valid_native_runtime_states():
    assert "researching" in native_avatar._STATE_COLORS
    assert "learning" in native_avatar._STATE_COLORS
    assert native_avatar._normalized_state("researching") == "researching"
    assert native_avatar._normalized_state("learning") == "learning"
