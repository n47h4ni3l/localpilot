from __future__ import annotations

import base64
from pathlib import Path

from localpilot import native_avatar


def test_illustrated_sprite_chunks_reconstruct_expected_png():
    data = native_avatar._read_avatar_sprite_data()
    assert data is not None
    assert len(data) == native_avatar._SPRITE_B64_LENGTH

    raw = base64.b64decode(data, validate=True)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    # PNG IHDR stores width/height as big-endian uint32 at bytes 16..24.
    assert int.from_bytes(raw[16:20], "big") == native_avatar.AVATAR_SIZE * 2
    assert int.from_bytes(raw[20:24], "big") == native_avatar.AVATAR_SIZE * 9


def test_illustrated_sprite_maps_every_existing_runtime_state():
    required = {
        "idle",
        "listening",
        "thinking",
        "working",
        "speaking",
        "success",
        "uncertain",
        "error",
        "restarting",
        "sleeping",
        "offline",
    }
    assert required <= set(native_avatar._SPRITE_ROWS)
    assert native_avatar._SPRITE_ROWS["thinking"] == 2
    assert native_avatar._SPRITE_ROWS["working"] == 4
    assert native_avatar._SPRITE_ROWS["success"] == 5
    assert native_avatar._SPRITE_ROWS["error"] == 6
    assert native_avatar._SPRITE_ROWS["uncertain"] == 7


def test_sprite_materialization_is_deterministic_and_gitignored(tmp_path, monkeypatch):
    source = native_avatar._read_avatar_sprite_data()
    assert source is not None

    asset_dir = tmp_path / "avatar"
    asset_dir.mkdir()
    monkeypatch.setattr(native_avatar, "_asset_dir", lambda: asset_dir)

    first = native_avatar._materialize_webview_sprite(source)
    assert first == asset_dir / "sprite.png"
    assert first is not None and first.exists()
    first_bytes = first.read_bytes()

    second = native_avatar._materialize_webview_sprite(source)
    assert second == first
    assert second.read_bytes() == first_bytes

    ignore_file = Path(native_avatar.__file__).resolve().parent / "webview" / "avatar" / ".gitignore"
    ignored = ignore_file.read_text(encoding="utf-8").splitlines()
    assert "sprite.png" in ignored
    assert "sprite.png.tmp" in ignored


def test_webview_loads_state_driven_illustrated_avatar_with_pixel_fallback():
    webview_dir = Path(native_avatar.__file__).resolve().parent / "webview"
    index = (webview_dir / "index.html").read_text(encoding="utf-8")
    script = (webview_dir / "illustrated-avatar.js").read_text(encoding="utf-8")

    assert '<script src="app.js"></script>' in index
    assert '<script src="illustrated-avatar.js"></script>' in index
    assert 'document.documentElement.dataset.state' in script
    assert 'sprite.src = "avatar/sprite.png"' in script
    assert 'this.canvas.style.opacity = "0"' in script
    assert "pixel fallback" in script
    assert "FRAME_HOLD_MS = 360" in script
