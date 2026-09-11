from __future__ import annotations

import hashlib
from pathlib import Path

from localpilot import native_avatar


def test_illustrated_sprite_is_committed_and_matches_source_hash():
    raw = native_avatar._read_avatar_sprite_data()
    assert raw is not None
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert hashlib.sha256(raw).hexdigest() == native_avatar._SPRITE_SHA256

    # PNG IHDR stores width/height as big-endian uint32 at bytes 16..24.
    assert int.from_bytes(raw[16:20], "big") == native_avatar.AVATAR_SIZE * 2
    assert int.from_bytes(raw[20:24], "big") == native_avatar.AVATAR_SIZE * 9
    assert native_avatar._sprite_path().name == "sprite.png"
    assert native_avatar._sprite_path().is_file()


def test_illustrated_sprite_maps_every_existing_runtime_state_and_future_personality_states():
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
        "researching",
        "learning",
    }
    assert required <= set(native_avatar._SPRITE_ROWS)
    assert native_avatar._SPRITE_ROWS["thinking"] == 2
    assert native_avatar._SPRITE_ROWS["researching"] == 3
    assert native_avatar._SPRITE_ROWS["working"] == 4
    assert native_avatar._SPRITE_ROWS["success"] == 5
    assert native_avatar._SPRITE_ROWS["error"] == 6
    assert native_avatar._SPRITE_ROWS["uncertain"] == 7
    assert native_avatar._SPRITE_ROWS["learning"] == 8


def test_native_motion_separates_line_boil_state_loop_and_transition():
    source = Path(native_avatar.__file__).read_text(encoding="utf-8")
    assert native_avatar._LINE_BOIL_TICKS >= 2
    assert native_avatar._TRANSITION_TICKS >= 2
    assert "line_frame = (self.frame // _LINE_BOIL_TICKS) % 2" in source
    assert "_loop_offset" in source
    assert "_transition_tick" in source

    # State animation is real movement, not just swapping the redraw frame.
    thinking_offsets = {native_avatar._loop_offset("thinking", frame) for frame in range(12)}
    working_offsets = {native_avatar._loop_offset("working", frame) for frame in range(12)}
    assert len(thinking_offsets) > 1
    assert len(working_offsets) > 1


def test_webview_has_passive_line_boil_short_state_loops_and_animated_transitions():
    webview_dir = Path(native_avatar.__file__).resolve().parent / "webview"
    index = (webview_dir / "index.html").read_text(encoding="utf-8")
    script = (webview_dir / "illustrated-avatar.js").read_text(encoding="utf-8")

    assert '<script src="app.js"></script>' in index
    assert '<script src="illustrated-avatar.js"></script>' in index
    assert 'document.documentElement.dataset.state' in script
    assert 'sprite.src = "avatar/sprite.png"' in script
    assert 'this.canvas.style.opacity = "0"' in script
    assert "pixel fallback" in script

    # These are deliberately separate concerns: redraw texture, pose movement,
    # and a dual-layer old->new state transition.
    assert "LINE_BOIL_MS = 520" in script
    assert "TRANSITION_MS = 280" in script
    assert "function stateMotion" in script
    assert "this.previousState" in script
    assert "illustrated-avatar-frame--previous" in script
    assert "illustrated-avatar-frame--current" in script
    assert "prefers-reduced-motion" in script


def test_missing_or_tampered_sprite_keeps_pixel_fallback(tmp_path, monkeypatch):
    missing = tmp_path / "missing.png"
    monkeypatch.setattr(native_avatar, "_sprite_path", lambda: missing)
    assert native_avatar._read_avatar_sprite_data() is None

    bad = tmp_path / "sprite.png"
    bad.write_bytes(b"not the approved sprite")
    monkeypatch.setattr(native_avatar, "_sprite_path", lambda: bad)
    assert native_avatar._read_avatar_sprite_data() is None
