from __future__ import annotations

import hashlib
from pathlib import Path

from localpilot import native_avatar


def test_illustrated_state_assets_are_committed_and_match_source_hashes():
    assert set(native_avatar._STATE_ASSET_SHA256) == set(range(9))

    for row, expected in native_avatar._STATE_ASSET_SHA256.items():
        raw = native_avatar._read_state_asset(row)
        assert raw is not None
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(raw).hexdigest() == expected

        # Every state asset contains two 128x128 independent redraws side by side.
        # PNG IHDR stores width/height as big-endian uint32 at bytes 16..24.
        assert int.from_bytes(raw[16:20], "big") == native_avatar.AVATAR_SIZE * 2
        assert int.from_bytes(raw[20:24], "big") == native_avatar.AVATAR_SIZE

        path = native_avatar._state_asset_path(row)
        assert path.name == f"state-{row}.png"
        assert path.is_file()


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
    assert "motion_age = max(0, self._state_age - _TRANSITION_TICKS)" in source

    # State animation is real movement, not just swapping the redraw frame.
    thinking_offsets = {native_avatar._loop_offset("thinking", frame) for frame in range(12)}
    working_offsets = {native_avatar._loop_offset("working", frame) for frame in range(12)}
    listening_offsets = {native_avatar._loop_offset("listening", frame) for frame in range(12)}
    assert len(thinking_offsets) > 1
    assert len(working_offsets) > 1
    assert len(listening_offsets) > 1


def test_webview_has_passive_line_boil_short_state_loops_and_animated_transitions():
    webview_dir = Path(native_avatar.__file__).resolve().parent / "webview"
    index = (webview_dir / "index.html").read_text(encoding="utf-8")
    script = (webview_dir / "illustrated-avatar.js").read_text(encoding="utf-8")

    assert '<script src="app.js"></script>' in index
    assert '<script src="illustrated-avatar.js"></script>' in index
    assert "document.documentElement.dataset.state" in script
    assert 'return "avatar/state-" + rowForState(state) + ".png";' in script
    assert 'image.src = "avatar/state-" + row + ".png";' in script
    assert "Promise.all" in script
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
    assert "stateMotion(this.state, now - this.stateStart - TRANSITION_MS)" in script
    assert "prefers-reduced-motion" in script


def test_missing_or_tampered_state_asset_keeps_pixel_fallback(tmp_path, monkeypatch):
    missing = tmp_path / "missing.png"
    monkeypatch.setattr(native_avatar, "_state_asset_path", lambda _row: missing)
    assert native_avatar._read_state_asset(0) is None

    bad = tmp_path / "state-0.png"
    bad.write_bytes(b"not the approved state asset")
    monkeypatch.setattr(native_avatar, "_state_asset_path", lambda _row: bad)
    assert native_avatar._read_state_asset(0) is None
