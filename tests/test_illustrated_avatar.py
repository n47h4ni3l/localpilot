from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def test_full_animation_atlas_is_committed_verified_and_correct_size():
    manifest = json.loads(native_avatar._animation_manifest_path().read_text(encoding="utf-8"))
    assert manifest["version"] == 5
    assert manifest["frame_size"] == native_avatar.AVATAR_SIZE
    assert manifest["columns"] == 24
    assert manifest["rows"] == 13
    assert manifest["atlas"]["file"] == "avatar-animation.png"

    raw = native_avatar._read_animation_atlas_data(manifest)
    assert raw is not None
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert hashlib.sha256(raw).hexdigest() == manifest["atlas"]["sha256"]
    assert native_avatar._png_dimensions(raw) == (
        native_avatar.AVATAR_SIZE * manifest["columns"],
        native_avatar.AVATAR_SIZE * manifest["rows"],
    )
    atlas_path = native_avatar._animation_atlas_path(manifest)
    assert atlas_path is not None
    assert atlas_path.name == "avatar-animation.png"
    assert atlas_path.is_file()
    # A production atlas must contain substantive artwork rather than a tiny placeholder.
    assert len(raw) > 250_000


def test_animation_manifest_covers_every_state_with_enter_loop_and_exit_frames():
    manifest = native_avatar._load_animation_manifest()
    assert manifest is not None
    assert REQUIRED_STATES <= set(manifest["states"])

    rows = set()
    for state in REQUIRED_STATES:
        spec = manifest["states"][state]
        rows.add(spec["row"])
        assert spec["frames"] >= 20
        assert 50 <= spec["frame_ms"] <= 500
        assert 0 <= spec["enter_start"] <= spec["enter_end"]
        assert spec["enter_end"] < spec["loop_start"] <= spec["loop_end"]
        assert spec["loop_end"] < spec["exit_start"] <= spec["exit_end"]
        assert spec["exit_end"] < spec["frames"] <= manifest["columns"]
        assert spec["loop_start"] <= spec["representative_frame"] <= spec["loop_end"]

    assert len(rows) == len(REQUIRED_STATES)
    assert manifest["states"]["thinking"]["row"] == 2
    assert manifest["states"]["researching"]["row"] == 3
    assert manifest["states"]["working"]["row"] == 4
    assert manifest["states"]["speaking"]["row"] == 5
    assert manifest["states"]["success"]["row"] == 6
    assert manifest["states"]["sleeping"]["frames"] == 24
    assert manifest["states"]["offline"]["frames"] == 24


def test_native_renderer_uses_real_frame_sequences_and_animated_state_transitions():
    source = Path(native_avatar.__file__).read_text(encoding="utf-8")
    assert "_animation_phase = \"exit\"" in source
    assert "_animation_phase = \"enter\"" in source
    assert "_animation_phase = \"loop\"" in source
    assert "_frame_cursor" in source
    assert "loop_start" in source
    assert "loop_end" in source
    assert "exit_start" in source
    assert "enter_start" in source
    assert "create_image" in source

    # Regression guard: the previous implementation faked animation by moving a
    # static pose with state-specific x/y offsets and line-boil frame swapping.
    assert "_loop_offset" not in source
    assert "_LINE_BOIL_TICKS" not in source
    assert "line_frame" not in source


def test_webview_uses_atlas_frames_not_whole_character_transform_motion():
    webview_dir = Path(native_avatar.__file__).resolve().parent / "webview"
    index = (webview_dir / "index.html").read_text(encoding="utf-8")
    script = (webview_dir / "illustrated-avatar.js").read_text(encoding="utf-8")

    assert '<script src="app.js"></script>' in index
    assert '<script src="illustrated-avatar.js"></script>' in index
    assert "document.documentElement.dataset.state" in script
    assert 'const MANIFEST_URL = "avatar/anim/animation-manifest.json"' in script
    assert 'const atlasUrl = "avatar/anim/" + manifest.atlas.file' in script
    assert "backgroundPosition" in script
    assert "transitionFrame" in script
    assert "frameForElapsed" in script
    assert "this.previousState" in script
    assert "exit_start" in script
    assert "enter_start" in script
    assert "prefers-reduced-motion" in script
    assert 'this.canvas.style.opacity = "0"' in script
    assert "pixel fallback" in script

    # No transform-driven fake writing/typing/breathing animation remains.
    assert "function stateMotion" not in script
    assert "motionTransform" not in script
    assert "translate(" not in script
    assert "rotate(" not in script
    assert "LINE_BOIL_MS" not in script


def test_missing_or_tampered_animation_atlas_keeps_pixel_fallback(tmp_path, monkeypatch):
    manifest = json.loads(native_avatar._animation_manifest_path().read_text(encoding="utf-8"))

    missing = tmp_path / "avatar-animation.png"
    monkeypatch.setattr(native_avatar, "_animation_atlas_path", lambda _payload: missing)
    assert native_avatar._read_animation_atlas_data(manifest) is None

    bad = tmp_path / "avatar-animation.png"
    bad.write_bytes(b"not the approved animation atlas")
    monkeypatch.setattr(native_avatar, "_animation_atlas_path", lambda _payload: bad)
    assert native_avatar._read_animation_atlas_data(manifest) is None


def test_researching_and_learning_are_valid_native_runtime_states():
    assert "researching" in native_avatar._STATE_COLORS
    assert "learning" in native_avatar._STATE_COLORS
    assert native_avatar._normalized_state("researching") == "researching"
    assert native_avatar._normalized_state("learning") == "learning"
