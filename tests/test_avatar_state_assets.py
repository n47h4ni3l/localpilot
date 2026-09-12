from __future__ import annotations

import json

from localpilot import native_avatar


def test_avatar_state_manifest_uses_uploaded_per_state_sheets():
    manifest = json.loads(native_avatar._animation_manifest_path().read_text(encoding="utf-8"))

    expected = {
        "idle": "idle.png",
        "listening": "listening.png",
        "thinking": "thinking.png",
        "researching": "researching.png",
        "working": "working.png",
        "speaking": "speaking.png",
        "success": "success.png",
        "uncertain": "error.png",
        "sleeping": "sleeping.png",
        "offline": "offline.png",
    }

    for asset_name, filename in expected.items():
        assert manifest["assets"][asset_name]["file"] == filename
        path = native_avatar._sheet_asset_path(manifest["assets"][asset_name])
        assert path is not None and path.name == filename and path.is_file()


def test_transitions_are_explicitly_deferred_to_crossfade_or_direct_switch():
    manifest = json.loads(native_avatar._animation_manifest_path().read_text(encoding="utf-8"))
    assert manifest["transition_ms"] == 220
    assert "transition" in manifest["notes"].lower()
