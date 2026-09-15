from __future__ import annotations

import json
from pathlib import Path

from localpilot import native_avatar


def test_thinking_loop_is_slowed_and_uses_stronger_stabilization():
    manifest = json.loads(native_avatar._animation_manifest_path().read_text(encoding="utf-8"))
    thinking = manifest["assets"]["thinking"]
    state = manifest["states"]["thinking"]
    learning = manifest["states"]["learning"]

    assert (thinking["columns"], thinking["rows"], thinking["frames"]) == (6, 4, 24)
    assert state["frame_ms"] == 126
    assert learning["frame_ms"] == 126
    assert thinking["body_motion_retention"] == 0.02
    assert thinking["max_stabilization_px"] == 12

    # Idle keeps the established global centering behaviour.
    idle = manifest["assets"]["idle"]
    assert "body_motion_retention" not in idle
    assert "max_stabilization_px" not in idle


def test_native_and_webview_honor_per_sheet_stabilization_overrides():
    assert native_avatar._asset_body_motion_retention({}) == native_avatar._BODY_MOTION_RETENTION
    assert native_avatar._asset_max_stabilization_px({}) == native_avatar._MAX_STABILIZATION_PX
    assert native_avatar._asset_body_motion_retention({"body_motion_retention": 0.02}) == 0.02
    assert native_avatar._asset_max_stabilization_px({"max_stabilization_px": 12}) == 12

    script = (
        Path(native_avatar.__file__).resolve().parent / "webview" / "illustrated-avatar.js"
    ).read_text(encoding="utf-8")
    assert "body_motion_retention" in script
    assert "max_stabilization_px" in script
    assert "stabilizationLimit" in script
