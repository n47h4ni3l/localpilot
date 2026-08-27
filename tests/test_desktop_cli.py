from __future__ import annotations

import sys
from types import ModuleType

from localpilot import cli


def _fake_desktop_module(name: str, calls: list[tuple[object, object]]) -> ModuleType:
    module = ModuleType(name)
    module.main = lambda root, config: calls.append((root, config))
    return module


def test_desktop_parser_defaults_to_webview_and_offers_explicit_tkinter_fallback():
    default = cli.build_parser().parse_args(["desktop"])
    fallback = cli.build_parser().parse_args(["desktop", "--tkinter"])
    assert default.command == "desktop"
    assert default.tkinter is False
    assert fallback.tkinter is True


def test_desktop_command_launches_webview_by_default(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "localpilot.webview_app", _fake_desktop_module("localpilot.webview_app", calls))
    monkeypatch.setitem(sys.modules, "localpilot.desktop", _fake_desktop_module("localpilot.desktop", []))
    monkeypatch.setattr(sys, "argv", ["localpilot", "desktop"])

    cli.main()

    assert calls == [(cli._root(), None)]


def test_desktop_command_launches_tkinter_only_when_requested(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "localpilot.desktop", _fake_desktop_module("localpilot.desktop", calls))
    monkeypatch.setitem(sys.modules, "localpilot.webview_app", _fake_desktop_module("localpilot.webview_app", []))
    monkeypatch.setattr(sys, "argv", ["localpilot", "desktop", "--tkinter"])

    cli.main()

    assert calls == [(cli._root(), None)]
