from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from localpilot.broker import load_or_create_broker_token
from localpilot.config import Config, load_config


def _markdown_segments(content: str) -> list[tuple[str, str]]:
    """Parse a small safe Markdown subset into Tk text-tag segments."""
    segments: list[tuple[str, str]] = []
    fenced = False
    inline = re.compile(r"(\*\*[^*\n]+\*\*|`[^`\n]+`)")
    for raw_line in str(content).splitlines(keepends=True):
        line = raw_line
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            segments.append((line, "code"))
            continue
        tag = "body"
        heading = re.match(r"^\s{0,3}#{1,6}\s+", line)
        if heading:
            line = line[heading.end():]
            tag = "heading"
        bullet = re.match(r"^(\s*)[-*+]\s+", line)
        if bullet:
            line = bullet.group(1) + "• " + line[bullet.end():]
        position = 0
        for match in inline.finditer(line):
            if match.start() > position:
                segments.append((line[position:match.start()], tag))
            token = match.group(0)
            segments.append((token[2:-2], "bold") if token.startswith("**") else (token[1:-1], "code"))
            position = match.end()
        if position < len(line):
            segments.append((line[position:], tag))
    return segments or [("", "body")]


class BrokerClient:
    def __init__(self, root: str | Path, config: Config) -> None:
        self.config = config
        self.base_url = f"http://{config.desktop.host}:{config.desktop.port}"
        self.token = load_or_create_broker_token(root, config)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 10.0,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def healthy(self) -> bool:
        try:
            return bool(self.request("GET", "/health", timeout=0.5, authenticated=False).get("ok"))
        except (OSError, urllib.error.URLError, ValueError):
            return False


def ensure_broker(
    root: str | Path,
    config: Config,
    *,
    config_path: str | Path | None = None,
    timeout: float = 10.0,
) -> BrokerClient:
    root = Path(root).resolve()
    client = BrokerClient(root, config)
    if client.healthy():
        return client
    argv = [sys.executable, "-m", "localpilot.broker", "--root", str(root)]
    if config_path:
        argv.extend(["--config", str(Path(config_path).resolve())])
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
        )
    else:
        start_new_session = True
    subprocess.Popen(
        argv,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        creationflags=creationflags,
        start_new_session=start_new_session,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.healthy():
            return client
        time.sleep(0.1)
    raise RuntimeError("LocalPilot broker did not become ready")


class PixelPilot:
    """Small original non-human pixel body driven entirely by runtime state events."""

    PALETTES = {
        "idle": ("#7fe3c3", "#173443"),
        "thinking": ("#f7d774", "#5b4520"),
        "working": ("#75b9ff", "#153b66"),
        "speaking": ("#c39bff", "#422a63"),
        "success": ("#8bea8b", "#17491f"),
        "error": ("#ff8b8b", "#5e1d29"),
        "restarting": ("#9ca9bd", "#283342"),
    }

    def __init__(self, canvas: Any) -> None:
        self.canvas = canvas
        self.state = "restarting"
        self.frame = 0
        self.draw()

    def set_state(self, state: str) -> None:
        normalized = state if state in self.PALETTES else "error"
        self.state = normalized
        self.frame += 1
        self.draw()

    def _pixel(self, x: int, y: int, color: str, scale: int = 7) -> None:
        self.canvas.create_rectangle(
            x * scale,
            y * scale,
            (x + 1) * scale,
            (y + 1) * scale,
            fill=color,
            outline=color,
        )

    def draw(self) -> None:
        self.canvas.delete("all")
        body, ink = self.PALETTES[self.state]
        accent = "#f6fbff"
        for x, y in ((6, 1), (6, 2), (5, 3), (6, 3), (7, 3)):
            self._pixel(x, y, ink if y < 3 else body)
        for y in range(4, 10):
            start, end = (3, 10) if y in {5, 6, 7, 8} else (4, 9)
            for x in range(start, end):
                self._pixel(x, y, body)
        eye_y = 6
        if self.state == "restarting":
            for x in (5, 8):
                self._pixel(x, eye_y, ink)
                self._pixel(x, eye_y + 1, ink)
        elif self.state == "error":
            for x in (5, 8):
                self._pixel(x, eye_y, ink)
            self._pixel(6, 8, ink)
            self._pixel(7, 8, ink)
        else:
            self._pixel(5, eye_y, accent)
            self._pixel(8, eye_y, accent)
            if self.state == "success":
                self._pixel(6, 8, ink)
                self._pixel(7, 8, ink)
            elif self.state == "speaking":
                self._pixel(6, 8, ink)
                self._pixel(7, 8, ink)
                self._pixel(7, 9, ink)
        arm_y = 7
        if self.state == "working":
            for x, y in ((2, 6), (1, 5), (10, 6), (11, 5)):
                self._pixel(x, y, body)
        else:
            self._pixel(2, arm_y, body)
            self._pixel(10, arm_y, body)
        for x, y in ((4, 10), (4, 11), (8, 10), (8, 11)):
            self._pixel(x, y, ink)


class DesktopChat:
    def __init__(self, client: BrokerClient, config: Config) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self.tk = tk
        self.client = client
        self.config = config
        self.root = tk.Tk()
        self.root.title(f"{config.agent.name} · Desktop")
        self.root.geometry("980x700")
        self.root.minsize(760, 520)
        self.root.configure(bg="#0e1722")
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.session_id: str | None = None
        self.after_event_id = 0
        self.sessions: list[dict[str, Any]] = []

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=10)

        sidebar = tk.Frame(self.root, bg="#111f2e", width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(
            sidebar,
            text="CONVERSATIONS",
            bg="#111f2e",
            fg="#8aa4bd",
            font=("Segoe UI Semibold", 9),
            pady=14,
        ).pack(fill="x")
        self.session_list = tk.Listbox(
            sidebar,
            bg="#111f2e",
            fg="#e7f0f8",
            selectbackground="#24445f",
            selectforeground="#ffffff",
            borderwidth=0,
            highlightthickness=0,
            activestyle="none",
        )
        self.session_list.pack(fill="both", expand=True, padx=8)
        self.session_list.bind("<<ListboxSelect>>", self._select_session)
        tk.Button(
            sidebar,
            text="＋ New conversation",
            command=self._new_session,
            bg="#1d3448",
            fg="#e7f0f8",
            activebackground="#294b67",
            activeforeground="#ffffff",
            relief="flat",
            padx=8,
            pady=10,
        ).pack(fill="x", padx=10, pady=12)

        main = tk.Frame(self.root, bg="#0e1722")
        main.pack(side="left", fill="both", expand=True)
        header = tk.Frame(main, bg="#142334", height=96)
        header.pack(fill="x")
        header.pack_propagate(False)
        canvas = tk.Canvas(header, width=92, height=92, bg="#142334", highlightthickness=0)
        canvas.pack(side="left", padx=(18, 8), pady=2)
        self.avatar = PixelPilot(canvas)
        identity = tk.Frame(header, bg="#142334")
        identity.pack(side="left", pady=20)
        tk.Label(
            identity,
            text=config.agent.name,
            bg="#142334",
            fg="#f3f7fb",
            font=("Segoe UI Semibold", 17),
        ).pack(anchor="w")
        self.status = tk.Label(
            identity,
            text="Reconnecting to local runtime…",
            bg="#142334",
            fg="#93abc0",
            font=("Segoe UI", 10),
        )
        self.status.pack(anchor="w")

        self.chat = tk.Text(
            main,
            wrap="word",
            bg="#0e1722",
            fg="#dce8f2",
            insertbackground="#ffffff",
            borderwidth=0,
            highlightthickness=0,
            padx=24,
            pady=18,
            spacing1=2,
            spacing3=10,
            state="disabled",
        )
        self.chat.tag_configure("user_label", foreground="#75b9ff", font=("Segoe UI Semibold", 10))
        self.chat.tag_configure("assistant_label", foreground="#7fe3c3", font=("Segoe UI Semibold", 10))
        self.chat.tag_configure("body", foreground="#dce8f2", font=("Segoe UI", 11))
        self.chat.tag_configure("heading", foreground="#f3f7fb", font=("Segoe UI Semibold", 12))
        self.chat.tag_configure("bold", foreground="#f3f7fb", font=("Segoe UI Semibold", 11))
        self.chat.tag_configure("code", foreground="#f7d774", font=("Consolas", 10))
        self.chat.tag_configure("pending", foreground="#8aa4bd", font=("Segoe UI", 11, "italic"))

        composer = tk.Frame(main, bg="#142334", padx=14, pady=12)
        composer.pack(side="bottom", fill="x")
        self.input = tk.Text(
            composer,
            height=3,
            wrap="word",
            bg="#1b2e40",
            fg="#f4f8fb",
            insertbackground="#ffffff",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#2c485f",
            highlightcolor="#75b9ff",
            padx=10,
            pady=8,
        )
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Control-Return>", lambda event: self._send())
        self.send_button = tk.Button(
            composer,
            text="Send",
            command=self._send,
            bg="#58a6e7",
            fg="#07131d",
            activebackground="#75b9ff",
            relief="flat",
            width=10,
            pady=12,
            font=("Segoe UI Semibold", 10),
        )
        self.send_button.pack(side="left", padx=(12, 0))
        self.chat.pack(side="top", fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        threading.Thread(target=self._bootstrap, daemon=True).start()
        threading.Thread(target=self._poll_events, daemon=True).start()
        self.root.after(50, self._drain_queue)

    def _bootstrap(self) -> None:
        try:
            sessions = self.client.request("GET", "/v1/sessions")["sessions"]
            if not sessions:
                sessions = [self.client.request("POST", "/v1/sessions", {"title": "New conversation"})["session"]]
            self.queue.put(("sessions", (sessions, sessions[0]["id"])))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _poll_events(self) -> None:
        while not self.stop_event.is_set():
            try:
                query = urllib.parse.urlencode(
                    {
                        "after": self.after_event_id,
                        "wait": 20,
                        **({"session_id": self.session_id} if self.session_id else {}),
                    }
                )
                events = self.client.request("GET", f"/v1/events?{query}", timeout=25)["events"]
                for event in events:
                    self.after_event_id = max(self.after_event_id, int(event["id"]))
                    self.queue.put(("event", event))
            except Exception:
                self.queue.put(("state", "restarting"))
                time.sleep(1.0)

    def _drain_queue(self) -> None:
        refresh = False
        try:
            while True:
                kind, value = self.queue.get_nowait()
                if kind == "sessions":
                    sessions, selected_id = value
                    self.sessions = list(sessions)
                    self.session_list.delete(0, "end")
                    for session in self.sessions:
                        self.session_list.insert("end", session["title"])
                    if self.sessions:
                        target_id = selected_id or self.session_id or self.sessions[0]["id"]
                        index = next(
                            (i for i, session in enumerate(self.sessions) if session["id"] == target_id),
                            0,
                        )
                        changed = self.session_id != self.sessions[index]["id"]
                        self.session_id = self.sessions[index]["id"]
                        self.session_list.selection_clear(0, "end")
                        self.session_list.selection_set(index)
                        self.session_list.see(index)
                        if changed:
                            self.after_event_id = 0
                        refresh = True
                elif kind == "event":
                    event_type = value["type"]
                    payload = value.get("payload") or {}
                    if event_type == "runtime.state":
                        self._set_state(str(payload.get("state") or "idle"), payload)
                    elif event_type.startswith("tool."):
                        self._set_state("working", payload)
                    elif event_type in {"runtime.error", "message.failed"}:
                        self._set_state("error", payload)
                    if event_type.startswith("message.") or event_type == "assistant.delta":
                        refresh = True
                    if event_type in {"message.completed", "message.failed"}:
                        self._refresh_sessions()
                elif kind == "state":
                    self._set_state(str(value), {})
                elif kind == "refresh":
                    refresh = True
                elif kind == "render":
                    session_id, messages = value
                    if session_id == self.session_id:
                        self._render_messages(messages)
                elif kind == "error":
                    self._set_state("error", {"message": value})
        except queue.Empty:
            pass
        if refresh:
            self._refresh_messages()
        if not self.stop_event.is_set():
            self.root.after(80, self._drain_queue)

    def _set_state(self, state: str, payload: dict[str, Any]) -> None:
        state = "error" if state in {"uncertain", "unavailable"} else state
        self.avatar.set_state(state)
        descriptions = {
            "idle": "Ready · broker connected",
            "thinking": "Thinking…",
            "working": f"Working · {payload.get('tool', 'local worker')}",
            "speaking": "Speaking…",
            "success": "Done",
            "error": f"Uncertain · {payload.get('message', 'runtime needs attention')}",
            "restarting": "Sleeping lightly · runtime reconnecting…",
        }
        self.status.configure(text=descriptions.get(state, state.capitalize()))

    def _refresh_messages(self) -> None:
        if not self.session_id:
            return
        session_id = self.session_id

        def fetch() -> None:
            try:
                messages = self.client.request("GET", f"/v1/sessions/{session_id}/messages")["messages"]
                self.queue.put(("render", (session_id, messages)))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=fetch, daemon=True).start()

    def _render_messages(self, messages: list[dict[str, Any]]) -> None:
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        for message in messages:
            role = message["role"]
            label = "YOU" if role == "user" else self.config.agent.name.upper()
            tag = "user_label" if role == "user" else "assistant_label"
            self.chat.insert("end", label + "\n", tag)
            content = str(message["content"])
            body_tag = "pending" if message["status"] == "streaming" and not content else "body"
            if body_tag == "pending":
                self.chat.insert("end", "…\n\n", body_tag)
            else:
                for text, markdown_tag in _markdown_segments(content):
                    self.chat.insert("end", text, markdown_tag)
                self.chat.insert("end", "\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _select_session(self, _event: Any) -> None:
        selection = self.session_list.curselection()
        if not selection:
            return
        self.session_id = self.sessions[int(selection[0])]["id"]
        self.after_event_id = 0
        self._refresh_messages()

    def _new_session(self) -> None:
        def create() -> None:
            try:
                session = self.client.request("POST", "/v1/sessions", {"title": "New conversation"})["session"]
                sessions = self.client.request("GET", "/v1/sessions")["sessions"]
                self.queue.put(("sessions", (sessions, session["id"])))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=create, daemon=True).start()

    def _send(self) -> None:
        content = self.input.get("1.0", "end-1c").strip()
        if not content or not self.session_id:
            return
        session_id = self.session_id
        self.input.delete("1.0", "end")
        self._set_state("thinking", {})

        def send() -> None:
            try:
                self.client.request(
                    "POST",
                    f"/v1/sessions/{session_id}/messages",
                    {"content": content},
                )
                self.queue.put(("refresh", None))
                self._refresh_sessions(selected_id=session_id)
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=send, daemon=True).start()

    def _refresh_sessions(self, *, selected_id: str | None = None) -> None:
        target_id = selected_id or self.session_id

        def fetch() -> None:
            try:
                sessions = self.client.request("GET", "/v1/sessions")["sessions"]
                self.queue.put(("sessions", (sessions, target_id)))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=fetch, daemon=True).start()

    def _close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(root: str | Path, config_path: str | Path | None = None) -> None:
    config = load_config(config_path)
    client = ensure_broker(root, config, config_path=config_path)
    DesktopChat(client, config).run()
