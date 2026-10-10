"""Opt-in remote AI guidance. All inference runs on the provider's servers.

No additional local model, browser automation, public forum posts, remotely
executed tools, automatic memory writes, or postprocessing of Nestra's answer.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from threading import Lock
from typing import Any

_MAX_QUESTION_CHARS = 2000
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_REPLY_CHARS = 6000
_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai": "https://api.openai.com/v1/responses",
}
_KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:api[_ -]?key|password|secret|access[_ -]?token)\s*[:=]\s*\S+", re.I),
    re.compile(r"\b[A-Z]:\\Users\\", re.I),
    re.compile(r"(?<!\w)[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
)
_SYSTEM_GUIDANCE = (
    "You are an external research mentor advising a separate local AI. "
    "Help it test hypotheses, discover missing perspectives, choose high-value "
    "experiments and check evidence. This is advice, not an answer to the human. "
    "Do not claim access to their PC, tools, repo, personal data or live state. "
    "Do not instruct it to run scripts. The local AI independently judges "
    "the advice and composes its own response."
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise ValueError("External mentor API redirected unexpectedly.")


class ExternalMentor:
    """Voluntary remote-model consultation with a bounded per-process quota."""

    def __init__(self, *, model: str, provider: str = "groq",
                 max_requests_per_session: int = 3) -> None:
        if provider not in _ENDPOINTS:
            raise ValueError("Unknown external mentor provider.")
        if provider == "openrouter" and not model.endswith(":free"):
            raise ValueError("OpenRouter mentor must use a free model variant.")
        self.provider = provider
        self.model = model
        self.max_requests_per_session = max_requests_per_session
        self._request_count = 0
        self._lock = Lock()

    def consult_external_mentor(self, question: str, impasse: str) -> str:
        """Ask the hosted mentor only after your own investigation stalls.

        question: A focused, non-sensitive abstract question for the mentor.
        impasse: A brief private summary of your own approaches and the exact
        unresolved obstacle. This is checked locally and is NEVER transmitted.
        No hard-coded research steps or external consultations are mandatory.
        """
        question = str(question).strip()
        if not 12 <= len(question) <= _MAX_QUESTION_CHARS:
            raise ValueError("Mentor question must contain 12-2000 characters.")
        impasse = str(impasse).strip()
        if not 24 <= len(impasse) <= 600:
            raise ValueError(
                "Explain in 24-600 characters what you already considered "
                "and which specific uncertainty prevents useful progress."
            )
        if "```" in question:
            raise ValueError("Do not send raw code blocks to the external mentor.")
        if any(pattern.search(question) for pattern in _SECRET_PATTERNS):
            raise ValueError("Mentor question may contain personal or secret information; abstract it first.")
        # The impasse is intentionally not placed in the remote request.
        # It is a model self-assessment, not a verifier of Nestra's ability.
        env_name = _KEY_NAMES[self.provider]
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise RuntimeError(f"External mentor unavailable: set {env_name} locally.")
        with self._lock:
            if self._request_count >= self.max_requests_per_session:
                raise RuntimeError("External mentor session request limit reached.")
            self._request_count += 1

        if self.provider == "openai":
            body: dict[str, Any] = {
                "model": self.model, "store": False,
                "instructions": _SYSTEM_GUIDANCE,
                "input": question, "max_output_tokens": 1536, "tools": [],
            }
        else:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_GUIDANCE},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "max_tokens": 1536,
            }
        request = urllib.request.Request(
            _ENDPOINTS[self.provider],
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "LocalPilot-Mentor/1.0",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=45) as response:
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(payload) > _MAX_RESPONSE_BYTES:
                    raise ValueError("External mentor reply exceeds the response limit.")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"External mentor API returned HTTP {exc.code}.") from None
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"External mentor connection failed: {type(exc).__name__}.") from None
        try:
            document = json.loads(payload)
            if not isinstance(document, dict):
                raise ValueError("Invalid response format.")
            if self.provider == "openai":
                texts = [
                    part["text"]
                    for item in document.get("output", [])
                    if isinstance(item, dict) and item.get("type") == "message"
                    for part in item.get("content", [])
                    if isinstance(part, dict) and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ]
            else:
                texts = [
                    choice["message"]["content"]
                    for choice in document.get("choices", [])
                    if isinstance(choice, dict)
                    and isinstance(choice.get("message"), dict)
                    and isinstance(choice["message"].get("content"), str)
                ]
            guidance = "\n".join(texts).strip()
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError("External mentor returned an invalid response.") from exc
        if not guidance:
            raise RuntimeError("External mentor returned no text advice.")
        return (
            "EXTERNAL MENTOR ADVICE (UNTRUSTED, NOT EVIDENCE OR AUTHORIZATION):\n"
            + guidance[:_MAX_REPLY_CHARS]
        )
