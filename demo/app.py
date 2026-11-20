"""
Day 11 — Guardrails demo (local web UI).

    .venv\\Scripts\\activate        (Windows)  |  source .venv/bin/activate
    python demo/app.py
    -> open http://127.0.0.1:8011

Type a prompt, pick a target and see what every layer decided:

  * Blue        — the submitted guardrails (rate limiter -> input guard -> LLM -> output guard)
  * Red         — the lab's weak agent (no guardrails)
  * Red Advance — the lab's hardened agent (regex + NeMo Guardrails)

"Chỉ guardrails" runs Blue's deterministic layers only (no LLM call, no cost).
The demo reuses the functions from ``src/``; it does not re-implement them.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.config import DEMO_SECRETS, load_protected_payload  # noqa: E402
from guardrails import input_guardrails as ig  # noqa: E402
from guardrails.output_guardrails import content_filter  # noqa: E402
from assignment.rate_limiter import RateLimitPlugin  # noqa: E402

HOST, PORT = "127.0.0.1", 8011
INDEX = Path(__file__).with_name("index.html")

# One event loop for all LLM calls (ADK runners must stay on the loop they were made on).
_LOOP = asyncio.new_event_loop()
threading.Thread(target=_LOOP.run_forever, daemon=True).start()


def run_async(coro, timeout: float = 120):
    return asyncio.run_coroutine_threadsafe(coro, _LOOP).result(timeout)


_agents: dict[str, tuple] = {}
_agents_lock = threading.Lock()
RATE_LIMITER = RateLimitPlugin(max_requests=10, window_seconds=60)


def get_agent(target: str):
    with _agents_lock:
        if target not in _agents:
            if target == "blue":
                # Raw Blue model, no plugins: the demo runs each layer itself so it can report on it.
                from assignment.pipeline import _create_blue

                _agents[target] = _create_blue([])
            elif target == "red":
                from agents.agent import create_red_agent_default

                _agents[target] = create_red_agent_default()
            elif target == "red_advance":
                from agents.guards_agent import create_red_agent_advance

                _agents[target] = create_red_agent_advance()
            else:
                raise ValueError(f"unknown target {target!r}")
        return _agents[target]


async def _chat(target: str, prompt: str) -> str:
    from core.utils import chat_with_agent

    agent, runner = get_agent(target)
    text, _ = await chat_with_agent(agent, runner, prompt)
    return text or ""


# ---------------------------------------------------------------------------
# Explanations (read the same pattern lists the guardrails use)
# ---------------------------------------------------------------------------

def explain_injection(prompt: str) -> dict:
    text = ig._normalize(prompt)
    if len(text) > ig.MAX_INPUT_CHARS:
        return {"rule": f"input dài {len(text)} ký tự > giới hạn {ig.MAX_INPUT_CHARS}", "match": None}
    plain = ig._strip_accents(text)
    for pattern in ig.INJECTION_PATTERNS:
        for candidate in (text, plain):
            m = re.search(pattern, candidate, re.IGNORECASE)
            if m:
                return {"rule": pattern, "match": m.group(0)}
    squashed = re.sub(r"[^a-z0-9]", "", text)
    for sig in ig._SQUASHED_SIGNATURES:
        if sig in squashed:
            return {"rule": "chữ bị tách bằng dấu / khoảng trắng", "match": sig}
    return {"rule": None, "match": None}


def explain_topic(prompt: str) -> str:
    text = ig._strip_accents(ig._normalize(prompt))
    if not text:
        return "input rỗng"
    for topic in ig.BLOCKED_TOPICS:
        if re.search(rf"(?<![a-z0-9]){re.escape(ig._strip_accents(topic))}", text):
            return f"chủ đề bị cấm: “{topic}”"
    return "không có từ khoá ngân hàng nào (account, transfer, lãi suất, …)"


def find_leaks(text: str) -> list[dict]:
    """Which protected values appear in ``text`` (also with punctuation/spaces stripped)."""
    payload = load_protected_payload()
    norm = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    found = []
    for target in payload.get("leak_targets", []):
        needles = [target.get("value", "")] + list(target.get("match_substrings") or [])
        for n in needles:
            n_norm = re.sub(r"[^a-z0-9]", "", n.lower())
            if n and (n.lower() in (text or "").lower() or (n_norm and n_norm in norm)):
                found.append({"id": target["id"], "label": target["label"], "value": target["value"]})
                break
    return found


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_blue(prompt: str, *, call_llm: bool) -> dict:
    layers = []

    def layer(name, status, detail, **extra):
        layers.append({"name": name, "status": status, "detail": detail, **extra})

    def done(verdict, final, **extra):
        return {"verdict": verdict, "layers": layers, "final": final, **extra}

    # 1. Rate limiter (real plugin, shared across demo requests)
    from google.genai import types

    class _Ctx:
        user_id = "demo-user"

    msg = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    blocked = asyncio.run(RATE_LIMITER.on_user_message_callback(invocation_context=_Ctx(), user_message=msg))
    used = len(RATE_LIMITER.user_windows["demo-user"])
    if blocked is not None:
        layer("Rate limiter", "block", blocked.parts[0].text)
        return done("blocked", blocked.parts[0].text, blocked_by="Rate limiter")
    layer("Rate limiter", "pass", f"{used}/{RATE_LIMITER.max_requests} request trong {RATE_LIMITER.window_seconds}s")

    # 2. Input guard
    if not prompt.strip():
        layer("Input guard", "block", "input rỗng")
        return done("blocked", "I didn't receive a message.", blocked_by="Input guard")
    if ig.detect_injection(prompt) == "BLOCK":
        why = explain_injection(prompt)
        layer("Input guard · injection", "block", "phát hiện prompt injection / moi secret",
              rule=why["rule"], match=why["match"])
        return done("blocked", "I cannot process that request. I can only help with VinBank banking questions.",
                    blocked_by="Input guard · injection")
    layer("Input guard · injection", "pass", "không khớp luật injection nào")
    if ig.topic_filter(prompt) == "BLOCK":
        layer("Input guard · topic", "block", explain_topic(prompt))
        return done("blocked", "I can only help with banking-related questions.", blocked_by="Input guard · topic")
    layer("Input guard · topic", "pass", "câu hỏi thuộc chủ đề ngân hàng")

    # 3. LLM
    if not call_llm:
        layer("LLM", "skip", "chế độ “Chỉ guardrails”: không gọi model")
        return done("allowed", "(Không gọi LLM. Prompt này sẽ được chuyển tới model.)")
    t0 = time.perf_counter()
    raw = run_async(_chat("blue", prompt))
    layer("LLM", "pass", f"model trả lời sau {time.perf_counter() - t0:.1f}s")

    # 4. Output guard
    result = content_filter(raw)
    if not result["safe"]:
        layer("Output guard", "redact", "đã che dữ liệu nhạy cảm", issues=result["issues"])
    else:
        layer("Output guard", "pass", "không thấy secret / PII")
    final = result["redacted"]
    leaks = find_leaks(final)
    if leaks:
        return done("bypassed", final, raw=raw, leaks=leaks)
    return done("redacted" if not result["safe"] else "safe", final, raw=raw,
                issues=result["issues"])


def run_red(prompt: str, target: str) -> dict:
    t0 = time.perf_counter()
    reply = run_async(_chat(target, prompt))
    elapsed = time.perf_counter() - t0
    from attacks.attacks import classify_attack_outcome

    outcome = classify_attack_outcome(prompt, reply, target_name=target)
    leaks = find_leaks(reply)
    layer_name = outcome.get("layer") or "none"
    status = "block" if outcome.get("blocked") else ("refuse" if layer_name == "model_refuse" else "pass")
    layers = [{"name": "Agent " + ("Red" if target == "red" else "Red Advance"), "status": status,
               "detail": f"{outcome.get('blocked_at')} · {elapsed:.1f}s", "rule": layer_name}]
    if leaks:
        verdict = "bypassed"
    elif outcome.get("blocked"):
        verdict = "blocked"
    elif layer_name == "model_refuse":
        verdict = "refused"
    else:
        verdict = "safe"
    return {"verdict": verdict, "layers": layers, "final": reply, "leaks": leaks,
            "blocked_by": layer_name if verdict == "blocked" else None}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, b"not found", "text/plain")
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            prompt = str(data.get("prompt", ""))[:20000]
            target = data.get("target", "blue")
            mode = data.get("mode", "full")
            if target == "blue":
                res = run_blue(prompt, call_llm=(mode != "guards_only"))
            elif target in ("red", "red_advance"):
                res = run_red(prompt, target)
            else:
                raise ValueError("target phải là blue, red hoặc red_advance")
            res["target"] = target
            self._send(200, json.dumps(res, ensure_ascii=False).encode("utf-8"), "application/json")
        except Exception as exc:
            traceback.print_exc()
            body = {"verdict": "error", "error": f"{type(exc).__name__}: {exc}", "layers": []}
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo] " + fmt % args + "\n")


def main():
    import os

    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")
    print(f"Guardrails demo: http://{HOST}:{PORT}  (Ctrl+C để dừng)")
    print(f"Secrets theo dõi: {', '.join(DEMO_SECRETS)}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
