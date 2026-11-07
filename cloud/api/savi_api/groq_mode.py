# -*- coding: utf-8 -*-
"""
SAVI Agent API — modo Groq (API compatible con OpenAI, capa gratuita).

Mismo contrato que el modo Claude: bucle de tool use sobre `savi_api.tools`, con la
respuesta final redactada por el modelo y anclada a lo que devuelven las herramientas.
Solo usa la librería estándar (urllib) para no añadir dependencias al despliegue.

Se activa con `GROQ_API_KEY`; el modelo se elige con `SAVI_GROQ_MODEL`. Cualquier
`GroqError` hace que `agent.answer` caiga al modo determinista.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1500


class GroqError(Exception):
    """Fallo de red, HTTP o formato al hablar con Groq."""


def groq_model() -> str:
    return os.environ.get("SAVI_GROQ_MODEL", DEFAULT_GROQ_MODEL)


def make_post(api_key: str, timeout: float = 60.0) -> Callable[[dict], dict]:
    """Devuelve una función `post(payload) -> respuesta JSON` contra Groq."""

    def post(payload: dict) -> dict:
        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                     "User-Agent": "savi-agent/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise GroqError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise GroqError(f"{type(exc).__name__}: {exc}") from exc

    return post


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convierte las herramientas (formato Anthropic) al formato `function` de OpenAI."""
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}}
        for t in tools
    ]


def _first_message(response: dict) -> tuple[dict, Optional[str]]:
    try:
        choice = response["choices"][0]
        return choice["message"], choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqError(f"respuesta sin choices: {str(response)[:200]}") from exc


def _add_usage(total: dict, response: dict) -> None:
    usage = response.get("usage") or {}
    for src, dst in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        if isinstance(usage.get(src), (int, float)):
            total[dst] = total.get(dst, 0) + usage[src]


def run(
    post: Callable[[dict], dict],
    tools_mod: Any,
    message: str,
    history: Optional[list[dict]],
    *,
    system_prompt: str,
    model: str,
    max_turns: int,
    run_tool: Optional[Callable[[Any, str, dict], dict]] = None,
    summarize: Optional[Callable[[str, dict], str]] = None,
    collect_evidence: Optional[Callable[[list[dict]], dict]] = None,
) -> dict:
    """Bucle de tool use. `post` se inyecta para poder probarlo sin red."""
    run_tool = run_tool or (lambda mod, name, args: mod.run_tool(name, args))
    summarize = summarize or (lambda name, result: name)
    collect_evidence = collect_evidence or (lambda results: {})

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in history or []:
        if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    tools_schema = to_openai_tools(tools_mod.TOOLS)
    tool_calls: list[dict] = []
    evidence_results: list[dict] = []
    usage: dict = {}
    base = {"model": model, "max_tokens": MAX_TOKENS, "temperature": 0.2}

    turns = 0
    while True:
        response = post({**base, "messages": messages, "tools": tools_schema, "tool_choice": "auto"})
        _add_usage(usage, response)
        msg, finish = _first_message(response)
        calls = msg.get("tool_calls") or []
        if finish != "tool_calls" and not calls:
            break

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call.get("function", {}).get("name", "")
            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except ValueError:
                args = {}
            output = run_tool(tools_mod, name, args)
            evidence_results.append(output)
            tool_calls.append({"name": name, "input": args, "summary": summarize(name, output)})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(output, ensure_ascii=False)})

        turns += 1
        if turns >= max_turns:
            response = post({**base, "messages": messages, "tools": tools_schema, "tool_choice": "none"})
            _add_usage(usage, response)
            msg, _ = _first_message(response)
            break

    answer_text = (msg.get("content") or "").strip()
    if not answer_text:
        raise GroqError("el modelo no devolvió texto final")
    return {
        "answer": answer_text,
        "mode": "groq",
        "tool_calls": tool_calls,
        "evidence": collect_evidence(evidence_results),
        "model": response.get("model", model),
        "usage": usage,
    }
