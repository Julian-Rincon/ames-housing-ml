# -*- coding: utf-8 -*-
"""Modo Groq del agente: bucle de tool use sobre la API compatible con OpenAI.

Nunca golpea la red: se inyecta un `post` falso que devuelve respuestas guionadas.
"""
from __future__ import annotations

import json
import types

from savi_api import agent, groq_mode


def _fake_tools():
    calls = []

    def run_tool(name, tool_input):
        calls.append((name, tool_input))
        return {"pid": tool_input.get("pid"), "avm_value": 185000.0, "decision": "APROBAR"}

    mod = types.SimpleNamespace(
        TOOLS=[{
            "name": "get_parcel",
            "description": "datos de una parcela",
            "input_schema": {"type": "object", "properties": {"pid": {"type": "string"}}, "required": ["pid"]},
        }],
        run_tool=run_tool,
    )
    return mod, calls


def _scripted_post(responses):
    sent = []

    def post(payload):
        sent.append(payload)
        return responses.pop(0)

    return post, sent


def test_tools_se_convierten_a_formato_openai():
    mod, _ = _fake_tools()
    out = groq_mode.to_openai_tools(mod.TOOLS)
    assert out == [{
        "type": "function",
        "function": {
            "name": "get_parcel",
            "description": "datos de una parcela",
            "parameters": mod.TOOLS[0]["input_schema"],
        },
    }]


def test_bucle_ejecuta_la_herramienta_y_devuelve_respuesta_final():
    mod, calls = _fake_tools()
    post, sent = _scripted_post([
        {"model": "m", "choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "get_parcel", "arguments": json.dumps({"pid": "0526301100"})}}]}}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        {"model": "m", "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "La parcela vale US$185.000 y la decisión es APROBAR."}}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 9}},
    ])
    out = groq_mode.run(post, mod, "¿cuánto vale 0526301100?", None, system_prompt="sys", model="m", max_turns=4)
    assert out["mode"] == "groq"
    assert "185.000" in out["answer"]
    assert calls == [("get_parcel", {"pid": "0526301100"})]
    assert out["tool_calls"][0]["name"] == "get_parcel"
    assert out["usage"] == {"input_tokens": 30, "output_tokens": 14}
    # el resultado de la herramienta vuelve al modelo como mensaje role=tool
    assert sent[1]["messages"][-1]["role"] == "tool"
    assert sent[1]["messages"][-1]["tool_call_id"] == "c1"


def test_argumentos_invalidos_no_rompen_el_bucle():
    mod, calls = _fake_tools()
    post, _ = _scripted_post([
        {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_parcel", "arguments": "{no json"}}]}}]},
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]},
    ])
    out = groq_mode.run(post, mod, "x", None, system_prompt="s", model="m", max_turns=4)
    assert calls == [("get_parcel", {})]
    assert out["answer"] == "ok"


def test_tope_de_turnos_fuerza_respuesta_sin_herramientas():
    mod, _ = _fake_tools()
    tool_turn = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_parcel", "arguments": "{}"}}]}}]}
    final = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "final"}}]}
    post, sent = _scripted_post([tool_turn, final])
    out = groq_mode.run(post, mod, "x", None, system_prompt="s", model="m", max_turns=1)
    assert out["answer"] == "final"
    assert sent[-1]["tool_choice"] == "none"


def test_answer_usa_groq_si_hay_clave_y_no_hay_anthropic(monkeypatch):
    mod, _ = _fake_tools()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(agent, "_build_anthropic_client", lambda: None)
    post, _ = _scripted_post([
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hola desde groq"}}]},
    ])
    monkeypatch.setattr(groq_mode, "make_post", lambda key, timeout=60.0: post)
    out = agent.answer("metodología", tools_module=mod)
    assert out["mode"] == "groq"
    assert out["answer"] == "hola desde groq"


def test_answer_cae_a_determinista_si_groq_falla(monkeypatch):
    mod, _ = _fake_tools()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(agent, "_build_anthropic_client", lambda: None)

    def boom(payload):
        raise groq_mode.GroqError("HTTP 429: rate limit")

    monkeypatch.setattr(groq_mode, "make_post", lambda key, timeout=60.0: boom)
    monkeypatch.setattr(agent, "_run_deterministic_mode",
                        lambda tm, m, h, fallback_reason=None: {"mode": "deterministic", "reason": fallback_reason})
    out = agent.answer("metodología", tools_module=mod)
    assert out["mode"] == "deterministic"
    assert "429" in out["reason"]


def test_sin_ninguna_clave_sigue_en_determinista(monkeypatch):
    mod, _ = _fake_tools()
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(agent, "_build_anthropic_client", lambda: None)
    monkeypatch.setattr(agent, "_run_deterministic_mode",
                        lambda tm, m, h, fallback_reason=None: {"mode": "deterministic"})
    assert agent.answer("x", tools_module=mod)["mode"] == "deterministic"
