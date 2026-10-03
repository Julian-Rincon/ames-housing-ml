# -*- coding: utf-8 -*-
"""
Tests de `handler.py` (ruteo, CORS, validación, 404) y `savi_api/agent.py`
(modo determinista y modo Claude con un cliente/anthropic falso).

Nunca se llama a AWS ni a la API real de Anthropic: `savi_api.tools` se inyecta
como módulo falso vía `sys.modules` (handler y agent acceden siempre con
`importlib.import_module("savi_api.tools")`, nunca con un import directo a
nivel de módulo, justamente para permitir esto sin tocar el filesystem), y el
modo Claude se prueba inyectando un cliente falso directamente en `agent.answer`.
"""
from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path

import pytest

import handler  # noqa: E402
from savi_api import agent  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes compartidos
# --------------------------------------------------------------------------- #

def _fake_tools_module(run_tool_fn=None, tools_schema=None):
    mod = types.ModuleType("savi_api.tools")
    mod.TOOLS = tools_schema if tools_schema is not None else [
        {"name": "get_parcel", "description": "obtiene una parcela", "input_schema": {"type": "object"}}
    ]

    def default_run_tool(name, tool_input):
        if name == "get_parcel":
            return {"pid": tool_input.get("pid"), "avm_value_today": 185000.0, "final": "APROBAR"}
        if name == "model_card":
            return {"run_id": "test-run", "hpi_latest": "2026Q2", "counts": {"parcels": 1}}
        return {"error": f"herramienta desconocida: {name}"}

    mod.run_tool = run_tool_fn or default_run_tool
    return mod


@pytest.fixture()
def install_fake_tools(monkeypatch):
    def _install(run_tool_fn=None, tools_schema=None):
        fake = _fake_tools_module(run_tool_fn, tools_schema)
        monkeypatch.setitem(sys.modules, "savi_api.tools", fake)
        return fake

    return _install


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch):
    """Por defecto, nunca hay clave disponible -> siempre modo determinista salvo
    que un test inyecte explícitamente un cliente falso."""
    monkeypatch.setattr(agent, "_resolve_api_key", lambda: None)


# --------------------------------------------------------------------------- #
# handler.py: ruteo básico, CORS, 404, validación
# --------------------------------------------------------------------------- #

def _event(method: str, path: str, *, query=None, body=None, headers=None) -> dict:
    return {
        "requestContext": {"http": {"method": method}},
        "rawPath": path,
        "queryStringParameters": query,
        "headers": headers or {},
        "body": body,
        "isBase64Encoded": False,
    }


def test_cors_headers_present_on_normal_route(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("GET", "/api/model"))
    assert resp["headers"]["Access-Control-Allow-Origin"] == "*"
    assert "GET" in resp["headers"]["Access-Control-Allow-Methods"]


def test_options_preflight(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("OPTIONS", "/api/chat"))
    assert resp["statusCode"] == 204
    assert resp["headers"]["Access-Control-Allow-Origin"] == "*"


def test_unknown_route_404(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("GET", "/api/no-existe"))
    assert resp["statusCode"] == 404
    assert "error" in json.loads(resp["body"])


def test_get_root_serves_static_index(install_fake_tools, monkeypatch, tmp_path):
    install_fake_tools()
    fake_index = tmp_path / "index.html"
    fake_index.write_text("<html>SAVI</html>", encoding="utf-8")
    monkeypatch.setattr(handler, "INDEX_HTML", fake_index)

    resp = handler.handler(_event("GET", "/"))
    assert resp["statusCode"] == 200
    assert resp["headers"]["Content-Type"].startswith("text/html")
    assert "SAVI" in resp["body"]


def test_health_route(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("GET", "/api/health"))
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200
    assert body["llm_mode"] == "deterministic"
    assert body["run_id"] == "test-run"


def test_parcel_route_valid_pid(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("GET", "/api/parcel/1234567890"))
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200
    assert body["pid"] == "1234567890"


def test_parcel_route_strips_dashes(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("GET", "/api/parcel/123-456-78-90"))
    assert resp["statusCode"] == 200


def test_parcel_route_invalid_pid_400(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("GET", "/api/parcel/abc"))
    assert resp["statusCode"] == 400
    assert "pid" in json.loads(resp["body"])["error"]


def test_parcel_route_not_found_404(install_fake_tools):
    install_fake_tools(run_tool_fn=lambda name, inp: {"error": "no existe"})
    resp = handler.handler(_event("GET", "/api/parcel/1234567890"))
    assert resp["statusCode"] == 404


def test_search_route_rejects_invalid_numeric_param(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("GET", "/api/search", query={"min_value": "no-es-numero"}))
    assert resp["statusCode"] == 400


def test_search_route_parses_numeric_params(install_fake_tools):
    captured = {}

    def run_tool_fn(name, tool_input):
        captured["input"] = tool_input
        return {"total_matches": 0, "results": []}

    install_fake_tools(run_tool_fn=run_tool_fn)
    resp = handler.handler(
        _event("GET", "/api/search", query={"min_value": "100000", "beds_min": "3", "descending": "true"})
    )
    assert resp["statusCode"] == 200
    assert captured["input"]["min_value"] == 100000.0
    assert captured["input"]["beds_min"] == 3
    assert captured["input"]["descending"] is True


def test_comps_route(install_fake_tools):
    install_fake_tools(run_tool_fn=lambda name, inp: {"comparables": [{"pid": "1", "price_today": 1.0}]})
    resp = handler.handler(_event("GET", "/api/comps/1234567890", query={"k": "3"}))
    assert resp["statusCode"] == 200


def test_valuate_route_post_body(install_fake_tools):
    install_fake_tools(run_tool_fn=lambda name, inp: {"avm_value_today": 210000.0})
    body = json.dumps({"living_area": 1500, "year_built": 2005})
    resp = handler.handler(_event("POST", "/api/valuate", body=body))
    assert resp["statusCode"] == 200


def test_valuate_route_invalid_json_400(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("POST", "/api/valuate", body="{not json"))
    assert resp["statusCode"] == 400


def test_body_too_large_400(install_fake_tools):
    install_fake_tools()
    huge = json.dumps({"message": "x" * (17 * 1024)})
    resp = handler.handler(_event("POST", "/api/chat", body=huge))
    assert resp["statusCode"] == 400


def test_body_base64_decoded(install_fake_tools):
    install_fake_tools(run_tool_fn=lambda name, inp: {"avm_value_today": 1})
    raw = json.dumps({"living_area": 1200})
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    event = _event("POST", "/api/valuate", body=encoded)
    event["isBase64Encoded"] = True
    resp = handler.handler(event)
    assert resp["statusCode"] == 200


def test_chat_message_too_long_400(install_fake_tools):
    install_fake_tools()
    body = json.dumps({"message": "x" * 2001})
    resp = handler.handler(_event("POST", "/api/chat", body=body))
    assert resp["statusCode"] == 400


def test_chat_missing_message_400(install_fake_tools):
    install_fake_tools()
    resp = handler.handler(_event("POST", "/api/chat", body=json.dumps({})))
    assert resp["statusCode"] == 400


def test_chat_history_too_long_400(install_fake_tools):
    install_fake_tools()
    history = [{"role": "user", "content": "hola"} for _ in range(11)]
    body = json.dumps({"message": "hola", "history": history})
    resp = handler.handler(_event("POST", "/api/chat", body=body))
    assert resp["statusCode"] == 400


def test_chat_history_bad_shape_400(install_fake_tools):
    install_fake_tools()
    body = json.dumps({"message": "hola", "history": [{"role": "user"}]})
    resp = handler.handler(_event("POST", "/api/chat", body=body))
    assert resp["statusCode"] == 400


def test_chat_deterministic_mode_end_to_end(install_fake_tools):
    install_fake_tools()
    body = json.dumps({"message": "¿Cuál es el valor de la parcela 1234567890?"})
    resp = handler.handler(_event("POST", "/api/chat", body=body))
    payload = json.loads(resp["body"])
    assert resp["statusCode"] == 200
    assert payload["mode"] == "deterministic"
    assert payload["tool_calls"][0]["name"] == "get_parcel"
    assert "1234567890" in payload["answer"]


# --------------------------------------------------------------------------- #
# savi_api/agent.py: modo determinista (unitario, sin pasar por el handler)
# --------------------------------------------------------------------------- #

def test_deterministic_mode_detects_pid(install_fake_tools):
    fake = install_fake_tools()
    result = agent.answer("dame el valor de la parcela 1234567890", tools_module=fake)
    assert result["mode"] == "deterministic"
    assert result["tool_calls"][0] == {
        "name": "get_parcel",
        "input": {"pid": "1234567890"},
        "summary": "parcela 1234567890, AVM $185,000, decisión=APROBAR",
    }
    assert "185,000" in result["answer"] or "185000" in result["answer"]


def test_deterministic_mode_no_pid_falls_back_to_knowledge(install_fake_tools):
    fake = install_fake_tools(run_tool_fn=lambda name, inp: {"results": [{"text": "doc de prueba"}]})
    result = agent.answer("¿qué es SAVI?", tools_module=fake)
    assert result["mode"] == "deterministic"
    assert result["tool_calls"][0]["name"] == "search_knowledge"


def test_deterministic_fallback_reason_included(install_fake_tools):
    fake = install_fake_tools()
    result = agent._run_deterministic_mode(fake, "parcela 1234567890", fallback_reason="RateLimitError: boom")
    assert "RateLimitError" in result["answer"]


# --------------------------------------------------------------------------- #
# savi_api/agent.py: modo Claude con un cliente anthropic falso
# --------------------------------------------------------------------------- #

class _FakeBlock:
    def __init__(self, type_, **kwargs):
        self.type = type_
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeUsage:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeStopDetails:
    def __init__(self, category=None, explanation=None):
        self.category = category
        self.explanation = explanation


class _FakeResponse:
    def __init__(self, stop_reason, content, model="claude-opus-5", usage=None, stop_details=None):
        self.stop_reason = stop_reason
        self.content = content
        self.model = model
        self.usage = usage or _FakeUsage(input_tokens=10, output_tokens=5)
        self.stop_details = stop_details


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("se pidieron más respuestas de las configuradas en el fake")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_claude_mode_tool_use_then_end_turn(install_fake_tools):
    fake_tools = install_fake_tools(
        run_tool_fn=lambda name, inp: {"pid": inp["pid"], "avm_value_today": 200000.0}
    )
    tool_use_block = _FakeBlock("tool_use", id="call_1", name="get_parcel", input={"pid": "1234567890"})
    responses = [
        _FakeResponse("tool_use", [tool_use_block]),
        _FakeResponse("end_turn", [_FakeBlock("text", text="Respuesta final en español.")]),
    ]
    client = _FakeClient(responses)

    result = agent.answer("valor de 1234567890", client=client, tools_module=fake_tools)

    assert result["mode"] == "claude"
    assert result["answer"] == "Respuesta final en español."
    assert len(client.messages.calls) == 2
    assert result["tool_calls"] == [
        {"name": "get_parcel", "input": {"pid": "1234567890"}, "summary": "parcela 1234567890, AVM $200,000"}
    ]

    second_call_messages = client.messages.calls[1]["messages"]
    assistant_turn = second_call_messages[-2]
    user_turn = second_call_messages[-1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"] == [tool_use_block]
    assert user_turn["role"] == "user"
    assert len(user_turn["content"]) == 1
    assert user_turn["content"][0]["type"] == "tool_result"
    assert user_turn["content"][0]["tool_use_id"] == "call_1"
    assert json.loads(user_turn["content"][0]["content"]) == {"pid": "1234567890", "avm_value_today": 200000.0}


def test_claude_mode_refusal_falls_back_to_deterministic(install_fake_tools):
    fake_tools = install_fake_tools()
    responses = [_FakeResponse("refusal", [], stop_details=_FakeStopDetails(category="cyber", explanation="no"))]
    client = _FakeClient(responses)

    result = agent.answer("valor de 1234567890", client=client, tools_module=fake_tools)

    assert result["mode"] == "deterministic"
    assert "rechaz" in result["answer"].lower()


def test_claude_mode_turn_cap(install_fake_tools, monkeypatch):
    monkeypatch.setenv("SAVI_MAX_AGENT_TURNS", "2")
    fake_tools = install_fake_tools()
    tool_use_block = _FakeBlock("tool_use", id="call_x", name="get_parcel", input={"pid": "1234567890"})
    responses = [
        _FakeResponse("tool_use", [tool_use_block]),
        _FakeResponse("tool_use", [tool_use_block]),
        _FakeResponse("end_turn", [_FakeBlock("text", text="Mejor esfuerzo final.")]),
    ]
    client = _FakeClient(responses)

    result = agent.answer("valor de 1234567890", client=client, tools_module=fake_tools)

    assert len(client.messages.calls) == 3
    assert client.messages.calls[2]["tool_choice"] == {"type": "none"}
    assert result["answer"] == "Mejor esfuerzo final."
    assert result["mode"] == "claude"
