# -*- coding: utf-8 -*-
"""
SAVI Agent API — Lambda Function URL handler (payload format 2.0).

Ver RAG_CONTRACT.md -> "API HTTP (handler.py, Function URL payload 2.0)".
Enrutamiento manual sin framework (sin dependencias extra permitidas salvo
`anthropic`, que ya usa `savi_api.agent`). CORS abierto en toda respuesta.

No se toca `savi_api.{store,inference,retrieval,tools}` (otro agente los escribe
en paralelo): se accede siempre vía `importlib.import_module`, nunca con un
import directo a nivel de módulo, para no romper la carga de este archivo si
esos módulos todavía no existen en disco.
"""
from __future__ import annotations

import base64
import importlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote_plus

logger = logging.getLogger("savi.handler")
logger.setLevel(logging.INFO)

HANDLER_DIR = Path(__file__).resolve().parent
STATIC_DIR = HANDLER_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Max-Age": "86400",
}

_MAX_BODY_BYTES = 16 * 1024
_MAX_MESSAGE_CHARS = 2000
_MAX_HISTORY_TURNS = 10
_PID_RE = re.compile(r"^\d{10}$")


class ApiError(Exception):
    """Error de validación/negocio con código HTTP asociado."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- #
# Acceso perezoso a savi_api.{tools,agent,store}
# --------------------------------------------------------------------------- #

def _tools_module():
    return importlib.import_module("savi_api.tools")


def _agent_module():
    return importlib.import_module("savi_api.agent")


def _run_tool(name: str, tool_input: dict) -> dict:
    tools_mod = _tools_module()
    result = tools_mod.run_tool(name, tool_input)
    if not isinstance(result, dict):
        return {"error": "resultado de herramienta inválido"}
    return result


# Precalentar el store al importar el módulo (arranque en frío de la Lambda).
# Si `store.py` todavía no existe (escrito por otro agente en paralelo) o falla
# por cualquier motivo, no debe tumbar el arranque de la Lambda.
try:
    _store_module = importlib.import_module("savi_api.store")
    _store_module.get_store()
    logger.info("Store precalentado correctamente")
except Exception as exc:  # noqa: BLE001
    logger.warning("No se pudo precalentar el store al importar: %s", exc)


# --------------------------------------------------------------------------- #
# Helpers de request/response (payload Function URL v2.0)
# --------------------------------------------------------------------------- #

def _get_body_text(event: dict) -> str:
    body = event.get("body")
    if body is None:
        return ""
    if event.get("isBase64Encoded"):
        return base64.b64decode(body).decode("utf-8", errors="replace")
    return body


def _json_response(status: int, payload: Any) -> dict:
    headers = dict(CORS_HEADERS)
    headers["Content-Type"] = "application/json"
    return {
        "statusCode": status,
        "headers": headers,
        "body": json.dumps(payload, ensure_ascii=False),
        "isBase64Encoded": False,
    }


def _html_response(status: int, html: str) -> dict:
    headers = dict(CORS_HEADERS)
    headers["Content-Type"] = "text/html; charset=utf-8"
    return {"statusCode": status, "headers": headers, "body": html, "isBase64Encoded": False}


def _error_response(status: int, message: str) -> dict:
    return _json_response(status, {"error": message})


def _parse_float(value: Optional[str], name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise ApiError(400, f"parámetro numérico inválido: {name}") from None


def _parse_int(value: Optional[str], name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise ApiError(400, f"parámetro numérico inválido: {name}") from None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    return value.strip().lower() in ("1", "true", "yes", "si", "sí")


def _validate_pid(raw_pid: str) -> str:
    pid = unquote_plus(raw_pid).replace("-", "")
    if not _PID_RE.match(pid):
        raise ApiError(400, "pid inválido: se esperan 10 dígitos (guiones opcionales)")
    return pid


def _parse_json_body(event: dict) -> dict:
    raw = event.get("body") or ""
    raw_bytes = raw.encode("utf-8") if not event.get("isBase64Encoded") else base64.b64decode(raw)
    if len(raw_bytes) > _MAX_BODY_BYTES:
        raise ApiError(400, f"cuerpo demasiado grande (máx {_MAX_BODY_BYTES} bytes)")
    text = _get_body_text(event)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ApiError(400, "cuerpo JSON inválido") from None
    if not isinstance(data, dict):
        raise ApiError(400, "cuerpo JSON debe ser un objeto")
    return data


# --------------------------------------------------------------------------- #
# Handlers de ruta
# --------------------------------------------------------------------------- #

def _route_static_index(event: dict, params: dict) -> dict:
    if not INDEX_HTML.is_file():
        return _error_response(404, "static/index.html no encontrado")
    return _html_response(200, INDEX_HTML.read_text(encoding="utf-8"))


def _route_health(event: dict, params: dict) -> dict:
    result = _run_tool("model_card", {})
    agent_mod = _agent_module()
    has_key = bool(agent_mod._resolve_api_key())  # noqa: SLF001 - uso interno intencional
    payload = {
        "status": "ok" if "error" not in result else "degraded",
        "llm_mode": "claude" if has_key else "deterministic",
    }
    for key in ("run_id", "hpi_latest", "counts"):
        if key in result:
            payload[key] = result[key]
    return _json_response(200, payload)


def _route_parcel(event: dict, params: dict) -> dict:
    pid = _validate_pid(params["pid"])
    result = _run_tool("get_parcel", {"pid": pid})
    status = 404 if "error" in result else 200
    return _json_response(status, result)


def _route_search(event: dict, params: dict) -> dict:
    qs = event.get("queryStringParameters") or {}
    tool_input: dict = {}
    for key in ("neighborhood", "subdivision", "style_contains", "decision", "sort_by"):
        if qs.get(key):
            tool_input[key] = qs[key]
    for key in ("min_value", "max_value", "min_area", "max_area"):
        value = _parse_float(qs.get(key), key)
        if value is not None:
            tool_input[key] = value
    for key in ("min_year", "max_year", "beds_min", "limit"):
        value = _parse_int(qs.get(key), key)
        if value is not None:
            tool_input[key] = value
    descending = _parse_bool(qs.get("descending"))
    if descending is not None:
        tool_input["descending"] = descending
    result = _run_tool("search_parcels", tool_input)
    return _json_response(200, result)


def _route_comps(event: dict, params: dict) -> dict:
    pid = _validate_pid(params["pid"])
    qs = event.get("queryStringParameters") or {}
    tool_input: dict = {"pid": pid}
    k = _parse_int(qs.get("k"), "k")
    if k is not None:
        tool_input["k"] = k
    result = _run_tool("find_comparables", tool_input)
    return _json_response(200, result)


def _route_valuate(event: dict, params: dict) -> dict:
    body = _parse_json_body(event)
    result = _run_tool("value_property", body)
    status = 400 if "error" in result else 200
    return _json_response(status, result)


def _route_areas(event: dict, params: dict) -> dict:
    # No hay una herramienta "listar todas las zonas" en savi_api.tools (area_stats
    # exige neighborhood/subdivision/cluster) — se expone directamente el `areas.json`
    # cargado por el store (neighborhoods[]/subdivisions[]/clusters[]), tal como lo
    # necesita la UI para el mapa/selector.
    store_mod = importlib.import_module("savi_api.store")
    store = store_mod.get_store()
    return _json_response(200, store.areas)


def _route_market(event: dict, params: dict) -> dict:
    qs = event.get("queryStringParameters") or {}
    tool_input = {"since": qs["since"]} if qs.get("since") else {}
    result = _run_tool("market_trend", tool_input)
    return _json_response(200, result)


def _route_model(event: dict, params: dict) -> dict:
    result = _run_tool("model_card", {})
    return _json_response(200, result)


def _route_chat(event: dict, params: dict) -> dict:
    body = _parse_json_body(event)
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ApiError(400, "'message' es requerido y debe ser texto no vacío")
    if len(message) > _MAX_MESSAGE_CHARS:
        raise ApiError(400, f"'message' excede el máximo de {_MAX_MESSAGE_CHARS} caracteres")

    history = body.get("history") or []
    if not isinstance(history, list):
        raise ApiError(400, "'history' debe ser una lista")
    if len(history) > _MAX_HISTORY_TURNS:
        raise ApiError(400, f"'history' excede el máximo de {_MAX_HISTORY_TURNS} turnos")
    for turn in history:
        if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
            raise ApiError(400, "cada turno de 'history' debe tener 'role' y 'content'")

    agent_mod = _agent_module()
    result = agent_mod.answer(message, history)
    return _json_response(200, result)


# --------------------------------------------------------------------------- #
# Tabla de ruteo
# --------------------------------------------------------------------------- #

_ROUTES: list[tuple[str, re.Pattern, Any]] = [
    ("GET", re.compile(r"^/$"), _route_static_index),
    ("GET", re.compile(r"^/api/health/?$"), _route_health),
    ("GET", re.compile(r"^/api/parcel/(?P<pid>[^/]+)/?$"), _route_parcel),
    ("GET", re.compile(r"^/api/search/?$"), _route_search),
    ("GET", re.compile(r"^/api/comps/(?P<pid>[^/]+)/?$"), _route_comps),
    ("POST", re.compile(r"^/api/valuate/?$"), _route_valuate),
    ("GET", re.compile(r"^/api/areas/?$"), _route_areas),
    ("GET", re.compile(r"^/api/market/?$"), _route_market),
    ("GET", re.compile(r"^/api/model/?$"), _route_model),
    ("POST", re.compile(r"^/api/chat/?$"), _route_chat),
]


def _dispatch(method: str, path: str, event: dict) -> tuple[dict, str]:
    """Devuelve (respuesta, nombre_de_ruta_para_logs)."""
    if method == "OPTIONS":
        headers = dict(CORS_HEADERS)
        return {"statusCode": 204, "headers": headers, "body": "", "isBase64Encoded": False}, "OPTIONS"

    for route_method, pattern, fn in _ROUTES:
        if route_method != method:
            continue
        match = pattern.match(path)
        if match:
            route_name = f"{method} {pattern.pattern}"
            try:
                return fn(event, match.groupdict()), route_name
            except ApiError as exc:
                return _error_response(exc.status, exc.message), route_name
            except Exception as exc:  # noqa: BLE001 - nunca debe tumbar la Lambda
                logger.exception("Error no controlado en %s", route_name)
                return _error_response(500, f"error interno: {exc}"), route_name

    return _error_response(404, "ruta no encontrada"), f"{method} {path}"


# --------------------------------------------------------------------------- #
# Entrypoint Lambda
# --------------------------------------------------------------------------- #

def handler(event: dict, context: Any = None) -> dict:
    start = time.monotonic()
    try:
        http = event.get("requestContext", {}).get("http", {})
        method = http.get("method", "GET")
        raw_path = event.get("rawPath", "/")
    except AttributeError:
        return _error_response(400, "evento Function URL inválido")

    response, route_name = _dispatch(method, raw_path, event)

    mode = None
    if isinstance(response.get("body"), str) and route_name.startswith("POST /api/chat"):
        try:
            mode = json.loads(response["body"]).get("mode")
        except (json.JSONDecodeError, AttributeError):
            mode = None

    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
    logger.info(
        json.dumps(
            {
                "route": route_name,
                "status": response.get("statusCode"),
                "ms": elapsed_ms,
                "mode": mode,
            },
            ensure_ascii=False,
        )
    )
    return response
