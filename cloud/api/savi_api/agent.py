# -*- coding: utf-8 -*-
"""
SAVI Agent API — agente (modo Claude + modo determinista).

Ver RAG_CONTRACT.md -> "agent.py". Expone `answer(message, history) -> dict`
con forma `{answer, mode, tool_calls[], evidence{}, model?, usage?}`.

Dos modos:
- **claude**: loop agéntico manual con el SDK oficial `anthropic` (sin numpy/pandas,
  la única dependencia externa permitida). Requiere una clave (env `ANTHROPIC_API_KEY`
  o SSM SecureString `SAVI_SSM_KEY_PARAM`, cacheada en memoria tras la primera lectura).
- **deterministic**: planner basado en reglas (regex + palabras clave en español) que
  llama 1-3 herramientas (`savi_api.tools.run_tool`) y arma la respuesta con plantillas,
  usando SOLO los números devueltos por las herramientas. Se usa cuando no hay clave o
  cuando el modo Claude falla (rate limit, error de API, conexión, rechazo del modelo).

No se toca `savi_api.{store,inference,retrieval,tools}` (los escribe otro agente en
paralelo) — se accede siempre vía `importlib.import_module("savi_api.tools")`, nunca con
un import directo a nivel de módulo, para no romper la importación de este archivo si
`tools.py` todavía no existe en disco, y para que los tests puedan inyectar un módulo
falso sin tocar el sistema de archivos.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("savi.agent")

# --------------------------------------------------------------------------- #
# Configuración (env vars, ver RAG_CONTRACT.md)
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_TOKENS = 4000
DEFAULT_SSM_PARAM = "/savi/anthropic_api_key"


def _model() -> str:
    return os.environ.get("SAVI_LLM_MODEL", DEFAULT_MODEL)


def _effort() -> str:
    return os.environ.get("SAVI_LLM_EFFORT", DEFAULT_EFFORT)


def _max_turns() -> int:
    try:
        return max(1, int(os.environ.get("SAVI_MAX_AGENT_TURNS", DEFAULT_MAX_TURNS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TURNS


def _ssm_param_name() -> str:
    return os.environ.get("SAVI_SSM_KEY_PARAM", DEFAULT_SSM_PARAM)


SYSTEM_PROMPT = """\
Eres SAVI, un agente de valuación de propiedades residenciales en Ames, Iowa (EE. UU.).

Reglas estrictas:
- SIEMPRE fundamenta cada cifra en el resultado de una herramienta (RAG). Nunca inventes
  datos, precios, direcciones o estadísticas.
- Cita PIDs de parcela y/o PIDs de venta y la fuente de cada número que uses.
- Cuando el usuario pregunte por la decisión del agente de RL, explica si es
  APROBAR, REVISAR o RECHAZAR y qué significa económicamente esa decisión
  (margen de valuación, consenso de reglas, política seleccionada).
- Declara la incertidumbre del modelo AVM (MAPE aproximado ~7.8%) y el rango/CI cuando
  corresponda.
- Admite explícitamente cuando falte información (no hay direcciones postales; los
  precios están expresados en dólares de hoy vía el índice FHFA HPI, no en dólares
  nominales de la venta).
- Si una herramienta no devuelve el dato pedido, dilo en vez de adivinar.
- Responde siempre en español, en markdown conciso (listas y negritas cuando ayuden).
"""

# Excepciones del SDK que hacen caer el modo Claude al modo determinista.
# Orden más-específico-primero: RateLimitError es subclase de APIStatusError.
_CLAUDE_FALLBACK_EXCEPTIONS: tuple[type[Exception], ...] = ()
try:
    import anthropic  # type: ignore

    _CLAUDE_FALLBACK_EXCEPTIONS = (
        anthropic.RateLimitError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
    )
except ImportError:  # pragma: no cover - anthropic no instalado en el entorno
    anthropic = None  # type: ignore[assignment]


class ClaudeRefusalError(Exception):
    """El modelo rechazó la solicitud (`stop_reason == "refusal"`)."""


# --------------------------------------------------------------------------- #
# Resolución de credenciales (cacheada en memoria por proceso)
# --------------------------------------------------------------------------- #

_cached_api_key: Optional[str] = None
_api_key_resolved = False


def _resolve_api_key() -> Optional[str]:
    """
    ANTHROPIC_API_KEY (env) -> SSM SecureString (SAVI_SSM_KEY_PARAM) -> None.
    El resultado se cachea en memoria para no repetir la llamada a SSM en cada
    invocación (la Lambda reutiliza el proceso entre invocaciones calientes).
    """
    global _cached_api_key, _api_key_resolved
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    if _api_key_resolved:
        return _cached_api_key

    _api_key_resolved = True
    try:
        import boto3  # import perezoso: no hace falta en tests que inyectan la clave

        ssm = boto3.client("ssm")
        param = ssm.get_parameter(Name=_ssm_param_name(), WithDecryption=True)
        _cached_api_key = param["Parameter"]["Value"]
        logger.info("Clave de Anthropic obtenida de SSM (%s)", _ssm_param_name())
    except Exception as exc:  # noqa: BLE001 - cualquier fallo -> modo determinista
        logger.warning("No se pudo obtener la clave de Anthropic desde SSM: %s", exc)
        _cached_api_key = None
    return _cached_api_key


def _build_anthropic_client():
    if anthropic is None:
        return None
    key = _resolve_api_key()
    if not key:
        return None
    try:
        return anthropic.Anthropic(api_key=key, timeout=150.0, max_retries=1)  # < timeout de la Lambda (180 s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo construir el cliente de Anthropic: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Acceso a savi_api.tools sin import directo a nivel de módulo
# --------------------------------------------------------------------------- #

def _tools_module():
    return importlib.import_module("savi_api.tools")


def _run_tool(tools_mod, name: str, tool_input: dict) -> dict:
    try:
        result = tools_mod.run_tool(name, tool_input)
    except Exception as exc:  # noqa: BLE001 - run_tool no debería lanzar, pero por las dudas
        logger.warning("run_tool('%s', %s) lanzó una excepción: %s", name, tool_input, exc)
        return {"error": str(exc)}
    if not isinstance(result, dict):
        return {"error": "resultado de herramienta inválido"}
    return result


def _summarize_tool_result(name: str, result: dict) -> str:
    """Resumen corto y legible del resultado de una herramienta para `tool_calls[].summary`."""
    if "error" in result:
        return f"error: {result['error']}"
    if name == "get_parcel":
        pid = result.get("pid")
        value = result.get("avm_value_today") or result.get("assessed_today")
        decision = result.get("final") or result.get("decision")
        bits = [f"parcela {pid}" if pid else "parcela"]
        if value is not None:
            bits.append(f"AVM ${value:,.0f}" if isinstance(value, (int, float)) else f"AVM {value}")
        if decision:
            bits.append(f"decisión={decision}")
        return ", ".join(bits)
    if name == "search_parcels":
        total = result.get("total_matches", len(result.get("results", []) or []))
        return f"{total} coincidencias"
    if name == "find_comparables":
        comps = result.get("comparables") or result.get("results") or []
        return f"{len(comps)} comparables"
    if name == "value_property":
        value = result.get("avm_value_today") or result.get("value")
        decision = result.get("final") or result.get("decision")
        bits = []
        if value is not None:
            bits.append(f"AVM ${value:,.0f}" if isinstance(value, (int, float)) else f"AVM {value}")
        if decision:
            bits.append(f"decisión={decision}")
        return ", ".join(bits) or "valuación calculada"
    if name == "area_stats":
        return "estadísticas de zona"
    if name == "market_trend":
        return "serie de mercado"
    if name == "search_knowledge":
        hits = result.get("results") or result.get("hits") or []
        return f"{len(hits)} fragmentos de documentación"
    if name == "model_card":
        return "ficha del modelo"
    return "resultado"


# --------------------------------------------------------------------------- #
# Evidencia para la UI (parcels/sales/areas con lat/lon cuando existan)
# --------------------------------------------------------------------------- #

def _collect_evidence(tool_results: list[dict]) -> dict:
    evidence: dict[str, list] = {"parcels": [], "sales": [], "areas": []}
    seen_parcels: set = set()
    seen_sales: set = set()
    seen_areas: set = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            # Venta: trae price + (sale_date o yr_sold).
            if "price" in node and ("sale_date" in node or "yr_sold" in node):
                key = node.get("pid") or id(node)
                if key not in seen_sales:
                    seen_sales.add(key)
                    evidence["sales"].append(node)
            elif "pid" in node and ("avm_value_today" in node or "assessed_today" in node or "living_area" in node):
                key = node["pid"]
                if key not in seen_parcels:
                    seen_parcels.add(key)
                    evidence["parcels"].append(node)
            elif "neighborhood" in node and ("n_sales" in node or "median_price_today" in node):
                key = node.get("neighborhood")
                if key not in seen_areas:
                    seen_areas.add(key)
                    evidence["areas"].append(node)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    for result in tool_results:
        _walk(result)
    return evidence


# --------------------------------------------------------------------------- #
# Modo Claude (loop agéntico manual)
# --------------------------------------------------------------------------- #

def _history_to_messages(history: Optional[list[dict]]) -> list[dict]:
    messages: list[dict] = []
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content is not None:
            messages.append({"role": role, "content": content})
    return messages


def _extract_text(content_blocks) -> str:
    parts = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
    return "\n".join(p for p in parts if p).strip()


def _accumulate_usage(total: dict, usage) -> None:
    if usage is None:
        return
    for field in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        value = getattr(usage, field, None)
        if isinstance(value, (int, float)):
            total[field] = total.get(field, 0) + value


def _run_claude_mode(client, tools_mod, message: str, history: Optional[list[dict]]) -> dict:
    model = _model()
    effort = _effort()
    max_turns = _max_turns()

    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})

    system_blocks = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    tools_schema = tools_mod.TOOLS

    tool_calls: list[dict] = []
    tool_results_for_evidence: list[dict] = []
    usage_total: dict = {}

    turns = 0
    response = None
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=system_blocks,
            tools=tools_schema,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            cache_control={"type": "ephemeral"},
        )
        _accumulate_usage(usage_total, getattr(response, "usage", None))

        if response.stop_reason == "refusal":
            category = None
            explanation = None
            stop_details = getattr(response, "stop_details", None)
            if stop_details is not None:
                category = getattr(stop_details, "category", None)
                explanation = getattr(stop_details, "explanation", None)
            raise ClaudeRefusalError(f"rechazo del modelo (categoria={category}): {explanation}")

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_input = block.input if isinstance(block.input, dict) else {}
                output = _run_tool(tools_mod, block.name, tool_input)
                tool_results_for_evidence.append(output)
                tool_calls.append(
                    {"name": block.name, "input": tool_input, "summary": _summarize_tool_result(block.name, output)}
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )
        messages.append({"role": "user", "content": results})

        turns += 1
        if turns >= max_turns:
            response = client.messages.create(
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=system_blocks,
                tools=tools_schema,
                tool_choice={"type": "none"},
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
            )
            _accumulate_usage(usage_total, getattr(response, "usage", None))
            break

    answer_text = _extract_text(response.content) if response is not None else ""
    if not answer_text:
        answer_text = "No tengo una respuesta final del modelo; por favor reformula la pregunta."

    return {
        "answer": answer_text,
        "mode": "claude",
        "tool_calls": tool_calls,
        "evidence": _collect_evidence(tool_results_for_evidence),
        "model": getattr(response, "model", model) if response is not None else model,
        "usage": usage_total,
    }


# --------------------------------------------------------------------------- #
# Modo determinista (planner basado en reglas, en español)
# --------------------------------------------------------------------------- #

_PID_RE = re.compile(r"\b[\d-]{10,14}\b")
_BEDS_RE = re.compile(r"(\d+)\s*(?:habitaciones?|cuartos?|dormitorios?|recamaras?)")
# Lookahead negativo: evita que "menos de 3 habitaciones"/"más de 2 baños" se
# confunda con un filtro de precio (sólo habitaciones/cuartos/baños/área tienen
# su propio regex; un precio sin unidad ($/mil/k/millón) y seguido de esas
# palabras no es un precio).
_NOT_PRICE_UNIT = r"(?!\s*(?:habitaci|cuartos?\b|dormitorio|recamara|ba[ñn]o|m2\b|sqft\b|pies\b))"
_MAX_PRICE_RE = re.compile(
    rf"(?:menos de|hasta|máximo|maximo)\s*\$?\s*([\d.,]+)\s*(mil|k|millon(?:es)?|m)?{_NOT_PRICE_UNIT}",
    re.IGNORECASE,
)
_MIN_PRICE_RE = re.compile(
    rf"(?:más de|mas de|desde|mínimo|minimo)\s*\$?\s*([\d.,]+)\s*(mil|k|millon(?:es)?|m)?{_NOT_PRICE_UNIT}",
    re.IGNORECASE,
)
_MIN_YEAR_RE = re.compile(r"(?:despu[ée]s de|desde el?)\s*(\d{4})")
_MAX_YEAR_RE = re.compile(r"(?:antes de|hasta el?)\s*(\d{4})")
_K_RE = re.compile(r"\b(\d{1,2})\s*comparables\b")


def _extract_pid(text: str) -> Optional[str]:
    for match in _PID_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) == 10:
            return digits
    return None


def _parse_price_token(value: str, unit: Optional[str]) -> Optional[float]:
    try:
        number = float(value.replace(".", "").replace(",", "")) if "," in value or "." in value else float(value)
    except ValueError:
        return None
    unit = (unit or "").lower()
    if unit in ("mil", "k"):
        number *= 1_000
    elif unit in ("millon", "millones", "m"):
        number *= 1_000_000
    elif number < 1_000:
        # Sin unidad y un número chico (p.ej. "menos de 3") casi seguro no es un
        # precio de vivienda — se descarta para no confundirlo con otro filtro.
        return None
    return number


def _extract_search_filters(text: str) -> dict:
    filters: dict = {}
    beds = _BEDS_RE.search(text)
    if beds:
        filters["beds_min"] = int(beds.group(1))
    max_price = _MAX_PRICE_RE.search(text)
    if max_price:
        value = _parse_price_token(max_price.group(1), max_price.group(2))
        if value is not None:
            filters["max_value"] = value
    min_price = _MIN_PRICE_RE.search(text)
    if min_price:
        value = _parse_price_token(min_price.group(1), min_price.group(2))
        if value is not None:
            filters["min_value"] = value
    min_year = _MIN_YEAR_RE.search(text)
    if min_year:
        filters["min_year"] = int(min_year.group(1))
    max_year = _MAX_YEAR_RE.search(text)
    if max_year:
        filters["max_year"] = int(max_year.group(1))
    return filters


def _get_store():
    """Store real (`savi_api.store.get_store()`), o `None` si no está disponible
    (p.ej. en tests que sólo inyectan un `tools_module` falso)."""
    try:
        store_mod = importlib.import_module("savi_api.store")
        return store_mod.get_store()
    except Exception as exc:  # noqa: BLE001
        logger.debug("No se pudo obtener el store para resolver nombres de zona: %s", exc)
        return None


def _extract_neighborhood(text: str) -> Optional[str]:
    """
    Primero intenta reconocer el nombre exacto de un barrio real (vía el store,
    que ya trae el alias resolver `resolve_neighborhood`), buscando cada nombre
    de `areas.neighborhoods` como substring del mensaje (case-insensitive). Si
    el store no está disponible, cae a una heurística por mayúsculas.
    """
    store = _get_store()
    if store is not None:
        lower = text.lower()
        # Coincidencia MÁS LARGA: "Northridge Heights" y "Northridge" son barrios distintos de Ames
        names = sorted((e.get("neighborhood") for e in store.areas.get("neighborhoods", []) if e.get("neighborhood")),
                       key=len, reverse=True)
        for name in names:
            if re.search(r"(?<![\w])" + re.escape(name.lower()) + r"(?![\w])", lower):
                return name

    match = re.search(
        r"(?:barrio|zona|neighborhood)\s+(?:de\s+)?([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]*(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]*)*)",
        text,
    )
    if match:
        return match.group(1).strip()
    return None


def _fmt_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:,.0f}"
    return str(value)


_ACTION_MEANING = {
    "APROBAR": "automatizar la valuación (el AVM es confiable para este perfil de predio)",
    "REVISAR": "enviar a un tasador humano (el costo de equivocarse supera el de revisar)",
    "RECHAZAR": "solicitar más información antes de valuar (perfil atípico o datos insuficientes)",
}


def _render_get_parcel(result: dict, pid: str) -> str:
    if "error" in result:
        return f"No encontré la parcela **{pid}**: {result['error']}."
    lines = [f"### Parcela {pid}"]
    value = result.get("avm_value_today")
    mape = 0.0776
    if value is not None:
        lines.append(f"- **Valor AVM (hoy)**: {_fmt_money(value)} "
                     f"(rango ±MAPE: {_fmt_money(value * (1 - mape))} – {_fmt_money(value * (1 + mape))})")
    assessed = result.get("assessed_today")
    if assessed is not None:
        gap = result.get("gap_vs_assessed")
        gap_txt = f" · brecha AVM vs avalúo: {gap * 100:+.1f}%" if gap is not None else ""
        lines.append(f"- **Avalúo oficial 2024 llevado a hoy**: {_fmt_money(assessed)}{gap_txt}")
    dec = result.get("agent_decision") or {}
    decision = dec.get("final") or result.get("final") or result.get("decision")
    if decision:
        lines.append(f"- **Decisión del agente RL**: **{decision}** → {_ACTION_MEANING.get(str(decision).upper(), '')}")
        q = dec.get("q_values") or {}
        if q:
            q_txt = ", ".join(f"{a} {v:,.0f}" for a, v in q.items())
            lines.append(f"  - Q-values DQN: {q_txt} (margen {dec.get('margin', 0):,.0f}); "
                         f"regla seleccionada en validación: `{dec.get('rule', '?')}`; "
                         f"DQN={dec.get('dqn')}, consenso VI/QL/DQN={dec.get('consensus')}")
    facts = []
    for key, label in [("neighborhood", "Barrio"), ("subdivision", "Subdivisión"), ("style", "Estilo"),
                       ("year_built", "Año"), ("living_area", "Área habitable (sqft)"),
                       ("beds_above", "Habitaciones"), ("grade", "Grado Iowa"), ("condition_label", "Condición")]:
        if result.get(key) not in (None, ""):
            facts.append(f"{label}: {result[key]}")
    if facts:
        lines.append("- **Características**: " + " · ".join(str(f) for f in facts))
    lines.append("\n_Fuente: padrón del Ames City Assessor 2024 + AVM XGBoost; montos en dólares de hoy (FHFA HPI). "
                 "Sin direcciones postales en los datos públicos._")
    return "\n".join(lines)


def _render_search(result: dict) -> str:
    if "error" in result:
        return f"No pude buscar parcelas: {result['error']}."
    total = result.get("total_matches", len(result.get("results", []) or []))
    results = result.get("results") or []
    if not results:
        return "### Búsqueda de parcelas\nNo hay parcelas del padrón 2024 que cumplan esos filtros."
    lines = [f"### {total} parcelas cumplen los filtros (mostrando {min(len(results), 10)})",
             "| PID | Barrio | Estilo | Año | Área (ft²) | Hab. | Valor AVM hoy | Avalúo hoy | Decisión |",
             "|---|---|---|---|---|---|---|---|---|"]
    for it in results[:10]:
        ad = it.get("agent_decision")
        dec = ad.get("final", "—") if isinstance(ad, dict) else (ad or "—")
        lines.append(f"| {it.get('pid')} | {it.get('neighborhood') or '—'} | {it.get('style') or '—'} | "
                     f"{it.get('year_built', '—')} | {it.get('living_area', '—')} | {it.get('beds_above', '—')} | "
                     f"{_fmt_money(it.get('avm_value_today'))} | {_fmt_money(it.get('assessed_today'))} | {dec} |")
    lines.append("\n_Padrón del Ames City Assessor 2024; valores AVM en dólares de hoy._")
    return "\n".join(lines)


def _render_comparables(result: dict, pid: Optional[str]) -> str:
    if "error" in result:
        return f"No pude obtener comparables: {result['error']}."
    comps = result.get("comparables") or result.get("results") or []
    header = f"### Comparables de {pid}" if pid else "### Comparables"
    lines = [header, "| PID venta | Barrio | Fecha | Precio | Precio hoy | Área | Similitud | Distancia |",
             "|---|---|---|---|---|---|---|---|"]
    for c in comps[:10]:
        d = str(c.get("sale_date", ""))
        dist = c.get("distance_km")
        lines.append(f"| {c.get('pid', '?')} | {c.get('neighborhood', '')} | {d[:4]}-{d[4:]} | "
                     f"{_fmt_money(c.get('price'))} | {_fmt_money(c.get('price_today'))} | "
                     f"{c.get('gr_liv_area', '?')} | {c.get('similarity', 0):.2f} | "
                     f"{f'{dist:.2f} km' if dist is not None else '—'} |")
    if not comps:
        return header + "\nSin comparables disponibles."
    est = result.get("comps_estimate")
    if est:
        line = f"\n**Estimación por comparables**: {_fmt_money(est.get('value_today'))}"
        if est.get("diff_vs_avm_pct") is not None:
            line += f" ({est['diff_vs_avm_pct']:+.1f}% vs AVM {_fmt_money(est.get('avm_value_today'))})"
        lines.append(line)
    lines.append("\n_Ventas reales De Cock 2006-2010 llevadas a dólares de hoy con el FHFA HPI de Ames._")
    return "\n".join(lines)


def _render_market(result: dict) -> str:
    if "error" in result:
        return f"No pude obtener el mercado: {result['error']}."
    lines = ["### Mercado inmobiliario de Ames"]
    for key, label, field, period, money in [("fhfa_hpi", "Índice FHFA HPI (Ames MSA)", "hpi", "period", False),
                                             ("zillow_zhvi", "Zillow ZHVI (valor típico de vivienda)", "zhvi", "month", True)]:
        d = result.get(key) or {}
        series = d.get("series") or []
        if not series:
            continue
        first, last = series[0], series[-1]
        fmt = _fmt_money if money else (lambda v: f"{v:,.1f}")
        change = 100 * (last[field] / first[field] - 1) if first.get(field) else None
        line = (f"- **{label}**: {fmt(first[field])} ({first[period]}) → {fmt(last[field])} ({last[period]})"
                + (f", **{change:+.1f}%** en el período" if change is not None else ""))
        if d.get("yoy_pct") is not None:
            line += f"; variación interanual {d['yoy_pct']:+.1f}%"
        lines.append(line)
    if len(lines) == 1:
        lines.append("Sin series de mercado disponibles.")
    lines.append("\n_Fuentes: FHFA All-Transactions HPI (CBSA 11180) y Zillow ZHVI de la ciudad de Ames._")
    return "\n".join(lines)


_AREA_LABELS = [("neighborhood", "Barrio", None), ("subdivision", "Subdivisión", None),
                ("n_sales", "Ventas reales 2006-2010", None), ("median_price_today", "Precio mediano de venta (hoy)", "$"),
                ("median_ppsf_today", "Precio mediano por ft² (hoy)", "$"), ("n_parcels", "Parcelas en el padrón 2024", None),
                ("median_avm_today", "Valor AVM mediano (hoy)", "$"), ("median_assessed_2024", "Avalúo mediano 2024", "$"),
                ("median_year_built", "Año de construcción mediano", None), ("median_living_area", "Área habitable mediana (ft²)", None)]


def _render_area(result: dict) -> str:
    if "error" in result:
        return f"No pude obtener estadísticas de la zona: {result['error']}."
    lines = ["### Estadísticas de la zona"]
    for key, label, kind in _AREA_LABELS:
        v = result.get(key)
        if v is not None:
            if kind == "$":
                txt = _fmt_money(v)
            elif isinstance(v, (int, float)):
                txt = f"{v:.0f}" if "year" in key else f"{v:,.0f}"  # años sin separador de miles
            else:
                txt = v
            lines.append(f"- **{label}**: {txt}")
    if len(lines) == 1:
        lines.append("Sin estadísticas disponibles.")
    return "\n".join(lines)


def _render_knowledge(result: dict) -> str:
    if "error" in result:
        return f"No pude buscar en la documentación: {result['error']}."
    hits = result.get("results") or result.get("hits") or []
    if not hits:
        return "No encontré documentación relevante para esa pregunta."
    lines = ["### Según la documentación del proyecto"]
    for hit in hits[:2]:
        text = (hit.get("text") or hit.get("content") or "").strip()
        if len(text) > 1500:  # cortar en el último párrafo completo, nunca a mitad de palabra
            cut = text.rfind("\n\n", 0, 1500)
            text = text[:cut if cut > 400 else text.rfind(" ", 0, 1500)] + " …"
        lines.append(f"\n**{hit.get('title') or hit.get('source', '')}** _( {hit.get('source', '')} )_\n\n{text}")
    others = [f"{h.get('title')} ({h.get('source')})" for h in hits[2:5] if h.get("title")]
    if others:
        lines.append("\n_Ver también: " + "; ".join(others) + "._")
    return "\n".join(lines)


def _render_model_card(result: dict) -> str:
    if "error" in result:
        return f"No pude obtener la ficha del modelo: {result['error']}."
    avm = result.get("avm_metrics") or {}
    t = result.get("test_reward") or {}
    lines = ["### Ficha del modelo SAVI"]
    if avm:
        lines.append(f"- **AVM XGBoost** (CV 5-fold, out-of-fold): R² log {avm.get('r2_log', 0):.3f} · "
                     f"R² USD {avm.get('r2_usd', 0):.3f} · MAE {_fmt_money(avm.get('mae_usd'))} · "
                     f"MAPE {100 * avm.get('mape', 0):.1f}%")
    if result.get("selected_rule"):
        lines.append(f"- **Regla de decisión elegida en validación**: `{result['selected_rule']}`")
    if t:
        ci = t.get("delta_vs_vi_ci95") or [None, None]
        lines.append(f"- **Test** (ventas nunca vistas): reward {t.get('reward', 0):.1f}; diferencia vs Value Iteration "
                     f"{t.get('delta_vs_vi', 0):+.1f} (IC95% [{ci[0]:.1f}, {ci[1]:.1f}]) → "
                     + ("**significativa**" if t.get("delta_significant") else "**no significativa**"))
    share = result.get("portfolio_action_share") or {}
    if share:
        lines.append("- **Cartera 2024**: " + " · ".join(f"{k} {100 * v:.1f}%" for k, v in share.items()))
    return "\n".join(lines)


def _safe_render(fn):
    """Un renderizador que falla no debe tumbar el chat: degrada a un mensaje y registra el error."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("Fallo renderizando %s", fn.__name__)
            return f"_(No pude formatear el resultado de {fn.__name__.removeprefix('_render_')}.)_"
    wrapper.__name__ = fn.__name__
    return wrapper


for _name in ("_render_get_parcel", "_render_search", "_render_comparables", "_render_market",
              "_render_area", "_render_knowledge", "_render_model_card"):
    globals()[_name] = _safe_render(globals()[_name])


def _run_deterministic_mode(
    tools_mod, message: str, history: Optional[list[dict]] = None, fallback_reason: Optional[str] = None
) -> dict:
    text = message or ""
    lower = text.lower()
    pid = _extract_pid(text)

    tool_calls: list[dict] = []
    tool_results_for_evidence: list[dict] = []
    answer_parts: list[str] = []

    def call(name: str, tool_input: dict) -> dict:
        output = _run_tool(tools_mod, name, tool_input)
        tool_results_for_evidence.append(output)
        tool_calls.append({"name": name, "input": tool_input, "summary": _summarize_tool_result(name, output)})
        return output

    wants_comparables = any(w in lower for w in ("comparable", "similar", "parecid"))
    wants_market = any(w in lower for w in ("mercado", "tendencia", "índice", "indice", "hpi", "zillow"))
    wants_methodology = any(
        w in lower for w in ("metodolog", "cómo funciona", "como funciona", "recompensa", "reward", "limitacion")
    )
    wants_decision = any(w in lower for w in ("decisión", "decision", "aprobar", "revisar", "rechazar"))
    search_filters = _extract_search_filters(text)
    wants_search = any(w in lower for w in ("buscar", "listar", "encuentra", "encontrar", "mostrar")) or bool(
        search_filters
    )
    wants_area = any(w in lower for w in ("barrio", "zona", "neighborhood")) and not pid and not wants_search

    wants_model_quality = any(w in lower for w in ("qué tan bueno", "que tan bueno", "precisión", "precision",
                                                   "métrica", "metrica", "exactitud", "confiable", "mape", "r2",
                                                   "r²", "error del modelo", "qué tan preciso", "que tan preciso"))

    if wants_model_quality and not pid:
        answer_parts.append(_render_model_card(call("model_card", {})))
        answer_parts.append(_render_knowledge(call("search_knowledge", {"query": "limitaciones métricas AVM validación"})))
    elif wants_methodology:
        answer_parts.append(_render_knowledge(call("search_knowledge", {"query": text})))
        if wants_decision or "model" in lower or "modelo" in lower:
            answer_parts.append(_render_model_card(call("model_card", {})))
    elif pid and (wants_comparables or wants_decision or not (wants_search or wants_area or wants_market)):
        answer_parts.append(_render_get_parcel(call("get_parcel", {"pid": pid}), pid))
        if wants_comparables:
            k_match = _K_RE.search(lower)
            k = int(k_match.group(1)) if k_match else 5
            answer_parts.append(_render_comparables(call("find_comparables", {"pid": pid, "k": k}), pid))
    elif wants_market:
        since_match = re.search(r"desde\s+(\d{4})", lower)
        market_input = {"since": since_match.group(1)} if since_match else {}
        answer_parts.append(_render_market(call("market_trend", market_input)))
    elif wants_search:
        # Búsqueda con filtros (incluye el caso "barrio X con N habitaciones/menos de $Y"):
        # search_parcels también acepta `neighborhood`, así que una zona mencionada junto
        # con filtros numéricos se resuelve acá, no como un area_stats aislado.
        neighborhood = _extract_neighborhood(text)
        if neighborhood:
            search_filters["neighborhood"] = neighborhood
        answer_parts.append(_render_search(call("search_parcels", search_filters)))
    elif wants_area:
        neighborhood = _extract_neighborhood(text)
        area_input = {"neighborhood": neighborhood} if neighborhood else {}
        answer_parts.append(_render_area(call("area_stats", area_input)))
    elif pid:
        answer_parts.append(_render_get_parcel(call("get_parcel", {"pid": pid}), pid))
    else:
        answer_parts.append(_render_knowledge(call("search_knowledge", {"query": text})))

    if not answer_parts:
        answer_parts.append(
            "No pude identificar qué necesitás. Probá con un PID de 10 dígitos, el nombre de un "
            "barrio, o palabras como 'comparables', 'mercado' o 'metodología'."
        )

    if fallback_reason:
        answer_parts.append(f"\n_(Nota: respuesta en modo determinista — {fallback_reason})_")

    return {
        "answer": "\n\n".join(answer_parts),
        "mode": "deterministic",
        "tool_calls": tool_calls,
        "evidence": _collect_evidence(tool_results_for_evidence),
    }


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #

def answer(
    message: str,
    history: Optional[list[dict]] = None,
    *,
    client: Any = None,
    tools_module: Any = None,
) -> dict:
    """
    Responde `message` (con `history` opcional de turnos previos) usando el modo
    Claude si hay credenciales disponibles, con fallback automático a modo
    determinista ante cualquier error del SDK o rechazo del modelo.

    `client` y `tools_module` son puntos de inyección para tests (nunca se debe
    golpear la red de verdad ni SSM/Anthropic reales desde el test suite).
    """
    tools_mod = tools_module if tools_module is not None else _tools_module()
    active_client = client if client is not None else _build_anthropic_client()

    if active_client is not None:
        try:
            return _run_claude_mode(active_client, tools_mod, message, history)
        except ClaudeRefusalError as exc:
            logger.warning("El modelo rechazó la solicitud: %s", exc)
            return _run_deterministic_mode(tools_mod, message, history, fallback_reason=str(exc))
        except _CLAUDE_FALLBACK_EXCEPTIONS as exc:  # type: ignore[misc]
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Modo Claude falló, usando modo determinista: %s", reason)
            return _run_deterministic_mode(tools_mod, message, history, fallback_reason=reason)

    return _run_deterministic_mode(tools_mod, message, history)
