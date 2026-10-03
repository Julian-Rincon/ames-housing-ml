# -*- coding: utf-8 -*-
"""
SAVI Agent API — TOOLS (formato Anthropic) + run_tool(name, input) → dict.

`run_tool` nunca lanza: cualquier excepción se captura y se devuelve como {"error": "..."}.
Las salidas son compactas (sin arrays de `features`, dinero redondeado a dólares enteros,
tamaños de lista acotados) porque se inyectan en el contexto de un LLM.
"""
from __future__ import annotations

from typing import Any

from . import inference as inf
from . import retrieval as ret
from .store import get_store


def _round_money(d: dict) -> dict:
    """Redondea a dólares enteros cualquier valor que parezca dinero (heurística por nombre)."""
    money_keys = {"avm_value_today", "assessed_today", "assessed_2024", "land_value_2024",
                 "price", "price_today", "median_price_today", "median_avm_today",
                 "median_assessed_2024", "avm_range_low", "avm_range_high", "avm_oof",
                 "land_value_today"}
    out = {}
    for k, v in d.items():
        if k in money_keys and isinstance(v, (int, float)):
            out[k] = int(round(v))
        elif isinstance(v, dict):
            out[k] = _round_money(v)
        elif isinstance(v, list):
            out[k] = [_round_money(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


# ════════════════════════════════════════════════════════════════════
# Definición de herramientas (formato Anthropic tool use)
# ════════════════════════════════════════════════════════════════════
TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_parcel",
        "description": (
            "Obtiene todos los datos de una parcela del padrón 2024 de Ames por su PID "
            "(número de parcela, 10 dígitos), incluida la decisión del agente RL (APROBAR/"
            "REVISAR/RECHAZAR), sus Q-values y su historial de ventas si existe. Úsala cuando "
            "el usuario pregunte por una parcela/propiedad específica identificada por PID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"pid": {"type": "string", "description": "PID de 10 dígitos, p.ej. '0526301100'"}},
            "required": ["pid"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_parcels",
        "description": (
            "Busca parcelas del padrón 2024 que cumplan filtros (barrio, subdivisión, rango de "
            "valor AVM, área habitable, año de construcción, dormitorios mínimos, texto del "
            "estilo, decisión del agente) y las ordena. Úsala para preguntas tipo 'parcelas en "
            "tal barrio con valor entre X e Y' o 'casas que el agente rechazó en tal zona'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhood": {"type": "string"}, "subdivision": {"type": "string"},
                "min_value": {"type": "number"}, "max_value": {"type": "number"},
                "min_area": {"type": "number"}, "max_area": {"type": "number"},
                "min_year": {"type": "integer"}, "max_year": {"type": "integer"},
                "beds_min": {"type": "integer"}, "style_contains": {"type": "string"},
                "decision": {"type": "string", "enum": ["APROBAR", "REVISAR", "RECHAZAR"]},
                "sort_by": {"type": "string",
                           "enum": ["avm_value_today", "gap_vs_assessed", "year_built", "living_area"]},
                "descending": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "find_comparables",
        "description": (
            "Encuentra ventas reales (De Cock, 2006-2010, ajustadas a dólares de hoy) parecidas "
            "a una parcela (por PID) o a una propiedad hipotética (por características). La "
            "similitud combina distancia estandarizada en las características y distancia "
            "geográfica; favorece ventas del mismo barrio. Úsala para 'comparables' o '¿con qué "
            "se compara esta casa?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pid": {"type": "string"},
                "living_area": {"type": "number"}, "year_built": {"type": "integer"},
                "beds_above": {"type": "integer"}, "grade": {"type": "string"},
                "condition": {"type": "string"}, "neighborhood": {"type": "string"},
                "style": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "value_property",
        "description": (
            "Valúa una propiedad con el AVM (XGBoost reproducido en Python puro) y aplica la "
            "regla de decisión del agente RL (APROBAR/REVISAR/RECHAZAR) con sus Q-values. Puede "
            "partir de una parcela existente (`pid`) y aplicar cambios hipotéticos (what-if: "
            "área habitable, año, dormitorios, grado, condición, barrio, etc.), o evaluar una "
            "propiedad totalmente nueva sin `pid`. Úsala para '¿cuánto vale esta casa?' o "
            "'¿qué pasaría si le agrego un dormitorio?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pid": {"type": "string"},
                "living_area": {"type": "number"}, "year_built": {"type": "integer"},
                "beds_above": {"type": "integer"}, "beds_below": {"type": "integer"},
                "rooms_above": {"type": "integer"}, "n_fireplaces": {"type": "integer"},
                "n_plumbing": {"type": "integer"}, "n_garages": {"type": "integer"},
                "grade": {"type": "string"}, "condition": {"type": "string"},
                "style": {"type": "string"}, "basement_type": {"type": "string"},
                "basement_finished": {"type": "boolean"}, "garage_type": {"type": "string"},
                "neighborhood": {"type": "string"}, "subdivision": {"type": "string"},
                "map_area": {"type": "string"}, "land_value": {"type": "number"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "area_stats",
        "description": (
            "Estadísticas agregadas de un barrio, una subdivisión o un cluster K-Means "
            "(mediana de valor AVM, avalúo, área habitable, año de construcción, número de "
            "parcelas/ventas, coordenadas). Úsala para preguntas sobre una zona en general, no "
            "una parcela puntual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhood": {"type": "string"}, "subdivision": {"type": "string"},
                "cluster": {"type": "integer", "minimum": 0, "maximum": 5},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "market_trend",
        "description": (
            "Serie histórica del índice de precios de vivienda FHFA (trimestral) y Zillow ZHVI "
            "(mensual) para Ames, con variación interanual. Úsala para preguntas sobre el "
            "mercado inmobiliario en general, no una propiedad puntual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"since": {"type": "string", "description": "Año desde el cual filtrar, p.ej. '2015'"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "search_knowledge",
        "description": (
            "Busca en la documentación interna de SAVI (metodología, fuentes de datos, función "
            "de recompensa y significado de las decisiones, limitaciones del sistema, glosario "
            "de variables como grado/condición/map_area). Úsala para preguntas conceptuales "
            "sobre cómo funciona el sistema, no para datos de una parcela concreta."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "model_card",
        "description": (
            "Ficha técnica del modelo: métricas del AVM (R², MAE, MAPE), silhouette de K-Means, "
            "regla de decisión seleccionada, recompensa en validación/prueba con intervalos de "
            "confianza, y distribución de acciones en la cartera. Úsala para preguntas sobre la "
            "calidad o el desempeño del modelo en sí."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

_TOOL_NAMES = {t["name"] for t in TOOLS}


def run_tool(name: str, tool_input: dict[str, Any] | None) -> dict[str, Any]:
    """Ejecuta una herramienta por nombre. Nunca lanza: errores → {"error": "..."}."""
    tool_input = tool_input or {}
    try:
        if name not in _TOOL_NAMES:
            return {"error": f"Herramienta desconocida: {name}"}
        store = get_store()

        if name == "get_parcel":
            pid = str(tool_input.get("pid", "")).strip()
            if not pid:
                return {"error": "Falta 'pid'"}
            return _round_money(ret.get_parcel(store, pid))

        if name == "search_parcels":
            return _round_money(ret.search_parcels(store, **{
                k: v for k, v in tool_input.items() if k in {
                    "neighborhood", "subdivision", "min_value", "max_value", "min_area",
                    "max_area", "min_year", "max_year", "beds_min", "style_contains",
                    "decision", "sort_by", "descending", "limit"}
            }))

        if name == "find_comparables":
            pid = tool_input.get("pid")
            k = tool_input.get("k", 5)
            overrides = {kk: v for kk, v in tool_input.items() if kk not in {"pid", "k"}}
            return _round_money(ret.find_comparables(store, pid=pid, overrides=overrides, k=k))

        if name == "value_property":
            pid = tool_input.get("pid")
            overrides = {k: v for k, v in tool_input.items() if k != "pid"}
            return _round_money(inf.value_property_core(store, pid=pid, overrides=overrides))

        if name == "area_stats":
            return _round_money(ret.area_stats(store, **{
                k: v for k, v in tool_input.items() if k in {"neighborhood", "subdivision", "cluster"}
            }))

        if name == "market_trend":
            return ret.market_trend(store, since=tool_input.get("since"))

        if name == "search_knowledge":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return {"error": "Falta 'query'"}
            return ret.search_knowledge(query, limit=tool_input.get("limit", 5))

        if name == "model_card":
            return _round_money(ret.model_card(store))

        return {"error": f"Herramienta no implementada: {name}"}
    except Exception as exc:  # run_tool NUNCA debe lanzar
        return {"error": f"{type(exc).__name__}: {exc}"}
