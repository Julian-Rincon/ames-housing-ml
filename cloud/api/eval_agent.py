#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación del agente SAVI: compara modos (determinista vs Groq) sobre un set fijo.

Métricas por pregunta:
- tool_ok:   el agente llamó a la herramienta esperada.
- grounded:  todas las cifras en US$ de la respuesta aparecen en la evidencia
             devuelta por las herramientas o en la ficha del modelo, o son un rango
             valor × (1 ± MAPE) derivado de ellas (tolerancia 1 %). Una respuesta sin
             cifras cuenta como grounded solo si la pregunta no pide cifras.
- answered:  respondió sin caer en el mensaje de "no pude identificar".

Uso (desde cloud/):
    SAVI_DATA_DIR=<dir con rag/ y serving/> .venv/bin/python api/eval_agent.py            # determinista
    SAVI_DATA_DIR=... GROQ_API_KEY=... .venv/bin/python api/eval_agent.py --mode groq
Escribe el resumen en api/eval_results_<modo>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))

from savi_api import agent  # noqa: E402

# (pregunta, herramienta esperada, pide_cifras)
QUESTIONS = [
    ("¿Cuánto vale hoy la parcela 0906200120?", {"get_parcel", "value_property"}, True),
    ("Dame 5 comparables de la parcela 0906200120", {"find_comparables"}, True),
    ("¿Qué decisión tomó el agente RL para 0906200120 y por qué?", {"get_parcel"}, False),
    ("Busca casas en College Creek con al menos 4 habitaciones por menos de 400 mil", {"search_parcels"}, True),
    ("¿Cómo está el mercado inmobiliario de Ames últimamente?", {"market_trend"}, False),
    ("Estadísticas del barrio Old Town", {"area_stats"}, True),
    ("¿Qué tan preciso es el modelo de valuación?", {"model_card"}, False),
    ("¿Qué metodología usa SAVI para decidir aprobar, revisar o rechazar?", {"search_knowledge", "model_card"}, False),
    ("Precio medio por pie cuadrado en Somerset", {"area_stats"}, True),
    ("¿Cuál es el índice FHFA más reciente de Ames?", {"market_trend"}, False),
]

_MONEY = re.compile(r"(?:US)?\$\s?([\d][\d.,]*)\s*(mil\b|k\b|millones?\b|M\b)?", re.IGNORECASE)


def _numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    return out


def _parse_money(raw: str, unit: str | None) -> float | None:
    s = raw.rstrip(".,")
    # 185.000 / 185,000 / 1.119.030 -> separadores de miles; 1,12 -> decimal
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", s):
        val = float(re.sub(r"[.,]", "", s))
    else:
        try:
            val = float(s.replace(",", "."))
        except ValueError:
            return None
    u = (unit or "").lower()
    if u in ("mil", "k"):
        val *= 1_000
    elif u.startswith("millon") or u == "m":
        val *= 1_000_000
    return val


def _with_derived(evidence_numbers: list[float]) -> list[float]:
    """Añade los rangos derivados de la evidencia: valor × (1 ± f), con f una fracción
    presente en la evidencia (p. ej. el MAPE), que es como las herramientas arman el rango."""
    fracs = sorted({round(f, 6) for f in evidence_numbers if 0 < f < 1})[:50]
    money = [e for e in evidence_numbers if abs(e) >= 100]
    derived = [e * (1 + s * f) for e in money for f in fracs for s in (1, -1)]
    return evidence_numbers + derived


def grounded(answer: str, evidence_numbers: list[float]) -> bool:
    evidence_numbers = _with_derived(evidence_numbers)
    for raw, unit in _MONEY.findall(answer):
        val = _parse_money(raw, unit)
        if val is None or val < 100:
            continue
        if not any(abs(val - e) <= max(1.0, 0.01 * abs(e)) for e in evidence_numbers):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deterministic", "groq"], default="deterministic")
    args = ap.parse_args()
    if args.mode == "deterministic":
        os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    agent._build_anthropic_client = lambda: None  # la evaluación nunca usa Claude

    rows = []
    for q, expected, wants_numbers in QUESTIONS:
        t0 = time.time()
        out = agent.answer(q)
        dt = time.time() - t0
        used = {c["name"] for c in out.get("tool_calls", [])}
        # la ficha del modelo (MAPE, R², etc.) es evidencia global: los rangos ± MAPE salen de ahí
        from savi_api import tools as _t
        ev_numbers = _numbers(_t.run_tool("model_card", {}))
        # cifras de todas las herramientas llamadas (re-ejecutadas, son deterministas)
        from savi_api import tools
        for c in out.get("tool_calls", []):
            ev_numbers += _numbers(tools.run_tool(c["name"], c["input"]))
        text = out.get("answer", "")
        has_money = bool(_MONEY.search(text))
        row = {
            "question": q,
            "mode": out.get("mode"),
            "tools": sorted(used),
            "tool_ok": bool(used & expected),
            "grounded": grounded(text, ev_numbers) and (has_money or not wants_numbers),
            "answered": "No pude identificar" not in text,
            "seconds": round(dt, 2),
            "answer": text[:600],
        }
        rows.append(row)
        print(f"[{row['mode']}] tool_ok={row['tool_ok']} grounded={row['grounded']} {dt:.1f}s  {q}")
        if args.mode == "groq":
            time.sleep(12)  # respeta el límite de tokens por minuto de la capa gratuita

    n = len(rows)
    summary = {
        "mode": args.mode,
        "n": n,
        "tool_accuracy": sum(r["tool_ok"] for r in rows) / n,
        "grounded_rate": sum(r["grounded"] for r in rows) / n,
        "answered_rate": sum(r["answered"] for r in rows) / n,
        "fell_back": sum(r["mode"] != args.mode for r in rows),
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
        "rows": rows,
    }
    (API_DIR / f"eval_results_{args.mode}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
