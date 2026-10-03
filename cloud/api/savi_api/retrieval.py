# -*- coding: utf-8 -*-
"""
SAVI Agent API — consultas de negocio sobre el Store: parcelas, búsqueda con filtros,
comparables (k-NN), estadísticas de zona, tendencia de mercado, ficha del modelo y
búsqueda de conocimiento (BM25 puro-Python sobre `knowledge/*.md`).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from . import inference as inf
from .store import Store, normalize_name

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


# ════════════════════════════════════════════════════════════════════
# Parcela + decisión del agente
# ════════════════════════════════════════════════════════════════════
def get_parcel(store: Store, pid: str) -> dict[str, Any]:
    p = store.parcels_by_pid.get(pid)
    if p is None:
        return {"error": f"Parcela {pid} no encontrada"}
    out = {k: v for k, v in p.items() if k != "features"}
    dec = store.decisions.get(pid)
    if dec:
        out["agent_decision"] = {
            "final": dec["final"], "dqn": dec["dqn"], "consensus": dec["consensus"],
            "margin": dec["margin"], "q_values": dict(zip(["APROBAR", "REVISAR", "RECHAZAR"], dec["q"])),
            "rule": store.policy.get("selected_rule"),
        }
    sales = store.sales_by_pid.get(pid, [])
    if sales:
        out["sales_history"] = [
            {"sale_date": s.get("sale_date"), "price": s.get("price"), "price_today": s.get("price_today")}
            for s in sales]
    return out


# ════════════════════════════════════════════════════════════════════
# Búsqueda de parcelas con filtros
# ════════════════════════════════════════════════════════════════════
_SORT_KEYS = {"avm_value_today", "gap_vs_assessed", "year_built", "living_area"}


def search_parcels(store: Store, *, neighborhood: str | None = None, subdivision: str | None = None,
                   min_value: float | None = None, max_value: float | None = None,
                   min_area: float | None = None, max_area: float | None = None,
                   min_year: int | None = None, max_year: int | None = None,
                   beds_min: int | None = None, style_contains: str | None = None,
                   decision: str | None = None, sort_by: str = "avm_value_today",
                   descending: bool = True, limit: int = 10) -> dict[str, Any]:
    limit = max(1, min(int(limit or 10), 25))
    sort_by = sort_by if sort_by in _SORT_KEYS else "avm_value_today"

    nb_key = normalize_name(neighborhood) if neighborhood else None
    sub_key = normalize_name(subdivision) if subdivision else None
    style_key = style_contains.lower() if style_contains else None

    results = []
    for p in store.parcels:
        if nb_key and normalize_name(p.get("neighborhood")) != nb_key:
            continue
        if sub_key and normalize_name(p.get("subdivision")) != sub_key:
            continue
        if min_value is not None and (p.get("avm_value_today") or 0) < min_value:
            continue
        if max_value is not None and (p.get("avm_value_today") or 0) > max_value:
            continue
        if min_area is not None and (p.get("living_area") or 0) < min_area:
            continue
        if max_area is not None and (p.get("living_area") or 0) > max_area:
            continue
        if min_year is not None and (p.get("year_built") or 0) < min_year:
            continue
        if max_year is not None and (p.get("year_built") or 0) > max_year:
            continue
        if beds_min is not None and (p.get("beds_above") or 0) < beds_min:
            continue
        if style_key and style_key not in (p.get("style") or "").lower():
            continue
        if decision is not None:
            dec = store.decisions.get(p["pid"])
            if not dec or dec["final"] != decision:
                continue
        results.append(p)

    results.sort(key=lambda p: (p.get(sort_by) if p.get(sort_by) is not None else -math.inf), reverse=descending)
    total = len(results)
    out = []
    for p in results[:limit]:
        rec = {k: v for k, v in p.items() if k != "features"}
        dec = store.decisions.get(p["pid"])
        if dec:
            rec["agent_decision"] = dec["final"]
        out.append(rec)
    return {"total_matches": total, "results": out}


# ════════════════════════════════════════════════════════════════════
# Comparables (k-NN sobre ventas in_training)
# ════════════════════════════════════════════════════════════════════
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_SIM_SCALE = 3.0  # escala de la similitud exp(-d/escala)


def find_comparables(store: Store, *, pid: str | None = None, overrides: dict[str, Any] | None = None,
                     k: int = 5) -> dict[str, Any]:
    k = max(1, min(int(k or 5), 10))
    overrides = overrides or {}
    if pid and pid not in store.parcels_by_pid:
        return {"error": f"Parcela {pid} no encontrada"}

    features, map_area, _meta = inf.features_from_overrides(store, overrides, base_pid=pid)
    mean, scale = store.kmeans["mean"], store.kmeans["scale"]
    z = [(v - mean[i]) / scale[i] if scale[i] else 0.0 for i, v in enumerate(features)]

    base_parcel = store.parcels_by_pid.get(pid) if pid else None
    lat = overrides.get("lat") or (base_parcel.get("lat") if base_parcel else None)
    lon = overrides.get("lon") or (base_parcel.get("lon") if base_parcel else None)
    neighborhood = overrides.get("neighborhood") or (base_parcel.get("neighborhood") if base_parcel else None)
    nb_key = normalize_name(neighborhood) if neighborhood else None

    scored = []
    for c in store.comp_vectors:
        if pid and c["sale"]["pid"] == pid:
            continue
        d2 = sum((z[i] - c["z"][i]) ** 2 for i in range(len(z)))
        dist = math.sqrt(d2)
        geo_term = 0.0
        if lat is not None and lon is not None and c["lat"] is not None and c["lon"] is not None:
            geo_km = _haversine_km(lat, lon, c["lat"], c["lon"])
            geo_term = geo_km / 10.0  # ~10 km ≈ 1 unidad estandarizada de distancia
        # Penalización (no bono) si es de otro barrio → el puntaje es una distancia ≥ 0
        other_nb = 0.5 if (nb_key and normalize_name(c["neighborhood"]) != nb_key) else 0.0
        total = dist + geo_term + other_nb
        scored.append((total, dist, geo_term * 10.0 if geo_term else None, c))

    scored.sort(key=lambda t: t[0])
    top = scored[:k]
    out = []
    for total, dist, geo_km, c in top:
        s = c["sale"]
        # Similitud absoluta y acotada en (0, 1]: comparable entre consultas (0.5 ≈ distancia 2.1)
        similarity = math.exp(-total / _SIM_SCALE)
        out.append({
            "pid": s["pid"], "similarity": round(similarity, 3),
            "distance_km": round(geo_km, 2) if geo_km is not None else None, "price": s.get("price"),
            "price_today": s.get("price_today"), "sale_date": s.get("sale_date"),
            "neighborhood": s.get("neighborhood"), "lat": s.get("lat"), "lon": s.get("lon"),
            "gr_liv_area": s.get("gr_liv_area"), "year_built": s.get("year_built"),
            "overall_qual": s.get("overall_qual"), "bedrooms": s.get("bedrooms"),
            "house_style": s.get("house_style"),
        })
    # Estimación por comparables: media ponderada por similitud del precio en $ de hoy,
    # ajustada por $/sqft cuando se conoce el área de la casa consultada.
    estimate = None
    w = [r["similarity"] for r in out if r.get("price_today")]
    if w and sum(w) > 0:
        area_q = overrides.get("living_area") or (base_parcel.get("living_area") if base_parcel else None)
        if area_q:
            ppsf = sum(r["similarity"] * r["price_today"] / r["gr_liv_area"] for r in out
                       if r.get("price_today") and r.get("gr_liv_area")) / sum(w)
            estimate = {"method": "similarity-weighted $/sqft × living_area", "value_today": round(ppsf * area_q),
                        "price_per_sqft_today": round(ppsf, 1)}
        else:
            estimate = {"method": "similarity-weighted price_today",
                        "value_today": round(sum(r["similarity"] * r["price_today"] for r in out
                                                 if r.get("price_today")) / sum(w))}
        if base_parcel and base_parcel.get("avm_value_today"):
            estimate["avm_value_today"] = base_parcel["avm_value_today"]
            estimate["diff_vs_avm_pct"] = round(100 * (estimate["value_today"] / base_parcel["avm_value_today"] - 1), 1)
    return {"k": len(out), "query_map_area": map_area, "comps_estimate": estimate, "results": out}


# ════════════════════════════════════════════════════════════════════
# Estadísticas de zona
# ════════════════════════════════════════════════════════════════════
def area_stats(store: Store, *, neighborhood: str | None = None, subdivision: str | None = None,
               cluster: int | None = None) -> dict[str, Any]:
    if neighborhood:
        rec = store.resolve_neighborhood(neighborhood)
        if not rec:
            return {"error": f"Barrio '{neighborhood}' no encontrado"}
        return {"type": "neighborhood", **rec}
    if subdivision:
        rec = store.resolve_subdivision(subdivision)
        if not rec:
            return {"error": f"Subdivisión '{subdivision}' no encontrada"}
        return {"type": "subdivision", **rec}
    if cluster is not None:
        rec = store.clusters_by_id.get(int(cluster))
        if not rec:
            return {"error": f"Cluster {cluster} no existe"}
        return {"type": "cluster", **rec}
    return {"error": "Indique neighborhood, subdivision o cluster"}


# ════════════════════════════════════════════════════════════════════
# Mercado (FHFA HPI + Zillow ZHVI)
# ════════════════════════════════════════════════════════════════════
def market_trend(store: Store, *, since: str | None = None) -> dict[str, Any]:
    m = store.market
    out: dict[str, Any] = {}
    hpi = m.get("fhfa_hpi")
    if hpi:
        series = hpi["series"]
        if since:
            series = [s for s in series if s["period"] >= f"{since}Q1"]
        out["fhfa_hpi"] = {"source": hpi["source"], "latest": hpi["latest"], "yoy_pct": hpi.get("yoy_pct"),
                           "since_2010_pct": hpi.get("since_2010_pct"), "series": series}
    zh = m.get("zillow_zhvi")
    if zh:
        series = zh["series"]
        if since:
            series = [s for s in series if s["month"] >= f"{since}-01"]
        out["zillow_zhvi"] = {"source": zh["source"], "latest": zh["latest"], "yoy_pct": zh.get("yoy_pct"),
                              "series": series}
    return out


# ════════════════════════════════════════════════════════════════════
# Model card
# ════════════════════════════════════════════════════════════════════
def model_card(store: Store) -> dict[str, Any]:
    man, pol = store.manifest, store.policy
    return {
        "run_id": man.get("run_id"), "version": man.get("version"), "hpi_latest": man.get("hpi_latest"),
        "counts": man.get("counts"), "avm_metrics": man.get("avm"),
        "kmeans_silhouette": man.get("kmeans_silhouette"),
        "selected_rule": pol.get("selected_rule"), "gated_tau": pol.get("gated_tau"),
        "test_reward": pol.get("test"), "rules_reward": pol.get("rules"),
        "reward_function": pol.get("reward_function"),
        "portfolio_action_share": pol.get("portfolio_action_share"),
        "policy_final": pol.get("policy_final"),
    }


# ════════════════════════════════════════════════════════════════════
# BM25 sobre knowledge/*.md (partido por encabezados Markdown)
# ════════════════════════════════════════════════════════════════════
_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "que", "se", "por", "con",
    "para", "es", "del", "al", "lo", "como", "su", "sus", "o", "no", "sí", "más", "e", "u",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _split_chunks(text: str, source: str) -> list[dict]:
    """Divide un .md en fragmentos por encabezado (# .. ######)."""
    lines = text.splitlines()
    chunks, current_title, current_lines = [], source, []

    def flush():
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                chunks.append({"source": source, "title": current_title, "text": body})

    for line in lines:
        if re.match(r"^#{1,6}\s+", line):
            flush()
            current_title = re.sub(r"^#{1,6}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return chunks


_BM25_CACHE: dict | None = None


def _load_bm25_index() -> dict:
    global _BM25_CACHE
    if _BM25_CACHE is not None:
        return _BM25_CACHE
    chunks = []
    if KNOWLEDGE_DIR.exists():
        for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
            chunks.extend(_split_chunks(md.read_text(encoding="utf-8"), md.name))
    docs = [_tokenize(f"{c['title']} {c['text']}") for c in chunks]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N if N else 0.0
    df: dict[str, int] = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1
    idf = {term: math.log(1 + (N - n + 0.5) / (n + 0.5)) for term, n in df.items()}
    _BM25_CACHE = {"chunks": chunks, "docs": docs, "idf": idf, "avgdl": avgdl, "N": N}
    return _BM25_CACHE


def search_knowledge(query: str, limit: int = 5, k1: float = 1.5, b: float = 0.75) -> dict[str, Any]:
    idx = _load_bm25_index()
    if idx["N"] == 0:
        return {"results": []}
    q_terms = _tokenize(query)
    scores = []
    for i, doc in enumerate(idx["docs"]):
        if not doc:
            scores.append(0.0)
            continue
        dl = len(doc)
        freq: dict[str, int] = {}
        for t in doc:
            freq[t] = freq.get(t, 0) + 1
        score = 0.0
        for t in q_terms:
            if t not in freq:
                continue
            idf = idx["idf"].get(t, 0.0)
            f = freq[t]
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / idx["avgdl"]))
        scores.append(score)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in order[:max(1, min(int(limit or 5), 10))]:
        if scores[i] <= 0:
            continue
        c = idx["chunks"][i]
        results.append({"source": c["source"], "title": c["title"], "score": round(scores[i], 3),
                        "text": c["text"][:1200]})
    return {"results": results}
