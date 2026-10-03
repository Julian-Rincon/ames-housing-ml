# -*- coding: utf-8 -*-
"""
Pruebas del núcleo RAG puro-Python de SAVI (`savi_api.store/inference/retrieval/tools`).

Verifican, contra los artefactos REALES del pipeline (`SAVI_DATA_DIR=../out/fixture`):
  (a) el AVM puro-Python reproduce `avm_value_today` de ≥2000 parcelas con <0.1% de error,
  (b) la asignación K-Means puro-Python coincide con `cluster` para todas las parcelas,
  (c) los Q-values del DQN puro-Python coinciden con `serving/decisions.json.gz["q"]` (±1e-2)
      para ≥500 parcelas,
  (d) `value_property(pid)` sin overrides reproduce `decisions.json["final"]`,
  más filtros de búsqueda, comparables, alias de barrio, BM25 y el esquema de TOOLS.

Ejecutar: cd api && SAVI_DATA_DIR=../out/fixture ../.venv/bin/python -m pytest tests/test_core.py -q
"""
from __future__ import annotations

import os
import random
import time

import pytest

os.environ.setdefault("SAVI_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "out", "fixture"))

# Estos tests verifican contra artefactos reales del pipeline: sin ellos (p.ej. en CI) se omiten
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(os.environ["SAVI_DATA_DIR"], "rag", "manifest.json")),
    reason="Faltan artefactos del pipeline en SAVI_DATA_DIR (genere out/fixture corriendo el pipeline)",
)

from savi_api import inference as inf  # noqa: E402
from savi_api import retrieval as ret  # noqa: E402
from savi_api import tools  # noqa: E402
from savi_api.store import get_store  # noqa: E402

RNG_SEED = 1234


@pytest.fixture(scope="module")
def store():
    return get_store(force_reload=True)


@pytest.fixture(scope="module")
def flat_trees(store):
    return inf.flatten_trees(store.avm_model)


# ════════════════════════════════════════════════════════════════════
# (a) AVM puro-Python vs avm_value_today
# ════════════════════════════════════════════════════════════════════
def test_avm_matches_pipeline_within_0_1_pct(store, flat_trees):
    rng = random.Random(RNG_SEED)
    sample = rng.sample(store.parcels, min(2500, len(store.parcels)))
    max_rel = 0.0
    for p in sample:
        row = inf.build_design_row(p["features"], p["map_area"], p["cluster"], store.avm_areas)
        pred = inf._xgb_predict_cached(store.avm_model, flat_trees, row)
        rel = abs(pred - p["avm_value_today"]) / p["avm_value_today"]
        max_rel = max(max_rel, rel)
    assert max_rel < 0.001, f"diferencia relativa máxima del AVM = {max_rel:.6f}"


# ════════════════════════════════════════════════════════════════════
# (b) K-Means puro-Python vs cluster
# ════════════════════════════════════════════════════════════════════
def test_kmeans_assignment_matches_all_parcels(store):
    mism = 0
    for p in store.parcels:
        cluster, _dist = inf.kmeans_assign(store.kmeans, p["features"])
        if cluster != p["cluster"]:
            mism += 1
    assert mism == 0, f"{mism}/{len(store.parcels)} parcelas con cluster distinto"


# ════════════════════════════════════════════════════════════════════
# (c) Q-values del DQN puro-Python vs serving/decisions.json.gz
# ════════════════════════════════════════════════════════════════════
def test_dqn_q_values_match_decisions(store, flat_trees):
    rng = random.Random(RNG_SEED)
    sample = rng.sample(store.parcels, min(800, len(store.parcels)))
    max_diff = 0.0
    for p in sample:
        pid = p["pid"]
        real = store.decisions.get(pid)
        if real is None:
            continue
        row = inf.build_design_row(p["features"], p["map_area"], p["cluster"], store.avm_areas)
        pred = inf._xgb_predict_cached(store.avm_model, flat_trees, row)
        state = inf.dqn_state_vector(store.dqn_state_def, row, pred, p["assessed_today"], inf.kmeans_assign(store.kmeans, p["features"])[1])
        q = inf.mlp_forward(store.dqn_weights, state)
        max_diff = max(max_diff, max(abs(q[i] - real["q"][i]) for i in range(3)))
    assert max_diff < 1e-2, f"diferencia máxima de Q = {max_diff}"


# ════════════════════════════════════════════════════════════════════
# (d) value_property(pid) sin overrides == decisions[pid]["final"]
# ════════════════════════════════════════════════════════════════════
def test_value_property_decision_matches_pipeline(store):
    rng = random.Random(RNG_SEED)
    sample = rng.sample(store.parcels, min(500, len(store.parcels)))
    mism = 0
    for p in sample:
        pid = p["pid"]
        real = store.decisions.get(pid)
        if real is None:
            continue
        out = inf.value_property_core(store, pid=pid, overrides={})
        if out["decision"]["final"] != real["final"]:
            mism += 1
    assert mism == 0, f"{mism}/{len(sample)} decisiones finales distintas"


def test_value_property_new_hypothetical_property_runs(store):
    out = inf.value_property_core(store, pid=None, overrides={
        "living_area": 1800, "year_built": 2015, "beds_above": 3, "grade": "5",
        "condition": "Normal", "neighborhood": "College Creek", "style": "2 Story Frame",
    })
    assert "avm_value_today" in out and out["avm_value_today"] > 0
    assert out["decision"]["final"] in {"APROBAR", "REVISAR", "RECHAZAR"}


# ════════════════════════════════════════════════════════════════════
# Búsqueda, comparables, alias, BM25
# ════════════════════════════════════════════════════════════════════
def test_search_parcels_filters_and_limit(store):
    out = ret.search_parcels(store, neighborhood="North Ames", min_value=100000, limit=5)
    assert out["total_matches"] >= len(out["results"])
    assert len(out["results"]) <= 5
    for r in out["results"]:
        assert r["neighborhood"] == "North Ames"
        assert r["avm_value_today"] >= 100000


def test_search_parcels_limit_capped_at_25(store):
    out = ret.search_parcels(store, limit=1000)
    assert len(out["results"]) <= 25


def test_find_comparables_sorted_and_prefers_same_neighborhood(store):
    pid = next(p["pid"] for p in store.parcels if p.get("neighborhood") == "College Creek")
    out = ret.find_comparables(store, pid=pid, k=8)
    assert out["k"] == len(out["results"]) > 0
    sims = [r["similarity"] for r in out["results"]]
    assert sims == sorted(sims, reverse=True)
    # al menos una porción relevante de los comparables debería ser del mismo barrio
    same_nb = sum(1 for r in out["results"] if r["neighborhood"] == "College Creek")
    assert same_nb >= 1


def test_neighborhood_alias_lookup(store):
    full = store.resolve_neighborhood("North Ames")
    short = store.resolve_neighborhood("NAmes")
    assert full is not None and short is not None
    assert full["neighborhood"] == short["neighborhood"]

    collgcr = store.resolve_neighborhood("CollgCr")
    assert collgcr is not None and collgcr["neighborhood"] == "College Creek"


def test_bm25_search_knowledge_returns_relevant_chunk():
    out = ret.search_knowledge("función de recompensa APROBAR REVISAR RECHAZAR")
    assert out["results"], "sin resultados BM25"
    assert any("recompensa" in r["source"] for r in out["results"])


def test_area_stats_cluster_and_errors(store):
    assert "n_parcels" in ret.area_stats(store, cluster=0)
    assert "error" in ret.area_stats(store, neighborhood="Zona Inexistente XYZ")
    assert "error" in ret.area_stats(store)


def test_market_trend_has_series(store):
    out = ret.market_trend(store)
    assert "fhfa_hpi" in out
    assert len(out["fhfa_hpi"]["series"]) > 0


def test_model_card_has_metrics(store):
    mc = ret.model_card(store)
    assert mc["avm_metrics"]["mape"] > 0
    assert mc["selected_rule"]


# ════════════════════════════════════════════════════════════════════
# TOOLS schema + run_tool
# ════════════════════════════════════════════════════════════════════
def test_all_tools_have_valid_schema():
    names = set()
    for t in tools.TOOLS:
        assert set(t.keys()) == {"name", "description", "input_schema"}
        assert isinstance(t["name"], str) and t["name"]
        assert isinstance(t["description"], str) and len(t["description"]) > 10
        schema = t["input_schema"]
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
        assert "properties" in schema
        names.add(t["name"])
    assert len(names) == len(tools.TOOLS), "nombres de herramientas duplicados"


def test_run_tool_bad_input_never_raises():
    assert "error" in tools.run_tool("no_existe", {})
    assert "error" in tools.run_tool("get_parcel", {})
    assert "error" in tools.run_tool("get_parcel", {"pid": "0000000000"})
    assert "error" in tools.run_tool("search_knowledge", {})
    # value_property sin pid ni overrides es una propiedad hipotética válida (no debe lanzar)
    out = tools.run_tool("value_property", None)
    assert "error" not in out


def test_run_tool_happy_paths(store):
    pid = store.parcels[0]["pid"]
    out = tools.run_tool("get_parcel", {"pid": pid})
    assert "error" not in out
    out = tools.run_tool("value_property", {"pid": pid})
    assert "error" not in out and "avm_value_today" in out
    out = tools.run_tool("search_parcels", {"limit": 3})
    assert "error" not in out and len(out["results"]) <= 3
    out = tools.run_tool("find_comparables", {"pid": pid, "k": 3})
    assert "error" not in out
    out = tools.run_tool("market_trend", {})
    assert "fhfa_hpi" in out
    out = tools.run_tool("model_card", {})
    assert "avm_metrics" in out
    out = tools.run_tool("search_knowledge", {"query": "K-Means clusters"})
    assert "results" in out
    out = tools.run_tool("area_stats", {"cluster": 1})
    assert "error" not in out


def test_tool_outputs_have_no_features_array(store):
    pid = store.parcels[0]["pid"]
    out = tools.run_tool("get_parcel", {"pid": pid})
    assert "features" not in out


# ════════════════════════════════════════════════════════════════════
# Rendimiento (informativo, no falla el build salvo que sea absurdo)
# ════════════════════════════════════════════════════════════════════
def test_cold_load_and_latency_report(capsys):
    t0 = time.time()
    store = get_store(force_reload=True)
    cold = time.time() - t0

    pid = store.parcels[0]["pid"]
    t0 = time.time()
    for _ in range(20):
        inf.value_property_core(store, pid=pid, overrides={})
    value_property_ms = (time.time() - t0) / 20 * 1000

    t0 = time.time()
    for _ in range(20):
        ret.find_comparables(store, pid=pid, k=5)
    comps_ms = (time.time() - t0) / 20 * 1000

    with capsys.disabled():
        print(f"\n[latencia] carga en frío={cold*1000:.1f}ms "
              f"value_property={value_property_ms:.2f}ms find_comparables={comps_ms:.2f}ms")

    assert cold < 2.0
    assert value_property_ms < 100
    assert comps_ms < 100
