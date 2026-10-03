# -*- coding: utf-8 -*-
"""
SAVI Agent API — inferencia en Python puro (sin numpy/xgboost/torch).

Reproduce EXACTAMENTE, a partir de los artefactos JSON de `rag/` y `serving/`:
  - AVM (XGBoost): recorrido de árboles con comparación en float32 (como hace el booster).
  - K-Means: estandarización + distancia euclídea a los 6 centroides (estado discreto del MDP).
  - Estado continuo del DQN: top-20 features del AVM + 3 señales, escalado MinMax con clip.
  - QNetwork (Double DQN): Linear → BatchNorm1d(eval) → ReLU → Linear → ReLU → Linear.
  - Regla de decisión: aplica `policy.json["selected_rule"]` igual que `savi_gpu_sagemaker.decision_rules`.

Y la traducción de entradas "humanas" (área habitable, grado, condición, estilo, etc.) al vector
de 27 NUM_FEATURES que usa `savi_cpu_pipeline.design_matrix`.
"""
from __future__ import annotations

import math
import re
import struct
from typing import Any

from .store import ACTIONS, NUM_FEATURES, Store, _BSMT, _COND

__all__ = [
    "f32", "xgb_predict", "kmeans_assign", "build_design_row", "dqn_state_vector",
    "mlp_forward", "decide", "features_from_overrides", "value_property_core",
]


# ════════════════════════════════════════════════════════════════════
# float32 helpers (igual que el booster de XGBoost al comparar splits)
# ════════════════════════════════════════════════════════════════════
def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", float(x)))[0]


# ════════════════════════════════════════════════════════════════════
# XGBoost puro-Python
# ════════════════════════════════════════════════════════════════════
def _flatten_tree(node: dict, out: dict[int, dict]) -> None:
    out[node["nodeid"]] = node
    for ch in node.get("children", ()):
        _flatten_tree(ch, out)


def _eval_tree_flat(nodes: dict[int, dict], row: dict[str, float]) -> float:
    node = nodes[0]
    while "leaf" not in node:
        feat = node["split"]
        val = row.get(feat)
        cond = node["split_condition"]
        if val is None:
            branch = "missing"
        else:
            branch = "yes" if f32(val) < f32(cond) else "no"
        node = nodes[node[branch]]
    return node["leaf"]


def xgb_predict(avm_model: dict, row: dict[str, float]) -> float:
    """expm1(base_score + Σ hoja) sobre todos los árboles (target = log1p(price_today))."""
    total = avm_model["base_score"]
    for tree in avm_model["trees"]:
        nodes: dict[int, dict] = {}
        _flatten_tree(tree, nodes)
        total += _eval_tree_flat(nodes, row)
    return math.expm1(total)


def _xgb_predict_cached(avm_model: dict, flat_trees: list[dict[int, dict]], row: dict[str, float]) -> float:
    total = avm_model["base_score"]
    for nodes in flat_trees:
        total += _eval_tree_flat(nodes, row)
    return math.expm1(total)


def flatten_trees(avm_model: dict) -> list[dict[int, dict]]:
    """Pre-aplana todos los árboles una vez (acelera evaluaciones repetidas, p.ej. en tests)."""
    flat = []
    for tree in avm_model["trees"]:
        nodes: dict[int, dict] = {}
        _flatten_tree(tree, nodes)
        flat.append(nodes)
    return flat


# ════════════════════════════════════════════════════════════════════
# K-Means
# ════════════════════════════════════════════════════════════════════
def kmeans_assign(kmeans_json: dict, features: list[float]) -> tuple[int, float]:
    """Estandariza `features` (orden = kmeans_json['features']) y devuelve (cluster, distancia_min)."""
    mean, scale, centroids = kmeans_json["mean"], kmeans_json["scale"], kmeans_json["centroids"]
    z = [(v - mean[i]) / scale[i] if scale[i] else 0.0 for i, v in enumerate(features)]
    best_c, best_d = -1, math.inf
    for c, centroid in enumerate(centroids):
        d = math.sqrt(sum((z[i] - centroid[i]) ** 2 for i in range(len(z))))
        if d < best_d:
            best_c, best_d = c, d
    return best_c, best_d


# ════════════════════════════════════════════════════════════════════
# Fila de diseño del AVM (27 NUM_FEATURES + area_<map_area> one-hot + cluster)
# ════════════════════════════════════════════════════════════════════
def build_design_row(features: list[float], map_area: str, cluster: int, avm_areas: list[str]) -> dict[str, float]:
    row = {name: features[i] for i, name in enumerate(NUM_FEATURES)}
    for a in avm_areas:
        row[f"area_{a}"] = 1.0 if a == map_area else 0.0
    row["cluster"] = float(cluster)
    return row


# ════════════════════════════════════════════════════════════════════
# Estado continuo del DQN
# ════════════════════════════════════════════════════════════════════
def dqn_state_vector(dqn_state_def: dict, design_row: dict[str, float], avm_value: float,
                     assessed_today: float, kmeans_distance: float) -> list[float]:
    top = dqn_state_def["top_features"]
    vals = [design_row.get(f, 0.0) for f in top]
    gap = (avm_value - assessed_today) / assessed_today if assessed_today else 0.0
    gap = max(-1.0, min(1.0, gap))
    vals += [math.log1p(avm_value), gap, kmeans_distance]
    mins, ranges = dqn_state_def["min"], dqn_state_def["range"]
    out = []
    for i, v in enumerate(vals):
        r = ranges[i]
        x = (v - mins[i]) / r if r else 0.0
        out.append(max(0.0, min(1.0, x)))
    return out


# ════════════════════════════════════════════════════════════════════
# QNetwork (Linear → BN eval → ReLU → Linear → ReLU → Linear)
# ════════════════════════════════════════════════════════════════════
def _linear(x: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
    return [sum(w * xv for w, xv in zip(row, x)) + b for row, b in zip(weight, bias)]


def _bn_eval(x: list[float], weight: list[float], bias: list[float], mean: list[float],
            var: list[float], eps: float) -> list[float]:
    return [(x[i] - mean[i]) / math.sqrt(var[i] + eps) * weight[i] + bias[i] for i in range(len(x))]


def _relu(x: list[float]) -> list[float]:
    return [max(0.0, v) for v in x]


def mlp_forward(dqn_weights: dict, state: list[float]) -> list[float]:
    sd, eps = dqn_weights["state_dict"], dqn_weights["bn_eps"]
    h = _linear(state, sd["net.0.weight"], sd["net.0.bias"])
    h = _bn_eval(h, sd["net.1.weight"], sd["net.1.bias"], sd["net.1.running_mean"], sd["net.1.running_var"], eps)
    h = _relu(h)
    h = _linear(h, sd["net.4.weight"], sd["net.4.bias"])
    h = _relu(h)
    q = _linear(h, sd["net.6.weight"], sd["net.6.bias"])
    return q


# ════════════════════════════════════════════════════════════════════
# Regla de decisión (igual que savi_gpu_sagemaker.decision_rules / RAG_CONTRACT.md)
# ════════════════════════════════════════════════════════════════════
def _margin(q: list[float]) -> float:
    s = sorted(q)
    return s[-1] - s[-2]


def _consensus_property(q: list[float], cluster: int, policy_vi: dict, policy_ql: dict) -> str:
    a_dqn = ACTIONS[max(range(len(q)), key=lambda i: q[i])]
    votes = [policy_vi[str(cluster)], policy_ql[str(cluster)], a_dqn]
    counts = {a: votes.count(a) for a in ACTIONS}
    top = max(counts.values())
    candidates = [a for a, v in counts.items() if v == top]
    return a_dqn if a_dqn in candidates else candidates[0]


def decide(policy: dict, cluster: int, q: list[float]) -> dict[str, Any]:
    """Aplica `policy['selected_rule']` sobre (cluster, q) → {"final","dqn","consensus","margin"}."""
    rule = policy["selected_rule"]
    cl = str(cluster)
    a_dqn = ACTIONS[max(range(len(q)), key=lambda i: q[i])]
    consensus_state = policy["policy_final"].get(cl, a_dqn)
    margin = _margin(q)

    by_state = {
        "vi": policy["policy_vi"].get(cl),
        "ql": policy["policy_ql"].get(cl),
        "dqn_state": policy["policy_dqn"].get(cl),
        "consensus_state": consensus_state,
    }
    if rule in by_state:
        final = by_state[rule]
    elif rule == "dqn_property":
        final = a_dqn
    elif rule == "consensus_property":
        final = _consensus_property(q, cluster, policy["policy_vi"], policy["policy_ql"])
    elif rule.startswith("gated_q"):
        tau = policy.get("gated_tau")
        if tau is None:
            tau = policy.get("taus", {}).get(rule, math.inf)
        cons_p = _consensus_property(q, cluster, policy["policy_vi"], policy["policy_ql"])
        final = a_dqn if margin >= tau else cons_p
    else:  # regla desconocida → el consenso por predio es la opción más conservadora
        final = _consensus_property(q, cluster, policy["policy_vi"], policy["policy_ql"])

    return {"final": final, "dqn": a_dqn, "consensus": consensus_state,
            "margin": round(margin, 3), "rule": rule}


# ════════════════════════════════════════════════════════════════════
# Construcción de features "what-if" a partir de entradas humanas
# ════════════════════════════════════════════════════════════════════
_STYLE_RE = re.compile(r"^(\d)(?: (\d)/(\d))?")
_GRADE_RE = re.compile(r"^(\d)([+-]\d+)?")


def _parse_style(style: str) -> dict[str, float]:
    s = style or ""
    m = _STYLE_RE.match(s.strip())
    stories = 1.0
    if m:
        base = float(m.group(1))
        frac = (float(m.group(2)) / float(m.group(3))) if m.group(2) and m.group(3) else 0.0
        stories = base + frac
    low = s.lower()
    return {
        "stories": stories,
        "is_split": 1.0 if "split" in low else 0.0,
        "is_condo": 1.0 if "condo" in low else 0.0,
        "is_townhouse": 1.0 if "townhouse" in low else 0.0,
        "is_brick": 1.0 if "brick" in low else 0.0,
    }


def _parse_grade(grade: str) -> tuple[float, float]:
    m = _GRADE_RE.match(str(grade).strip())
    if not m:
        return 4.0, 0.0
    base = float(m.group(1))
    mod = float(m.group(2)) if m.group(2) else 0.0
    return base, mod


def _parse_condition(value: Any) -> float:
    if value is None:
        return 4.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(_COND.get(str(value), 4))


def _parse_garage(garage_type: str | None) -> dict[str, float]:
    gt = (garage_type or "").strip()
    has = 1.0 if gt else 0.0
    low = gt.lower()
    return {
        "has_garage": has,
        "garage_attached": 1.0 if (low.startswith("att") or low.startswith("carport att")) else 0.0,
        "garage_carport": 1.0 if "carport" in low else 0.0,
    }


def _parse_basement(basement_type: str | None, finished: Any) -> dict[str, float]:
    frac = _BSMT.get((basement_type or "").strip(), 0.0)
    fin = 1.0 if (isinstance(finished, str) and finished.strip().lower() == "yes") else (1.0 if finished is True else 0.0)
    return {"bsmt_frac": frac, "bsmt_finished": fin}


def features_from_overrides(store: Store, overrides: dict[str, Any],
                            base_pid: str | None = None) -> tuple[list[float], str | None, dict]:
    """
    Construye el vector de 27 NUM_FEATURES (+ map_area) a partir de:
      - una parcela base (`base_pid`), si se da, y/o
      - overrides legibles por humanos (living_area, year_built, beds_above, grade, condition,
        style, basement_type/basement_finished, garage_type, fireplaces, plumbing, land_value,
        neighborhood/subdivision/map_area, ...).
    Devuelve (features[27], map_area, meta) donde meta trae detalles de la derivación.
    """
    base_parcel = store.parcels_by_pid.get(base_pid) if base_pid else None
    if base_parcel and base_parcel.get("features"):
        values = dict(zip(NUM_FEATURES, base_parcel["features"]))
        map_area = base_parcel.get("map_area")
    else:
        values = dict(store.median_features)
        map_area = None

    meta: dict[str, Any] = {"base_pid": base_pid, "used_median_defaults": base_parcel is None}

    # alias legibles → nombre de NUM_FEATURES
    direct_aliases = {
        "living_area": "living_area", "year_built": "year_built",
        "beds_above": "beds_above", "beds": "beds_above", "beds_below": "beds_below",
        "rooms_above": "rooms_above", "rooms": "rooms_above", "rooms_below": "rooms_below",
        "n_fireplaces": "n_fireplaces", "fireplaces": "n_fireplaces",
        "n_plumbing": "n_plumbing", "plumbing": "n_plumbing",
        "n_garages": "n_garages", "n_additions": "n_additions", "n_porches": "n_porches",
        "n_decks": "n_decks", "attic_finished": "attic_finished",
        "grade_base": "grade_base", "grade_mod": "grade_mod",
    }
    for k, target in direct_aliases.items():
        if k in overrides and overrides[k] is not None:
            values[target] = float(overrides[k])

    if "grade" in overrides and overrides["grade"] is not None:
        base, mod = _parse_grade(overrides["grade"])
        values["grade_base"], values["grade_mod"] = base, mod

    if "condition" in overrides and overrides["condition"] is not None:
        values["condition"] = _parse_condition(overrides["condition"])

    if "style" in overrides and overrides["style"] is not None:
        values.update(_parse_style(overrides["style"]))

    if "basement_type" in overrides or "basement_finished" in overrides:
        values.update(_parse_basement(overrides.get("basement_type"), overrides.get("basement_finished")))

    if "garage_type" in overrides and overrides["garage_type"] is not None:
        gvals = _parse_garage(overrides["garage_type"])
        values.update(gvals)
        if "n_garages" not in overrides:
            values["n_garages"] = 1.0 if gvals["has_garage"] else 0.0

    # Zona: neighborhood/subdivision/map_area → map_area (para el one-hot del AVM)
    resolved_area = store.map_area_for(
        map_area=overrides.get("map_area"), neighborhood=overrides.get("neighborhood"),
        subdivision=overrides.get("subdivision"))
    if resolved_area:
        map_area = resolved_area
    meta["map_area"] = map_area

    # Valor de la tierra → log_land_value (misma transformación to_today del pipeline)
    if overrides.get("land_value") is not None:
        land_2024 = float(overrides["land_value"])
        meta["land_value_source"] = "override"
    elif "neighborhood" in overrides or "map_area" in overrides or "subdivision" in overrides:
        land_2024 = store.median_land_value_2024(map_area)
        meta["land_value_source"] = "area_median"
    elif base_parcel is not None:
        land_2024 = None  # ya viene en `values['log_land_value']` desde la parcela base
        meta["land_value_source"] = "base_parcel"
    else:
        land_2024 = store.median_land_value_2024(map_area)
        meta["land_value_source"] = "global_median"
    if land_2024 is not None:
        land_today = land_2024 * store.hpi_ratio
        values["log_land_value"] = math.log1p(land_today)
        meta["land_value_2024"] = land_2024
        meta["land_value_today"] = land_today

    features = [float(values[f]) for f in NUM_FEATURES]
    return features, map_area, meta


def value_property_core(store: Store, *, pid: str | None = None, overrides: dict[str, Any] | None = None,
                        flat_trees: list[dict[int, dict]] | None = None) -> dict[str, Any]:
    """Valuación AVM + estado DQN + Q-values + decisión, para una parcela existente o hipotética."""
    overrides = overrides or {}
    if store.avm_model is None:
        return {"error": "No hay avm_model.json (AVM no era XGBoost en esta corrida)"}

    features, map_area, meta = features_from_overrides(store, overrides, base_pid=pid)
    assumptions = []
    if map_area is None:
        # Sin ubicación: el map_area con más parcelas del padrón (y se declara como supuesto)
        counts: dict[str, int] = {}
        for p in store.parcels:
            counts[p["map_area"]] = counts.get(p["map_area"], 0) + 1
        map_area = max(counts, key=counts.get) if counts else "0000"
        assumptions.append(f"sin ubicación: se asumió el map_area más frecuente ({map_area}); "
                           "indique neighborhood o subdivision para una valuación local")

    base_parcel = store.parcels_by_pid.get(pid) if pid else None
    assessed_today = overrides.get("assessed_today")
    if assessed_today is None:
        assessed_today = base_parcel["assessed_today"] if base_parcel else None

    cluster, kdist = kmeans_assign(store.kmeans, features)
    design_row = build_design_row(features, map_area, cluster, store.avm_areas)

    if flat_trees is not None:
        avm_value = _xgb_predict_cached(store.avm_model, flat_trees, design_row)
    else:
        avm_value = xgb_predict(store.avm_model, design_row)

    if assessed_today is None:
        # Casa nueva sin avalúo oficial: brecha AVM-vs-avalúo neutra (0) en el estado del DQN
        assessed_today = avm_value
        assumptions.append("sin avalúo oficial: brecha AVM vs avalúo = 0 (neutra) para el agente")
    state = dqn_state_vector(store.dqn_state_def, design_row, avm_value, assessed_today, kdist)
    q = mlp_forward(store.dqn_weights, state)
    decision = decide(store.policy, cluster, q)

    mape = store.manifest.get("avm", {}).get("mape", 0.0)
    return {
        "pid": pid, "map_area": map_area, "cluster": cluster, "kmeans_distance": round(kdist, 4),
        "avm_value_today": round(avm_value, 0),
        "avm_range_low": round(avm_value * (1 - mape), 0), "avm_range_high": round(avm_value * (1 + mape), 0),
        "assessed_today": round(assessed_today, 0) if assessed_today else None,
        "gap_vs_assessed": round((avm_value - assessed_today) / assessed_today, 4) if assessed_today else None,
        "q_values": {a: round(v, 3) for a, v in zip(ACTIONS, q)},
        "decision": decision, "features_meta": meta, "assumptions": assumptions,
    }
