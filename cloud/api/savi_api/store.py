# -*- coding: utf-8 -*-
"""
SAVI Agent API — carga de artefactos y construcción de índices en memoria.

Fuente de los artefactos (ver RAG_CONTRACT.md):
  - Local: env `SAVI_DATA_DIR` con subcarpetas `rag/` y `serving/`.
  - S3: `s3://$SAVI_PROCESSED_BUCKET/serving/latest.json` apunta a `rag_prefix`/`serving_prefix`
    dentro del mismo bucket; se descargan a `/tmp/savi_cache/<run_id>/` (boto3 se importa
    perezosamente: en Lambda está disponible en el runtime, en local/tests no hace falta).

Todo el módulo es stdlib puro (json, gzip, pathlib, unicodedata, statistics, math).
Construye el `Store` UNA sola vez por proceso (`get_store()` cachea en memoria).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import statistics
import unicodedata
from pathlib import Path
from typing import Any

# ════════════════════════════════════════════════════════════════════
# Constantes del pipeline (deben coincidir EXACTO con savi_cpu_pipeline.NUM_FEATURES)
# ════════════════════════════════════════════════════════════════════
NUM_FEATURES = [
    "year_built", "living_area", "rooms_above", "rooms_below", "beds_above", "beds_below",
    "n_additions", "n_porches", "n_decks", "n_plumbing", "n_fireplaces", "n_garages",
    "stories", "is_split", "is_condo", "is_townhouse", "is_brick", "grade_base", "grade_mod",
    "condition", "bsmt_frac", "bsmt_finished", "attic_finished", "has_garage",
    "garage_attached", "garage_carport", "log_land_value",
]

ROLL_YEAR, ROLL_QTR = 2024, 1
ACTIONS = ["APROBAR", "REVISAR", "RECHAZAR"]

_COND = {"Very Poor": 0, "Poor": 1, "Fair": 2, "Below Normal": 3, "Normal": 4,
         "Above Normal": 5, "Good": 6, "Very Good": 7, "Excellent": 8}
_COND_REV = {v: k for k, v in _COND.items()}
_BSMT = {"Full": 1.0, "3/4": 0.75, "1/2": 0.5, "1/4": 0.25, "Bsmt SF (Obsv)": 0.5}

# Códigos cortos De Cock (data_description.txt) → nombre de barrio completo usado en
# rag/sales.json.gz y rag/areas.json (ya expandidos por el pipeline). Alias case-insensitive.
NEIGHBORHOOD_ALIASES = {
    "blmngtn": "Bloomington Heights", "blueste": "Blueste", "brdale": "Briardale",
    "brkside": "Brookside", "clearcr": "Clear Creek", "collgcr": "College Creek",
    "crawfor": "Crawford", "edwards": "Edwards", "gilbert": "Gilbert",
    "idotrr": "Iowa DOT and Rail Road", "meadowv": "Meadow Village", "mitchel": "Mitchell",
    "names": "North Ames", "noridge": "Northridge", "npkvill": "Northpark Villa",
    "nridght": "Northridge Heights", "nwames": "Northwest Ames",
    "oldtown": "Old Town", "swisu": "South and West of Iowa State University",
    "sawyer": "Sawyer", "sawyerw": "Sawyer West", "somerst": "Somerset",
    "stonebr": "Stone Brook", "timber": "Timberland", "veenker": "Veenker",
    "greens": "Greens", "grnhill": "Green Hills", "landmrk": "Landmark",
}


def normalize_name(s: str | None) -> str:
    """Minúsculas, sin acentos, sin guiones bajos: tolerante para comparar nombres de zona."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.replace("_", " ").replace("-", " ").strip().lower()
    return " ".join(s.split())


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


# ════════════════════════════════════════════════════════════════════
# Carga de artefactos: local o S3
# ════════════════════════════════════════════════════════════════════
def _resolve_dirs() -> tuple[Path, Path]:
    """Devuelve (rag_dir, serving_dir) ya disponibles localmente."""
    data_dir = os.environ.get("SAVI_DATA_DIR")
    if data_dir:
        base = Path(data_dir)
        return base / "rag", base / "serving"
    return _resolve_dirs_s3()


def _resolve_dirs_s3() -> tuple[Path, Path]:
    """Descarga rag/+serving/ desde S3 (sólo cuando no hay SAVI_DATA_DIR) a /tmp/savi_cache."""
    import boto3  # import perezoso: sólo existe en el runtime de Lambda

    bucket = os.environ["SAVI_PROCESSED_BUCKET"]
    s3 = boto3.client("s3")
    latest = json.loads(s3.get_object(Bucket=bucket, Key="serving/latest.json")["Body"].read())
    run_id = latest["run_id"]
    cache = Path("/tmp/savi_cache") / run_id
    rag_dir, serving_dir = cache / "rag", cache / "serving"
    if rag_dir.exists() and serving_dir.exists():
        return rag_dir, serving_dir
    rag_dir.mkdir(parents=True, exist_ok=True)
    serving_dir.mkdir(parents=True, exist_ok=True)
    for prefix, dest in ((latest["rag_prefix"], rag_dir), (latest["serving_prefix"], serving_dir)):
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                name = key.rsplit("/", 1)[-1]
                if not name:
                    continue
                s3.download_file(bucket, key, str(dest / name))
    return rag_dir, serving_dir


# ════════════════════════════════════════════════════════════════════
# Store
# ════════════════════════════════════════════════════════════════════
class Store:
    """Artefactos + índices construidos una sola vez por proceso."""

    def __init__(self, rag_dir: Path, serving_dir: Path):
        self.rag_dir, self.serving_dir = rag_dir, serving_dir

        self.manifest: dict = _read_json(rag_dir / "manifest.json")
        self.parcels: list[dict] = _read_json(rag_dir / "parcels.json.gz")
        self.sales: list[dict] = _read_json(rag_dir / "sales.json.gz")
        self.market: dict = _read_json(rag_dir / "market.json")
        self.areas: dict = _read_json(rag_dir / "areas.json")
        self.kmeans: dict = _read_json(rag_dir / "kmeans.json")
        self.dqn_state_def: dict = _read_json(rag_dir / "dqn_state.json")
        avm_path = rag_dir / "avm_model.json"
        self.avm_model: dict | None = _read_json(avm_path) if avm_path.exists() else None

        self.decisions: dict = _read_json(serving_dir / "decisions.json.gz")
        self.dqn_weights: dict = _read_json(serving_dir / "dqn_weights.json")
        self.policy: dict = _read_json(serving_dir / "policy.json")

        self._build_indexes()

    # ── índices ──
    def _build_indexes(self) -> None:
        self.parcels_by_pid = {p["pid"]: p for p in self.parcels}

        self.sales_by_pid: dict[str, list[dict]] = {}
        for s in self.sales:
            self.sales_by_pid.setdefault(s["pid"], []).append(s)

        self.neighborhoods_by_name = {normalize_name(n["neighborhood"]): n
                                      for n in self.areas.get("neighborhoods", []) if n.get("neighborhood")}
        self.subdivisions_by_name = {normalize_name(s["subdivision"]): s
                                     for s in self.areas.get("subdivisions", []) if s.get("subdivision")}
        self.clusters_by_id = {int(c["cluster"]): c for c in self.areas.get("clusters", [])}

        # map_area <-> neighborhood/subdivision (moda) para construir features what-if
        area_nb_votes: dict[str, dict[str, int]] = {}
        area_land: dict[str, list[float]] = {}
        all_land: list[float] = []
        feat_cols: dict[str, list[float]] = {f: [] for f in NUM_FEATURES}
        ratios: list[float] = []
        sub_to_area: dict[str, str] = {}
        for p in self.parcels:
            ma = p.get("map_area")
            nb = p.get("neighborhood")
            sub = p.get("subdivision")
            lv = p.get("land_value_2024")
            if ma and nb:
                area_nb_votes.setdefault(ma, {}).setdefault(nb, 0)
                area_nb_votes[ma][nb] += 1
            if ma and sub and sub not in sub_to_area:
                sub_to_area[sub] = ma
            if ma and lv is not None:
                area_land.setdefault(ma, []).append(lv)
            if lv is not None:
                all_land.append(lv)
            feats = p.get("features")
            if feats and len(feats) == len(NUM_FEATURES):
                for name, val in zip(NUM_FEATURES, feats):
                    if val is not None:
                        feat_cols[name].append(val)
            a2024, atoday = p.get("assessed_2024"), p.get("assessed_today")
            if a2024 and atoday:
                ratios.append(atoday / a2024)

        self.area_to_neighborhood = {ma: max(votes, key=votes.get) for ma, votes in area_nb_votes.items()}
        self.neighborhood_to_area: dict[str, str] = {}
        for ma, votes in area_nb_votes.items():
            nb = max(votes, key=votes.get)
            key = normalize_name(nb)
            # conserva el área con más parcelas para ese barrio
            if key not in self.neighborhood_to_area or votes[nb] > area_nb_votes[self.neighborhood_to_area[key]].get(nb, 0):
                self.neighborhood_to_area[key] = ma
        self.subdivision_to_area = {normalize_name(k): v for k, v in sub_to_area.items()}
        self.subdivision_to_area_raw = sub_to_area

        self.median_land_value_by_area = {ma: statistics.median(vs) for ma, vs in area_land.items()}
        self.median_land_value_global = statistics.median(all_land) if all_land else 0.0
        self.median_features = {f: (statistics.median(v) if v else 0.0) for f, v in feat_cols.items()}
        # HPI 2024Q1 → hoy: constante para todas las parcelas (to_today usa el mismo trimestre base)
        self.hpi_ratio = statistics.median(ratios) if ratios else 1.0

        # áreas (columnas one-hot) y orden de columnas del AVM
        if self.avm_model:
            self.avm_columns: list[str] = self.avm_model["columns"]
            self.avm_areas: list[str] = [c[len("area_"):] for c in self.avm_columns if c.startswith("area_")]
        else:
            self.avm_columns, self.avm_areas = [], []

        # vectores estandarizados de comparables: ventas in_training con features de su parcela
        mean, scale = self.kmeans["mean"], self.kmeans["scale"]
        comps = []
        for s in self.sales:
            if not s.get("in_training"):
                continue
            p = self.parcels_by_pid.get(s["pid"])
            if not p or not p.get("features"):
                continue
            feats = p["features"]
            if len(feats) != len(mean):
                continue
            z = [(v - mean[i]) / scale[i] if v is not None and scale[i] else 0.0
                 for i, v in enumerate(feats)]
            comps.append({"sale": s, "z": z, "lat": s.get("lat"), "lon": s.get("lon"),
                         "neighborhood": s.get("neighborhood")})
        self.comp_vectors = comps

    # ── utilidades de resolución de zona ──
    def resolve_neighborhood(self, name: str) -> dict | None:
        key = normalize_name(name)
        alias = NEIGHBORHOOD_ALIASES.get(key.replace(" ", ""))
        if alias:
            key = normalize_name(alias)
        return self.neighborhoods_by_name.get(key)

    def resolve_subdivision(self, name: str) -> dict | None:
        return self.subdivisions_by_name.get(normalize_name(name))

    def map_area_for(self, *, map_area: str | None = None, neighborhood: str | None = None,
                     subdivision: str | None = None) -> str | None:
        if map_area:
            return str(map_area)
        if neighborhood:
            key = normalize_name(neighborhood)
            alias = NEIGHBORHOOD_ALIASES.get(key.replace(" ", ""))
            if alias:
                key = normalize_name(alias)
            if key in self.neighborhood_to_area:
                return self.neighborhood_to_area[key]
        if subdivision:
            key = normalize_name(subdivision)
            if key in self.subdivision_to_area:
                return self.subdivision_to_area[key]
        return None

    def median_land_value_2024(self, map_area: str | None) -> float:
        if map_area and map_area in self.median_land_value_by_area:
            return self.median_land_value_by_area[map_area]
        return self.median_land_value_global


# ════════════════════════════════════════════════════════════════════
# Caché a nivel de proceso
# ════════════════════════════════════════════════════════════════════
_STORE: Store | None = None


def get_store(force_reload: bool = False) -> Store:
    global _STORE
    if _STORE is None or force_reload:
        rag_dir, serving_dir = _resolve_dirs()
        _STORE = Store(rag_dir, serving_dir)
    return _STORE
