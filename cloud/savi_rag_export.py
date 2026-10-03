# -*- coding: utf-8 -*-
"""
SAVI v3 · Exportación de la BASE DE CONOCIMIENTO del agente RAG (corre dentro del motor CPU).

Convierte todo lo que el pipeline ya calculó + las fuentes reales en artefactos JSON
autocontenidos que la API (Lambda, sólo Python estándar) carga en memoria:

  rag/parcels.json.gz       18k parcelas del padrón 2024: características legibles, avalúo,
                            valor AVM hoy, brecha, cluster, barrio, coordenadas, vector de features
  rag/sales.json.gz         2,930 ventas reales De Cock: atributos clave, precio nominal y en $ hoy,
                            lat/lon, barrio, y (si entraron al entrenamiento) error OOF del AVM y split
  rag/market.json           FHFA HPI trimestral + Zillow ZHVI mensual + variaciones
  rag/areas.json            estadísticas por barrio (ventas) y por subdivisión (padrón)
  rag/avm_model.json        árboles XGBoost (dump JSON) + orden de columnas → valuación "what-if"
  rag/kmeans.json           scaler + centroides → estado discreto del MDP para casas nuevas
  rag/dqn_state.json        definición del estado continuo del DQN (top features + MinMax)
  rag/manifest.json         versión, conteos y descripción de cada archivo

El contrato de estos archivos está en RAG_CONTRACT.md.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

import utils as U

log = U.get_logger("savi.rag")

GEO_FILE = "ames_decock_geo.csv"
ZILLOW_FILE = "zillow_zhvi_ames.csv"
RAG_VERSION = "1.0"

# Atributos De Cock que se exponen en las ventas (nombre limpio → columna sin espacios)
_SALE_ATTRS = {
    "gr_liv_area": "GrLivArea", "year_built": "YearBuilt", "year_remod": "YearRemodAdd",
    "overall_qual": "OverallQual", "overall_cond": "OverallCond", "bedrooms": "BedroomAbvGr",
    "full_bath": "FullBath", "half_bath": "HalfBath", "garage_cars": "GarageCars",
    "total_bsmt_sf": "TotalBsmtSF", "lot_area": "LotArea", "house_style": "HouseStyle",
    "bldg_type": "BldgType", "kitchen_qual": "KitchenQual", "exter_qual": "ExterQual",
    "fireplaces": "Fireplaces", "sale_condition": "SaleCondition", "sale_type": "SaleType",
}


def _num(v):
    """JSON-safe: NaN → None, numpy → Python, floats a 4 decimales."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _num(v) for k, v in row.items()} for row in df.to_dict("records")]


def _dump(obj, path: Path, gz: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_num).encode("utf-8")
    if gz:
        path.write_bytes(gzip.compress(data, compresslevel=9))
    else:
        path.write_bytes(data)
    return path.stat().st_size


# ════════════════════════════════════════════════════════════════════
# MODELOS → JSON (inferencia en Python puro en la Lambda)
# ════════════════════════════════════════════════════════════════════
def export_xgb(model, columns: list[str]) -> dict:
    """Dump de árboles XGBoost + base_score (margen) para evaluar sin la librería."""
    booster = model.get_booster()
    cfg = json.loads(booster.save_config())
    base = cfg["learner"]["learner_model_param"]["base_score"]
    base = float(str(base).strip("[]").split(",")[0])
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json")]
    return {"type": "xgboost", "objective": cfg["learner"]["objective"]["name"],
            "base_score": base, "target": "log1p(price_today)", "columns": columns,
            "feature_index": {c: i for i, c in enumerate(columns)}, "trees": trees}


def export_kmeans(scaler, km, features: list[str]) -> dict:
    return {"features": features, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "centroids": km.cluster_centers_.tolist()}


def export_dqn_state(mms, top: list[str]) -> dict:
    return {"top_features": top, "extra_signals": ["log_avm_value", "gap_vs_assessed", "kmeans_distance"],
            "min": mms.data_min_.tolist(), "range": mms.data_range_.tolist(), "clip": [0.0, 1.0]}


# ════════════════════════════════════════════════════════════════════
# CONOCIMIENTO
# ════════════════════════════════════════════════════════════════════
def _parcel_geo(portfolio: pd.DataFrame, geo: pd.DataFrame | None) -> pd.DataFrame:
    """
    Coordenadas por parcela: exacta si la parcela se vendió en De Cock; si no, el centroide de
    las ventas geolocalizadas de su subdivisión; si no, el de su map_area (geo_precision lo indica).
    También propaga el nombre de barrio De Cock por PID → subdivisión → map_area.
    """
    out = portfolio[["PID", "subdivision", "map_area"]].copy()
    if geo is None:
        out[["lat", "lon", "neighborhood"]] = None
        out["geo_precision"] = "none"
        return out
    g = geo.merge(portfolio[["PID", "subdivision", "map_area"]], on="PID", how="inner")
    sub_c = g.groupby("subdivision")[["lat", "lon"]].mean()
    area_c = g.groupby("map_area")[["lat", "lon"]].mean()
    sub_n = g.groupby("subdivision")["neighborhood"].agg(lambda s: s.mode().iat[0])
    area_n = g.groupby("map_area")["neighborhood"].agg(lambda s: s.mode().iat[0])
    exact = geo.drop_duplicates("PID").set_index("PID")

    lat, lon, prec, nb = [], [], [], []
    for pid, sub, area in out[["PID", "subdivision", "map_area"]].itertuples(index=False):
        if pid in exact.index:
            r = exact.loc[pid]
            lat.append(r.lat), lon.append(r.lon), prec.append("exact"), nb.append(r.neighborhood)
        elif sub in sub_c.index:
            r = sub_c.loc[sub]
            lat.append(r.lat), lon.append(r.lon), prec.append("subdivision"), nb.append(sub_n.get(sub))
        elif area in area_c.index:
            r = area_c.loc[area]
            lat.append(r.lat), lon.append(r.lon), prec.append("map_area"), nb.append(area_n.get(area))
        else:
            lat.append(None), lon.append(None), prec.append("none"), nb.append(None)
    out["lat"], out["lon"], out["geo_precision"], out["neighborhood"] = lat, lon, prec, nb
    return out


def build_parcels(portfolio: pd.DataFrame, port_pred, port_cl, geo, num_features) -> pd.DataFrame:
    gp = _parcel_geo(portfolio, geo)
    p = pd.DataFrame({
        "pid": portfolio["PID"], "subdivision": portfolio["subdivision"], "map_area": portfolio["map_area"],
        "neighborhood": gp["neighborhood"], "lat": gp["lat"], "lon": gp["lon"],
        "geo_precision": gp["geo_precision"], "classification": portfolio["classification"],
        "style": portfolio["MAIN STYLE"], "grade": portfolio["GRADE"].astype(str),
        "condition_label": portfolio["CONDITION"], "basement_type": portfolio["BASEMENT TYPE"],
        "garage_type": portfolio["GARAGE TYPE"],
        "year_built": portfolio["year_built"], "living_area": portfolio["living_area"],
        "rooms_above": portfolio["rooms_above"], "beds_above": portfolio["beds_above"],
        "beds_below": portfolio["beds_below"], "n_plumbing": portfolio["n_plumbing"],
        "n_fireplaces": portfolio["n_fireplaces"],
        "land_value_2024": portfolio["land_value_2024"], "assessed_2024": portfolio["assessed_2024"],
        "assessed_today": portfolio["assessed_today"].round(0), "avm_value_today": np.round(port_pred, 0),
        "gap_vs_assessed": np.round((port_pred - portfolio["assessed_today"]) / portfolio["assessed_today"], 4),
        "cluster": port_cl,
    })
    p["price_per_sqft_today"] = (p["avm_value_today"] / p["living_area"]).round(1)
    recs = _records(p)
    feats = portfolio[num_features].astype(float).values
    for r, f in zip(recs, feats):
        # Precisión completa (sin redondeo): los árboles cortan en umbrales float32 muy finos
        r["features"] = [float(x) if np.isfinite(x) else None for x in f]  # orden = kmeans.json["features"]
    return recs


def build_sales(decock_path: Path, train: pd.DataFrame, oof_pred, errors, train_cl, split,
                geo, hpi, latest) -> list[dict]:
    sep = "\t" if decock_path.suffix.lower() == ".txt" else ","
    d = pd.read_csv(decock_path, sep=sep, dtype={"PID": str}, low_memory=False)
    d.columns = [c.replace(" ", "").replace("/", "") for c in d.columns]
    d["PID"] = d["PID"].astype(str).str.zfill(10)
    s = pd.DataFrame({"pid": d["PID"], "yr_sold": d["YrSold"], "mo_sold": d["MoSold"],
                      "price": d["SalePrice"].astype(float)})
    for k, c in _SALE_ATTRS.items():
        s[k] = d[c] if c in d.columns else None
    q = ((s["mo_sold"] - 1) // 3 + 1).astype(int)
    base = np.array([hpi[(int(y), int(qq))] for y, qq in zip(s["yr_sold"], q)])
    s["price_today"] = (s["price"] * hpi[latest] / base).round(0)
    s["price_per_sqft_today"] = (s["price_today"] / s["gr_liv_area"]).round(1)
    if geo is not None:
        s = s.merge(geo.drop_duplicates("PID").rename(columns={"PID": "pid"}), on="pid", how="left")
    # Ventas que entraron al entrenamiento: error OOF, cluster y split (para auditoría del agente)
    t = pd.DataFrame({"pid": train["PID"].values, "sale_date": train["sale_date"].values,
                      "avm_oof": np.round(oof_pred, 0), "avm_error": np.round(errors, 4),
                      "cluster": train_cl, "split": np.array(["train", "val", "test"])[split]})
    s["sale_date"] = s["yr_sold"] * 100 + s["mo_sold"]
    s = s.merge(t, on=["pid", "sale_date"], how="left")
    s["in_training"] = s["avm_oof"].notna()
    return _records(s)


def build_market(hpi_path: Path, zillow_path: Path | None) -> dict:
    h = pd.read_csv(hpi_path)
    h["hpi"] = pd.to_numeric(h["hpi"], errors="coerce")
    h = h.dropna(subset=["hpi"])
    h = h[h["year"] >= 2000]
    series = [{"period": f"{int(r.year)}Q{int(r.qtr)}", "hpi": round(float(r.hpi), 2)} for r in h.itertuples()]
    last = h.iloc[-1]
    yoy = h[(h.year == last.year - 1) & (h.qtr == last.qtr)]
    out = {"fhfa_hpi": {"source": "FHFA All-Transactions HPI, Ames MSA (CBSA 11180)", "series": series,
                        "latest": series[-1],
                        "yoy_pct": round(100 * (last.hpi / yoy.hpi.iat[0] - 1), 2) if len(yoy) else None,
                        "since_2010_pct": round(100 * (last.hpi / h[h.year == 2010].hpi.mean() - 1), 1)}}
    if zillow_path is not None and zillow_path.exists():
        z = pd.read_csv(zillow_path)
        months = [c for c in z.columns if c[:2] in ("19", "20")]
        vals = z[months].iloc[0]
        zs = [{"month": m[:7], "zhvi": round(float(v), 0)} for m, v in vals.items() if pd.notna(v)]
        out["zillow_zhvi"] = {"source": "Zillow Home Value Index (city of Ames, IA, mid-tier, SA)",
                              "series": [x for x in zs if x["month"] >= "2010-01"], "latest": zs[-1],
                              "yoy_pct": round(100 * (zs[-1]["zhvi"] / zs[-13]["zhvi"] - 1), 2) if len(zs) > 13 else None}
    return out


def build_areas(parcels: list[dict], sales: list[dict]) -> dict:
    p, s = pd.DataFrame(parcels), pd.DataFrame(sales)
    nb = (s.dropna(subset=["neighborhood"]).groupby("neighborhood")
          .agg(n_sales=("pid", "size"), median_price_today=("price_today", "median"),
               median_ppsf_today=("price_per_sqft_today", "median"), median_year_built=("year_built", "median"),
               median_living_area=("gr_liv_area", "median"), lat=("lat", "mean"), lon=("lon", "mean"))
          .reset_index())  # _num redondea a 4 decimales (lat/lon ≈ 11 m)
    pn = (p.dropna(subset=["neighborhood"]).groupby("neighborhood")
          .agg(n_parcels=("pid", "size"), median_avm_today=("avm_value_today", "median"),
               median_assessed_2024=("assessed_2024", "median")).round(0).reset_index())
    nb = nb.merge(pn, on="neighborhood", how="outer")
    sub = (p.groupby("subdivision")
           .agg(n_parcels=("pid", "size"), neighborhood=("neighborhood", lambda x: x.mode().iat[0] if x.notna().any() else None),
                median_avm_today=("avm_value_today", "median"), median_assessed_2024=("assessed_2024", "median"),
                median_year_built=("year_built", "median"), median_living_area=("living_area", "median"),
                lat=("lat", "mean"), lon=("lon", "mean"))
           .reset_index())
    cl = (p.groupby("cluster")
          .agg(n_parcels=("pid", "size"), median_avm_today=("avm_value_today", "median"),
               median_living_area=("living_area", "median"), median_year_built=("year_built", "median"),
               share_condo=("style", lambda x: float(x.fillna("").str.contains("Condo").mean())))
          .round(3).reset_index())
    return {"neighborhoods": _records(nb), "subdivisions": _records(sub), "clusters": _records(cl)}


def export_rag(out: Path, *, reference_fetch, decock_path: Path, train: pd.DataFrame, portfolio: pd.DataFrame,
               port_pred, port_cl, oof_pred, errors, train_cl, split, hpi, latest, hpi_path: Path,
               avm, avm_name: str, avm_columns: list[str], scaler, km, mms, top: list[str],
               num_features: list[str], metrics: dict) -> dict:
    """Escribe out/rag/*. `reference_fetch(name)` devuelve la ruta local de un archivo opcional o None."""
    rag = out / "rag"
    geo_path, z_path = reference_fetch(GEO_FILE), reference_fetch(ZILLOW_FILE)
    geo = pd.read_csv(geo_path, dtype={"PID": str}) if geo_path else None
    if geo is not None:
        geo["PID"] = geo["PID"].str.zfill(10)

    parcels = build_parcels(portfolio, port_pred, port_cl, geo, num_features)
    sales = build_sales(decock_path, train, oof_pred, errors, train_cl, split, geo, hpi, latest)
    sizes = {
        "parcels.json.gz": _dump(parcels, rag / "parcels.json.gz", gz=True),
        "sales.json.gz": _dump(sales, rag / "sales.json.gz", gz=True),
        "market.json": _dump(build_market(hpi_path, z_path), rag / "market.json"),
        "areas.json": _dump(build_areas(parcels, sales), rag / "areas.json"),
        "kmeans.json": _dump(export_kmeans(scaler, km, num_features), rag / "kmeans.json"),
        "dqn_state.json": _dump(export_dqn_state(mms, top), rag / "dqn_state.json"),
    }
    if avm_name == "XGBoost":
        sizes["avm_model.json"] = _dump(export_xgb(avm, avm_columns), rag / "avm_model.json", gz=False)
    else:
        log.warning("AVM seleccionado=%s: la valuación what-if de la API requiere XGBoost", avm_name)
    manifest = {
        "version": RAG_VERSION, "run_id": metrics.get("run_id"), "hpi_latest": metrics.get("hpi_latest"),
        "counts": {"parcels": len(parcels), "sales": len(sales),
                   "sales_in_training": int(sum(1 for s in sales if s.get("in_training"))),
                   "parcels_geo_exact": int(sum(1 for p in parcels if p["geo_precision"] == "exact"))},
        "files": sizes, "avm": {k: metrics["supervised"][avm_name][k] for k in ("r2_log", "r2_usd", "mae_usd", "mape")},
        "kmeans_silhouette": metrics.get("kmeans_silhouette"),
    }
    _dump(manifest, rag / "manifest.json")
    log.info("RAG exportado: %s", {k: f"{v / 1e6:.2f} MB" for k, v in sizes.items()})
    return manifest
