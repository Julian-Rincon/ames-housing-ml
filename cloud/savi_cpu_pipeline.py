# -*- coding: utf-8 -*-
"""
SAVI v2 · Cloud — MOTOR CPU (se ejecuta en EC2).

Flujo:
  1. Descarga de S3 (o lee en local) las fuentes REALES de Ames, Iowa:
       · input/      → ventas De Cock 2006-2010 (AmesHousing.txt, 2,930 ventas con PID)
       · reference/  → padrón del City Assessor 2024 (características actuales + avalúo)
                       índice FHFA HPI Ames MSA (trimestral) y, si existe,
                       ventas del assessor 2020-2022 (residential-sales.xlsx)
  2. Integra por número de parcela (PID) y lleva todos los precios a dólares de hoy (FHFA).
  3. K-Means (estados MDP) · XGBoost vs LightGBM con CV 5-fold (AVM, errores out-of-fold)
  4. MDP Value Iteration + Q-Learning tabular.
  5. Exporta modelos, tablas Q, políticas y tensores de estado continuo para el DQN.
  6. Escribe _SUCCESS.json AL FINAL → dispara la Lambda que lanza SageMaker.

Uso:
  Local : python savi_cpu_pipeline.py --input data/AmesHousing.txt --reference data/ --output out/
  EC2   : python savi_cpu_pipeline.py --input s3://raw/input/AmesHousing.txt \
                                      --reference s3://raw/reference/ --output s3://processed/runs/<id>/
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import savi_rag_export as RAG
import utils as U

log = U.get_logger("savi.cpu")

# Archivos de referencia esperados dentro de --reference
ASSESSOR_ROLL = "residential-properties-with-detail-2024.xlsx"
HPI_FILE = "fhfa_hpi_ames.csv"
ASSESSOR_SALES = "residential-sales.xlsx"   # opcional (ventas 2020-2022)
ROLL_YEAR, ROLL_QTR = 2024, 1               # el avalúo 2024 es al 1-ene-2024

N_DQN_TOP = 20                              # top features del AVM para el estado del DQN


# ════════════════════════════════════════════════════════════════════
# E/S: rutas locales o s3:// indistintamente
# ════════════════════════════════════════════════════════════════════
def _split_s3(uri: str) -> tuple[str, str]:
    m = re.match(r"s3://([^/]+)/?(.*)", uri)
    if not m:
        raise ValueError(f"URI S3 inválida: {uri}")
    return m.group(1), m.group(2)


def fetch(uri: str, workdir: Path) -> Path:
    """Devuelve una ruta local para `uri` (descargando de S3 si hace falta)."""
    if not uri.startswith("s3://"):
        return Path(uri)
    import boto3
    bucket, key = _split_s3(uri)
    dest = workdir / Path(key).name
    log.info("Descargando s3://%s/%s", bucket, key)
    boto3.client("s3").download_file(bucket, key, str(dest))
    return dest


def fetch_reference(ref: str, name: str, workdir: Path, required: bool = True) -> Path | None:
    """Busca un archivo de referencia en un directorio local o prefijo S3."""
    if ref.startswith("s3://"):
        import boto3
        from botocore.exceptions import ClientError
        bucket, prefix = _split_s3(ref)
        key = f"{prefix.rstrip('/')}/{name}".lstrip("/")
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
        except ClientError:
            if required:
                raise FileNotFoundError(f"Falta referencia obligatoria s3://{bucket}/{key}")
            log.warning("Referencia opcional no encontrada: s3://%s/%s", bucket, key)
            return None
        return fetch(f"s3://{bucket}/{key}", workdir)
    p = Path(ref) / name
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Falta referencia obligatoria {p}")
        log.warning("Referencia opcional no encontrada: %s", p)
        return None
    return p


def publish(local_dir: Path, output: str) -> None:
    """Copia el árbol de artefactos a `output` (dir local o prefijo S3). _SUCCESS va al final."""
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    success = [p for p in files if p.name == Path(U.ARTIFACTS["success"]).name]
    ordered = [p for p in files if p not in success] + success
    if output.startswith("s3://"):
        import boto3
        s3 = boto3.client("s3")
        bucket, prefix = _split_s3(output)
        for p in ordered:
            key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}".lstrip("/")
            s3.upload_file(str(p), bucket, key)
            log.info("  ↑ s3://%s/%s", bucket, key)
    else:
        out = Path(output)
        for p in ordered:
            dst = out / p.relative_to(local_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        log.info("Artefactos copiados a %s", out)


# ════════════════════════════════════════════════════════════════════
# PASO 1 — FUENTES REALES
# ════════════════════════════════════════════════════════════════════
def load_hpi(path: Path) -> tuple[dict, tuple[int, int]]:
    """FHFA All-Transactions HPI Ames MSA → {(año, trimestre): índice} y el último trimestre."""
    h = pd.read_csv(path)
    h["hpi"] = pd.to_numeric(h["hpi"], errors="coerce")
    h = h.dropna(subset=["hpi"])
    table = {(int(r.year), int(r.qtr)): float(r.hpi) for r in h.itertuples()}
    latest = max(table)
    log.info("FHFA HPI Ames: %d trimestres, último %sT%s = %.2f", len(table), *latest, table[latest])
    return table, latest


def to_today(values, years, quarters, hpi: dict, latest: tuple[int, int]) -> np.ndarray:
    """Lleva valores nominales a dólares del último trimestre disponible del HPI."""
    base = np.array([hpi[(int(y), int(q))] for y, q in zip(years, quarters)])
    return np.asarray(values, dtype=float) * hpi[latest] / base


_COND = {"Very Poor": 0, "Poor": 1, "Fair": 2, "Below Normal": 3, "Normal": 4,
         "Above Normal": 5, "Good": 6, "Very Good": 7, "Excellent": 8}
_BSMT = {"Full": 1.0, "3/4": 0.75, "1/2": 0.5, "1/4": 0.25, "Bsmt SF (Obsv)": 0.5}


def load_assessor_roll(path: Path) -> pd.DataFrame:
    """
    Padrón residencial 2024 del Ames City Assessor (18,931 registros, 25 columnas reales).
    Se convierte a features numéricas SIN inventar columnas (a diferencia del CSV combinado
    original, que rellenaba 61 variables con constantes).
    """
    a = pd.read_excel(path)
    a.columns = [c.strip() for c in a.columns]
    a["PID"] = a["PARCERL NUMBER"].astype(str).str.replace("-", "", regex=False).str.zfill(10)

    # Un mismo PID aparece varias veces cuando la parcela tiene varios garajes → 1 fila por PID
    n_gar = a.groupby("PID")["GARAGE TYPE"].transform(lambda s: s.notna().sum())
    a = a.assign(n_garages=n_gar).drop_duplicates("PID", keep="first").copy()

    style = a["MAIN STYLE"].fillna("")
    st = style.str.extract(r"^(\d)(?: (\d)/(\d))?")  # "1 1/2 Story" → 1 + 1/2 = 1.5
    a["stories"] = (pd.to_numeric(st[0], errors="coerce").fillna(1)
                    + (pd.to_numeric(st[1], errors="coerce") / pd.to_numeric(st[2], errors="coerce")).fillna(0))
    a["is_split"] = style.str.contains("Split").astype(int)
    a["is_condo"] = style.str.contains("Condo").astype(int)
    a["is_townhouse"] = style.str.contains("Townhouse").astype(int)
    a["is_brick"] = style.str.contains("Brick").astype(int)

    # Grado de construcción del manual de Iowa: base (1-6) + modificador (±5/±10 %)
    g = a["GRADE"].astype(str).str.extract(r"^(\d)([+-]\d+)?")
    a["grade_base"] = pd.to_numeric(g[0], errors="coerce")
    a["grade_mod"] = pd.to_numeric(g[1], errors="coerce").fillna(0)

    a["condition"] = a["CONDITION"].map(_COND)
    a["bsmt_frac"] = a["BASEMENT TYPE"].map(_BSMT).fillna(0.0)
    a["bsmt_finished"] = (a["BASEMENT FINISH"] == "Yes").astype(int)
    a["attic_finished"] = a["ATTIC TYPE"].fillna("").str.contains("Finished").astype(int)
    gt = a["GARAGE TYPE"].fillna("")
    a["has_garage"] = (gt != "").astype(int)
    a["garage_attached"] = gt.str.startswith(("Att", "Carport Att")).astype(int)
    a["garage_carport"] = gt.str.contains("Carport").astype(int)
    a["map_area"] = a["PID"].str[:4]  # township+sección: proxy geográfico público

    a = a.rename(columns={
        "YEAR BUILT": "year_built", "TOTAL LIVING AREA": "living_area",
        "TOTAL ROOMS ABOVE GRADE": "rooms_above", "TOTAL ROOMS BELOW GRADE": "rooms_below",
        "TOTAL BEDROOMS ABOVE GRADE": "beds_above", "TOTAL BEDROOMS BELOW GRADE": "beds_below",
        "# OF ADDITIONS": "n_additions", "# OF PORCHES": "n_porches",
        "# OF DECKS/PATIOS": "n_decks", "# OF PLUMBING FIXTURES": "n_plumbing",
        "# OF FIREPLACES": "n_fireplaces", "2024 LAND VALUE": "land_value_2024",
        "2024 TOTAL VALUE": "assessed_2024", "CLASSIFICATION": "classification",
        "SUBDIVISION": "subdivision",
    })
    log.info("Padrón 2024: %d parcelas únicas (%s)", len(a), a["classification"].value_counts().to_dict())
    return a


NUM_FEATURES = [
    "year_built", "living_area", "rooms_above", "rooms_below", "beds_above", "beds_below",
    "n_additions", "n_porches", "n_decks", "n_plumbing", "n_fireplaces", "n_garages",
    "stories", "is_split", "is_condo", "is_townhouse", "is_brick", "grade_base", "grade_mod",
    "condition", "bsmt_frac", "bsmt_finished", "attic_finished", "has_garage",
    "garage_attached", "garage_carport", "log_land_value",
]


def load_decock_sales(path: Path) -> pd.DataFrame:
    """Ventas reales De Cock (2006-2010). Acepta el .txt tabulado original o un CSV con PID."""
    sep = "\t" if path.suffix.lower() == ".txt" else ","
    d = pd.read_csv(path, sep=sep, dtype={"PID": str}, low_memory=False)
    d.columns = [c.replace(" ", "").replace("/", "") for c in d.columns]
    if "PID" not in d.columns:
        raise ValueError(
            "El dataset de ventas no trae PID (número de parcela). El CSV 'combined' de Kaggle "
            "no sirve: suba AmesHousing.txt (De Cock, 2,930 ventas) a input/.")
    d["PID"] = d["PID"].astype(str).str.zfill(10)
    sales = pd.DataFrame({
        "PID": d["PID"], "sale_price": d["SalePrice"].astype(float),
        "yr_sold": d["YrSold"].astype(int), "mo_sold": d["MoSold"].astype(int),
        "sale_condition": d.get("SaleCondition", "Normal"),
        "decock_living_area": d.get("GrLivArea"), "source": "decock_2006_2010",
    })
    log.info("Ventas De Cock: %d (%s)", len(sales), sales["yr_sold"].value_counts().sort_index().to_dict())
    return sales


def load_assessor_sales(path: Path | None) -> pd.DataFrame | None:
    """
    Ventas 2020-2022 del Ames City Assessor (descarga manual desde cityofames.org).
    El esquema exacto puede variar → se detectan columnas por nombre y, si no cuadra,
    se omite con un warning en lugar de romper el pipeline.
    """
    if path is None:
        return None
    s = pd.read_excel(path)
    cols = {c: re.sub(r"[^a-z]", "", str(c).lower()) for c in s.columns}
    pick = lambda *keys: next((c for c, n in cols.items() if any(k in n for k in keys)), None)
    c_pid, c_price, c_date = pick("parcel", "pid"), pick("price", "amount"), pick("date")
    if not all([c_pid, c_price, c_date]):
        log.warning("Esquema de %s no reconocido (%s) → se omite", path.name, list(s.columns))
        return None
    dt = pd.to_datetime(s[c_date], errors="coerce")
    out = pd.DataFrame({
        "PID": s[c_pid].astype(str).str.replace("-", "", regex=False).str.zfill(10),
        "sale_price": pd.to_numeric(s[c_price], errors="coerce"),
        "yr_sold": dt.dt.year, "mo_sold": dt.dt.month,
        "sale_condition": "Normal", "decock_living_area": np.nan, "source": "assessor_2020_2022",
    }).dropna(subset=["sale_price", "yr_sold"])
    out[["yr_sold", "mo_sold"]] = out[["yr_sold", "mo_sold"]].astype(int)
    log.info("Ventas assessor 2020-2022: %d", len(out))
    return out


def build_datasets(sales: pd.DataFrame, roll: pd.DataFrame, hpi: dict, latest) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une ventas ↔ padrón por PID y lleva precios y avalúos a dólares de hoy."""
    roll = roll.copy()
    land_today = to_today(roll["land_value_2024"], [ROLL_YEAR] * len(roll), [ROLL_QTR] * len(roll), hpi, latest)
    roll["log_land_value"] = np.log1p(land_today)
    roll["assessed_today"] = to_today(roll["assessed_2024"], [ROLL_YEAR] * len(roll),
                                      [ROLL_QTR] * len(roll), hpi, latest)

    # ── Cartera a evaluar: viviendas del padrón con datos válidos ──
    valid = (roll["classification"].isin(["Residential", "Ag Dwelling"])
             & (roll["assessed_2024"] > 0) & (roll["living_area"] > 0) & (roll["year_built"] > 1800))
    portfolio = roll[valid].copy()
    portfolio[NUM_FEATURES] = portfolio[NUM_FEATURES].fillna(portfolio[NUM_FEATURES].median())

    # ── Ventas de mercado (arm's length) ──
    s = sales[sales["sale_condition"].isin(["Normal", "Partial"])].copy()
    n0 = len(s)
    train = s.merge(portfolio, on="PID", how="inner")
    log.info("Ventas con match en padrón: %d / %d", len(train), n0)
    # La casa no debe haber cambiado mucho desde la venta (el padrón refleja el estado actual)
    if train["decock_living_area"].notna().any():
        diff = (train["living_area"] - train["decock_living_area"]).abs() / train["decock_living_area"]
        keep = diff.isna() | (diff <= 0.20)
        log.info("Descartadas %d ventas cuya área cambió >20%% desde la venta", (~keep).sum())
        train = train[keep]
    q = ((train["mo_sold"] - 1) // 3 + 1).astype(int)
    train["price_today"] = to_today(train["sale_price"], train["yr_sold"], q, hpi, latest)
    train["sale_date"] = train["yr_sold"] * 100 + train["mo_sold"]
    train = train.sort_values(["sale_date", "PID"]).reset_index(drop=True)  # orden cronológico
    log.info("Dataset de entrenamiento final: %d ventas reales | precio hoy mediano $%s",
             len(train), f"{train['price_today'].median():,.0f}")
    return train, portfolio.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════
# PASO 2-3 — K-MEANS + AVM SUPERVISADO
# ════════════════════════════════════════════════════════════════════
def design_matrix(df: pd.DataFrame, areas: list[str]) -> pd.DataFrame:
    """Features numéricas + one-hot del área geográfica con columnas fijas."""
    X = df[NUM_FEATURES].astype(float).copy()
    for a in areas:
        X[f"area_{a}"] = (df["map_area"] == a).astype(float)
    return X


def fit_clusters(portfolio: pd.DataFrame, train: pd.DataFrame):
    """K-Means sobre la cartera (la población que el agente evalúa) y asignación de las ventas."""
    scaler = StandardScaler().fit(portfolio[NUM_FEATURES])
    Xp = scaler.transform(portfolio[NUM_FEATURES])
    km = KMeans(n_clusters=U.N_CLUSTERS, n_init=15, random_state=U.SEED).fit(Xp)
    sil = silhouette_score(Xp, km.labels_, sample_size=min(5000, len(Xp)), random_state=U.SEED)
    Xt = scaler.transform(train[NUM_FEATURES])
    log.info("K-Means k=%d silhouette=%.4f | tamaños cartera=%s", U.N_CLUSTERS, sil,
             np.bincount(km.labels_).tolist())
    dist_p = km.transform(Xp).min(axis=1)
    dist_t = km.transform(Xt).min(axis=1)
    return scaler, km, km.labels_, km.predict(Xt), dist_p, dist_t, float(sil)


def fit_avm(X: pd.DataFrame, y_log: np.ndarray):
    """XGBoost vs LightGBM con CV 5-fold. Devuelve el mejor modelo (re-entrenado) y OOF."""
    import lightgbm as lgb
    import xgboost as xgb

    makers = {
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=5,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=U.SEED, n_jobs=-1, verbosity=0),
        "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                              random_state=U.SEED, n_jobs=-1, verbose=-1),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=U.SEED)
    results = {}
    for name, make in makers.items():
        oof = np.zeros(len(y_log))
        t0 = time.time()
        for tr, va in kf.split(X):
            m = make().fit(X.iloc[tr], y_log[tr])
            oof[va] = m.predict(X.iloc[va])
        real, pred = np.expm1(y_log), np.expm1(oof)
        results[name] = {
            "oof": oof, "r2_log": float(r2_score(y_log, oof)),
            "r2_usd": float(r2_score(real, pred)), "mae_usd": float(mean_absolute_error(real, pred)),
            "mape": float(np.mean(np.abs(pred - real) / real)),
            "pct_err_lt_10": float(np.mean(np.abs(pred - real) / real < 0.10)),
        }
        log.info("  %-8s CV5 → R²(log)=%.4f R²($)=%.4f MAE=$%s MAPE=%.1f%% (%.1fs)", name,
                 results[name]["r2_log"], results[name]["r2_usd"], f"{results[name]['mae_usd']:,.0f}",
                 100 * results[name]["mape"], time.time() - t0)
    best = max(results, key=lambda k: results[k]["r2_log"])
    model = makers[best]().fit(X, y_log)
    log.info("  → AVM seleccionado: %s", best)
    return best, model, results


# ════════════════════════════════════════════════════════════════════
# PASO 4-6 — MDP, VALUE ITERATION, Q-LEARNING
# ════════════════════════════════════════════════════════════════════
def build_mdp(clusters: np.ndarray, errors: np.ndarray):
    """
    R[s][a]: recompensa media con errores OUT-OF-FOLD del AVM (sin sesgo in-sample).
    P[s][s']: transición entre ventas consecutivas en ORDEN CRONOLÓGICO (flujo real de
              solicitudes de valuación), no el orden arbitrario de filas del CSV.
    """
    S = list(range(U.N_CLUSTERS))
    Rm = U.reward_matrix(errors)
    R = np.zeros((U.N_CLUSTERS, U.N_ACTIONS))
    for s in S:
        m = clusters == s
        if m.any():
            R[s] = Rm[m].mean(axis=0)
    P = np.zeros((U.N_CLUSTERS, U.N_CLUSTERS))
    for s, ns in zip(clusters[:-1], clusters[1:]):
        P[s, ns] += 1
    rows = P.sum(axis=1, keepdims=True)
    P = np.where(rows > 0, P / np.maximum(rows, 1), np.eye(U.N_CLUSTERS))  # estado sin datos → absorbente
    for s in S:
        log.info("  S%d n=%4d  R: APROBAR=%8.1f REVISAR=%8.1f RECHAZAR=%8.1f", s,
                 (clusters == s).sum(), *R[s])
    return R, P


def value_iteration(R: np.ndarray, P: np.ndarray):
    V = np.zeros(U.N_CLUSTERS)
    it = 0
    while True:
        it += 1
        Q = R + U.GAMMA * (P @ V)[:, None]
        V_new = Q.max(axis=1)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < U.THETA:
            break
    Q = R + U.GAMMA * (P @ V)[:, None]
    policy = {s: U.ACTIONS[int(Q[s].argmax())] for s in range(U.N_CLUSTERS)}
    log.info("  VI convergió en %d iteraciones → %s", it, policy)
    return V, Q, policy, it


def q_learning(clusters: np.ndarray, errors: np.ndarray, P: np.ndarray):
    """
    Q-Learning tabular. Misma lógica que el monolito con dos mejoras:
      · las transiciones s' de cada episodio se muestrean vectorizadas (antes: un
        np.random.choice por paso = principal cuello de botella del EC2);
      · α decae con las visitas a (s,a): α = ALPHA_QL / (1 + n_sa / QL_ALPHA_HALF).
        Con α constante (0.1) y recompensas de hasta -2000 la tabla Q oscilaba más que
        la diferencia entre acciones y la política dependía del ruido (Robbins-Monro).
    """
    rng = np.random.default_rng(U.SEED)
    Q = [[0.0] * U.N_ACTIONS for _ in range(U.N_CLUSTERS)]
    visits = [[0] * U.N_ACTIONS for _ in range(U.N_CLUSTERS)]
    Rm = U.reward_matrix(errors).tolist()
    cum = np.cumsum(P, axis=1)
    eps, curve = U.EPSILON_0, []
    n = len(clusters)
    t0 = time.time()
    for ep in range(U.EPISODES_QL):
        order = rng.permutation(n)
        s_arr = clusters[order]
        s_next = (rng.random(n)[:, None] > cum[s_arr]).sum(axis=1).clip(max=U.N_CLUSTERS - 1)
        explore = (rng.random(n) < eps).tolist()
        rand_a = rng.integers(0, U.N_ACTIONS, n).tolist()
        ep_r = 0.0
        for i, idx in enumerate(order.tolist()):
            s = int(s_arr[i])
            qs = Q[s]
            a = rand_a[i] if explore[i] else max(range(U.N_ACTIONS), key=qs.__getitem__)
            r = Rm[idx][a]
            visits[s][a] += 1
            alpha = U.ALPHA_QL / (1.0 + visits[s][a] / U.QL_ALPHA_HALF)
            qs[a] += alpha * (r + U.GAMMA * max(Q[int(s_next[i])]) - qs[a])
            ep_r += r
        eps = max(U.EPS_MIN, eps * U.EPS_DECAY)
        curve.append(ep_r / n)
        if (ep + 1) % 2000 == 0:
            log.info("  QL ep %5d/%d | ε=%.3f | reward medio=%.1f | %.0fs", ep + 1,
                     U.EPISODES_QL, eps, curve[-1], time.time() - t0)
    policy = {s: U.ACTIONS[int(np.argmax(Q[s]))] for s in range(U.N_CLUSTERS)}
    log.info("  Política Q-Learning → %s", policy)
    return np.array(Q), policy, curve


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def run(input_uri: str, reference: str, output: str, run_id: str) -> dict:
    U.set_seed()
    t_start = time.time()
    work = Path(tempfile.mkdtemp(prefix="savi_"))
    out = work / "artifacts"
    out.mkdir()
    log.info("═" * 60)
    log.info("SAVI v2 · MOTOR CPU · run_id=%s", run_id)
    log.info("input=%s | reference=%s | output=%s", input_uri, reference, output)

    # ── PASO 1: datos reales ──
    log.info("[PASO 1] Integración de fuentes reales de Ames, Iowa")
    hpi, latest = load_hpi(fetch_reference(reference, HPI_FILE, work))
    roll = load_assessor_roll(fetch_reference(reference, ASSESSOR_ROLL, work))
    input_path = fetch(input_uri, work)
    sales = load_decock_sales(input_path)
    extra = load_assessor_sales(fetch_reference(reference, ASSESSOR_SALES, work, required=False))
    if extra is not None:
        sales = pd.concat([sales, extra], ignore_index=True)
    train, portfolio = build_datasets(sales, roll, hpi, latest)

    # ── PASO 2: K-Means ──
    log.info("[PASO 2] K-Means → estados del MDP")
    scaler, km, port_cl, train_cl, dist_p, dist_t, sil = fit_clusters(portfolio, train)

    # ── PASO 3: AVM supervisado ──
    log.info("[PASO 3] AVM supervisado (precio en dólares de %sT%s)", *latest)
    areas = sorted(portfolio["map_area"].unique())
    X_tr = design_matrix(train, areas)
    X_tr["cluster"] = train_cl
    y_log = np.log1p(train["price_today"].values)
    best_name, avm, sup = fit_avm(X_tr, y_log)
    oof_pred = np.expm1(sup[best_name]["oof"])
    errors = np.clip(np.abs(oof_pred - train["price_today"].values) / train["price_today"].values, 0, 5)

    X_port = design_matrix(portfolio, areas)
    X_port["cluster"] = port_cl
    port_pred = np.expm1(avm.predict(X_port))

    # ── Estado continuo del DQN: top-20 features del AVM + 3 señales de incertidumbre ──
    imp = pd.Series(avm.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    top = imp.index[:N_DQN_TOP].tolist()
    log.info("  Top-5 features AVM: %s", top[:5])

    def dqn_state(X: pd.DataFrame, pred, assessed, dist) -> np.ndarray:
        extra_sig = np.column_stack([
            np.log1p(pred),                                   # magnitud de la valuación
            np.clip((pred - assessed) / assessed, -1, 1),     # brecha AVM vs avalúo oficial
            dist,                                             # atipicidad (distancia al centroide)
        ])
        return np.hstack([X[top].values, extra_sig])

    S_tr = dqn_state(X_tr, oof_pred, train["assessed_today"].values, dist_t)
    S_port = dqn_state(X_port, port_pred, portfolio["assessed_today"].values, dist_p)
    mms = MinMaxScaler().fit(S_port)
    S_tr = np.clip(mms.transform(S_tr), 0, 1).astype(np.float32)
    S_port = np.clip(mms.transform(S_port), 0, 1).astype(np.float32)

    # ── PASO 4-6: MDP + VI + QL ──
    # Split estratificado por cluster: los agentes aprenden SÓLO en TRAIN; VAL elige la
    # regla final (en SageMaker) y TEST se reporta una única vez → métricas sin sesgo.
    split = U.stratified_split(train_cl)
    tr = split == U.SPLIT_TRAIN
    log.info("[PASO 4] Ambiente RL (R con errores OOF, P cronológica) | split train/val/test = %s",
             np.bincount(split, minlength=3).tolist())
    R, P = build_mdp(train_cl[tr], errors[tr])  # la máscara conserva el orden cronológico
    log.info("[PASO 5] Value Iteration (γ=%.2f θ=%g)", U.GAMMA, U.THETA)
    V, Q_vi, pol_vi, it_vi = value_iteration(R, P)
    log.info("[PASO 6] Q-Learning tabular (%d episodios × %d ventas de TRAIN)", U.EPISODES_QL, int(tr.sum()))
    Q_ql, pol_ql, ql_curve = q_learning(train_cl[tr], errors[tr], P)

    # ── Exportación ──
    log.info("[PASO 7] Exportando artefactos")
    A = U.ARTIFACTS
    next_idx = U.next_within_split(split)  # siguiente venta cronológica del MISMO subconjunto
    for key, arr in [("train_states", S_tr), ("train_errors", errors.astype(np.float32)),
                     ("train_next", next_idx), ("train_clusters", train_cl), ("train_split", split),
                     ("portfolio_states", S_port), ("portfolio_clusters", port_cl)]:
        (out / A[key]).parent.mkdir(parents=True, exist_ok=True)
        np.save(out / A[key], arr)
    meta = pd.DataFrame({
        "PID": portfolio["PID"], "subdivision": portfolio["subdivision"],
        "map_area": portfolio["map_area"], "cluster": port_cl,
        "avm_value_today": port_pred.round(0), "assessed_2024": portfolio["assessed_2024"],
        "assessed_today": portfolio["assessed_today"].round(0),
        "gap_vs_assessed": ((port_pred - portfolio["assessed_today"]) / portfolio["assessed_today"]).round(4),
    })
    (out / A["portfolio_meta"]).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / A["portfolio_meta"], index=False)
    (out / "data").mkdir(exist_ok=True)
    pd.DataFrame({"PID": train["PID"], "source": train["source"], "sale_date": train["sale_date"],
                  "price_today": train["price_today"].round(0), "avm_oof": oof_pred.round(0),
                  "error": errors.round(4), "cluster": train_cl,
                  "split": np.array(["train", "val", "test"])[split]}).to_csv(out / "data/train_sales.csv", index=False)

    U.save_json(pol_vi, out / A["policy_vi"])
    U.save_json(pol_ql, out / A["policy_ql"])
    U.save_json({str(s): dict(zip(U.ACTIONS, Q_ql[s])) for s in range(U.N_CLUSTERS)}, out / A["q_table_ql"])
    U.save_json({"R": R, "P": P, "V_vi": V, "Q_vi": Q_vi, "vi_iterations": it_vi,
                 "ql_reward_curve": ql_curve[::10]}, out / A["mdp"])
    models = out / "models"
    models.mkdir()
    joblib.dump({"scaler": scaler, "kmeans": km}, models / "kmeans.joblib")
    joblib.dump({"model": avm, "name": best_name, "columns": list(X_tr.columns), "areas": areas},
                models / "avm.joblib")
    joblib.dump({"minmax": mms, "top_features": top}, models / "dqn_state_scaler.joblib")

    metrics = {
        "run_id": run_id, "hpi_latest": f"{latest[0]}Q{latest[1]}", "n_train_sales": int(len(train)),
        "n_portfolio": int(len(portfolio)), "sources": train["source"].value_counts().to_dict(),
        "kmeans_silhouette": sil,
        "supervised": {k: {m: v for m, v in r.items() if m != "oof"} for k, r in sup.items()},
        "avm_selected": best_name, "dqn_state_dim": int(S_tr.shape[1]), "dqn_state_features": top
        + ["log_avm_value", "gap_vs_assessed", "kmeans_distance"],
        "split_sizes": dict(zip(["train", "val", "test"], np.bincount(split, minlength=3).tolist())),
        **{f"reward_{n}_{g}": U.eval_policy(pol, train_cl[split == k], errors[split == k])
           for n, pol in [("vi", pol_vi), ("ql", pol_ql)]
           for g, k in [("train", U.SPLIT_TRAIN), ("val", U.SPLIT_VAL), ("test", U.SPLIT_TEST)]},
        "policy_vi": pol_vi, "policy_ql": pol_ql,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    # ── PASO 8: base de conocimiento del agente RAG (rag/*.json) ──
    log.info("[PASO 8] Exportando base de conocimiento RAG")
    metrics["rag"] = RAG.export_rag(
        out, reference_fetch=lambda name: fetch_reference(reference, name, work, required=False),
        decock_path=input_path, train=train, portfolio=portfolio, port_pred=port_pred, port_cl=port_cl,
        oof_pred=oof_pred, errors=errors, train_cl=train_cl, split=split, hpi=hpi, latest=latest,
        hpi_path=fetch_reference(reference, HPI_FILE, work), avm=avm, avm_name=best_name,
        avm_columns=list(X_tr.columns), scaler=scaler, km=km, mms=mms, top=top,
        num_features=NUM_FEATURES, metrics=metrics)
    U.save_json(metrics, out / A["metrics_cpu"])
    log.info("  Reward medio en VAL → VI=%.1f | QL=%.1f", metrics["reward_vi_val"], metrics["reward_ql_val"])

    # _SUCCESS se escribe AL FINAL: es la señal para la Lambda 2
    U.save_json({"run_id": run_id, "status": "SUCCEEDED", "output": output,
                 "finished_utc": datetime.now(timezone.utc).isoformat(),
                 "artifacts": A}, out / A["success"])
    publish(out, output)
    shutil.rmtree(work, ignore_errors=True)
    log.info("MOTOR CPU completado en %.1fs", time.time() - t_start)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="SAVI v2 · motor CPU (EC2)")
    ap.add_argument("--input", required=True, help="Ventas (AmesHousing.txt) local o s3://")
    ap.add_argument("--reference", required=True, help="Dir local o prefijo s3:// con referencias")
    ap.add_argument("--output", required=True, help="Dir local o prefijo s3:// de salida")
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()
    run(args.input, args.reference, args.output, args.run_id)


if __name__ == "__main__":
    main()
