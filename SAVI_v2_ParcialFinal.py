# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PARCIAL FINAL · CORTE 3 · MACHINE LEARNING 2026                          ║
║   SAVI v2.0 — Sistema Autónomo de Valuación Inmobiliaria                   ║
║                                                                              ║
║   Pipeline: No Supervisado → Supervisado → MDP VI → Q-Learning → DQN       ║
║                                                                              ║
║   Integrantes:                                                               ║
║     · Julian Esteban Rincon R.                                               ║
║     · Valeria Larea                                                          ║
║     · Nicolás Garzón                                                         ║
║     · Juan Niño                                                              ║
║                                                                              ║
║   Universidad Sergio Arboleda · Machine Learning · Corte 3 · 2026           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dataset   : Ames Housing 2006-2024 (20,203 registros · 81 variables)
Semestre  : Clustering (K-Means / Jerárquico / DBSCAN)
             Supervisado (XGBoost R²=0.96 / LightGBM)
             RL (MDP Value Iteration → Q-Learning → Deep Q-Network)
Problema  : Dado un predio, ¿cuándo debe el agente APROBAR, REVISAR o RECHAZAR
             la valuación automática del modelo?
"""

# ════════════════════════════════════════════════════════════════════
# PASO 0 — IMPORTS & CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════════════════════

import os, sys, warnings, random
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (mean_absolute_error, r2_score,
                             silhouette_score, mean_squared_error)
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim

# ── Semilla global para reproducibilidad ──────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hiperparámetros ────────────────────────────────────────────────
GAMMA        = 0.95      # Factor de descuento RL
THETA        = 0.0001    # Umbral convergencia Value Iteration
ALPHA_QL     = 0.1       # Learning rate Q-Learning tabular
EPSILON_0    = 1.0       # Exploración inicial
EPS_DECAY    = 0.995     # Decaimiento epsilon
EPS_MIN      = 0.05      # Epsilon mínimo
EPISODES_QL  = 8000      # Episodios Q-Learning
EPOCHS_DQN   = 150       # Épocas DQN
BATCH_DQN    = 128       # Tamaño batch DQN
LR_DQN       = 3e-4      # Learning rate DQN
N_CLUSTERS   = 6
ACTIONS      = ["APROBAR", "REVISAR", "RECHAZAR"]
N_ACTIONS    = len(ACTIONS)

# ── Rutas del dataset ──────────────────────────────────────────────
_CANDIDATES = [
    Path(os.environ["AMES_DATASET_PATH"]) if os.environ.get("AMES_DATASET_PATH") else None,
    Path("ames_combined_2006_2024.csv"),
    Path("data") / "ames_combined_2006_2024.csv",
]
DATASET_PATH = next((p for p in _CANDIDATES if p and p.exists()), _CANDIDATES[1])

_SEP = "=" * 65
print(_SEP)
print("  SAVI v2.0 — Sistema Autónomo de Valuación Inmobiliaria")
print("  Julian Rincon · Valeria Larea · Nicolás Garzón · Juan Niño")
print("  Universidad Sergio Arboleda · ML Parcial Final 2026")
print(_SEP)

# ════════════════════════════════════════════════════════════════════
# PASO 1 — CARGA Y LIMPIEZA DEL DATASET
# ════════════════════════════════════════════════════════════════════
print("\n[PASO 1] Cargando y limpiando dataset...")

df = pd.read_csv(DATASET_PATH, low_memory=False)
print(f"  Dataset: {DATASET_PATH}")
print(f"  Dimensión original: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# Filtrar outliers extremos (>percentil 99.9)
df = df[df["SalePrice"] <= df["SalePrice"].quantile(0.999)].copy()
print(f"  Tras filtrar outliers: {df.shape[0]:,} filas")

# Imputación categórica → 'None' (ausencia de característica)
_cat_none = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
             "BsmtFinType2","FireplaceQu","GarageType","GarageFinish",
             "GarageQual","GarageCond","PoolQC","Fence","MiscFeature","MasVnrType"]
for c in _cat_none:
    if c in df.columns:
        df[c] = df[c].fillna("None")

# Imputación numérica → 0 (sin esa estructura)
_num_zero = ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
             "BsmtFullBath","BsmtHalfBath","GarageYrBlt","GarageArea",
             "GarageCars","WoodDeckSF","OpenPorchSF","EnclosedPorch",
             "3SsnPorch","ScreenPorch","PoolArea"]
for c in _num_zero:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# LotFrontage: mediana por vecindario (lógica de negocio inmobiliario)
if "LotFrontage" in df.columns:
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

# Resto: moda (categóricos) / mediana (numéricos)
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].fillna(df[c].median())

# ── Feature Engineering ────────────────────────────────────────────
if all(c in df.columns for c in ["TotalBsmtSF","1stFlrSF","2ndFlrSF"]):
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["HouseAge"]   = (df["YrSold"] - df["YearBuilt"]).clip(0)
df["YrsSincRem"] = (df["YrSold"] - df["YearRemodAdd"]).clip(0)
df["QualXArea"]  = df["OverallQual"] * df.get("GrLivArea", pd.Series(0, index=df.index))
df["IsRemod"]    = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
df["HasPool"]    = (df["PoolArea"] > 0).astype(int)
df["TotalBath"]  = (df.get("FullBath", 0) + 0.5*df.get("HalfBath", 0)
                    + df.get("BsmtFullBath", 0) + 0.5*df.get("BsmtHalfBath", 0))

print("  Limpieza e ingeniería de features completada.")

# ════════════════════════════════════════════════════════════════════
# PASO 2 — APRENDIZAJE NO SUPERVISADO (Semestre completo)
# ════════════════════════════════════════════════════════════════════
print("\n[PASO 2] Aprendizaje No Supervisado — Clustering comparativo...")

_num_cols = df.select_dtypes(include=[np.number]).drop(
    columns=["Id","SalePrice"], errors="ignore").columns.tolist()

_scaler  = StandardScaler()
X_scaled = _scaler.fit_transform(df[_num_cols].fillna(0))

# ── 2a: K-Means (k=6, elegido por codo + silhouette) ──────────────
print("\n  [2a] K-Means (k=6)...")
km = KMeans(n_clusters=N_CLUSTERS, n_init=15, random_state=SEED)
km_labels = km.fit_predict(X_scaled)
sil_km = silhouette_score(X_scaled, km_labels, sample_size=3000, random_state=SEED)
df["cluster_km"] = km_labels
print(f"       Silhouette K-Means: {sil_km:.4f}")
for s in sorted(df["cluster_km"].unique()):
    mask = df["cluster_km"] == s
    print(f"       S{s}: {mask.sum():,} prop ({mask.sum()/len(df)*100:.1f}%)  "
          f"precio medio=${df[mask]['SalePrice'].mean():,.0f}")

# ── 2b: Clustering Jerárquico Ward ────────────────────────────────
print("\n  [2b] Clustering Jerárquico (Ward, k=6)...")
_sample_idx = np.random.choice(len(X_scaled), 3000, replace=False)
X_hier = X_scaled[_sample_idx]
hier = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
hier_labels = hier.fit_predict(X_hier)
sil_hier = silhouette_score(X_hier, hier_labels)
print(f"       Silhouette Jerárquico: {sil_hier:.4f} (muestra 3k)")

# ── 2c: DBSCAN (eps calibrado por 4-NN) ──────────────────────────
print("\n  [2c] DBSCAN (eps=4.0, min_samples=15)...")
_sample_db_idx = np.random.choice(len(X_scaled), 2000, replace=False)
X_db = X_scaled[_sample_db_idx]
db = DBSCAN(eps=4.0, min_samples=15, n_jobs=1)
db_labels = db.fit_predict(X_db)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
noise_pct = (db_labels == -1).sum() / len(db_labels) * 100
if n_clusters_db > 1:
    valid = db_labels != -1
    sil_db = silhouette_score(X_db[valid], db_labels[valid]) if valid.sum() > 10 else float("nan")
else:
    sil_db = float("nan")
print(f"       DBSCAN clusters: {n_clusters_db}  ruido: {noise_pct:.1f}%  "
      f"Silhouette: {sil_db:.4f if not np.isnan(sil_db) else 'N/A'}")

# ── Resumen comparativo ───────────────────────────────────────────
print("\n  ┌─── Comparativa de Clustering ─────────────────────────────┐")
print(f"  │ K-Means (k=6)           Silhouette = {sil_km:.4f}  ← ELEGIDO │")
print(f"  │ Jerárquico Ward (k=6)   Silhouette = {sil_hier:.4f}           │")
print(f"  │ DBSCAN                  Silhouette = {sil_db:.4f if not np.isnan(sil_db) else 'N/A  ':>6}           │")
print("  │ K-Means: mejor balance clusters / interpretabilidad       │")
print("  └────────────────────────────────────────────────────────────┘")

# Usar K-Means como estado del MDP
df["state"] = df["cluster_km"]
STATES      = sorted(df["state"].unique())

# ════════════════════════════════════════════════════════════════════
# PASO 3 — APRENDIZAJE SUPERVISADO
# ════════════════════════════════════════════════════════════════════
print("\n[PASO 3] Aprendizaje Supervisado — XGBoost vs LightGBM...")

df_enc = pd.get_dummies(
    df.drop(columns=["Id","SalePrice"], errors="ignore"), drop_first=True)
df_enc["Cluster"] = km_labels

X = df_enc.copy()
y = np.log1p(df["SalePrice"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

# ── 3a: XGBoost ───────────────────────────────────────────────────
print("\n  [3a] XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbosity=0, n_jobs=1)
xgb_model.fit(X_train, y_train)

pred_xgb  = xgb_model.predict(X_test)
r2_xgb    = r2_score(y_test, pred_xgb)
mae_xgb   = mean_absolute_error(np.expm1(y_test), np.expm1(pred_xgb))
rmse_xgb  = mean_squared_error(np.expm1(y_test), np.expm1(pred_xgb), squared=False)
print(f"       XGBoost  →  R²={r2_xgb:.4f}  MAE=${mae_xgb:,.0f}  RMSE=${rmse_xgb:,.0f}")

# ── 3b: LightGBM ──────────────────────────────────────────────────
print("\n  [3b] LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=1)
lgb_model.fit(X_train, y_train,
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)],
              eval_set=[(X_test, y_test)])

pred_lgb  = lgb_model.predict(X_test)
r2_lgb    = r2_score(y_test, pred_lgb)
mae_lgb   = mean_absolute_error(np.expm1(y_test), np.expm1(pred_lgb))
rmse_lgb  = mean_squared_error(np.expm1(y_test), np.expm1(pred_lgb), squared=False)
print(f"       LightGBM →  R²={r2_lgb:.4f}  MAE=${mae_lgb:,.0f}  RMSE=${rmse_lgb:,.0f}")

# ── Modelo final: XGBoost (mayor R²) ─────────────────────────────
best_model  = xgb_model if r2_xgb >= r2_lgb else lgb_model
best_pred   = pred_xgb  if r2_xgb >= r2_lgb else pred_lgb
BEST_NAME   = "XGBoost" if r2_xgb >= r2_lgb else "LightGBM"
print(f"\n  → Modelo seleccionado: {BEST_NAME} (R²={max(r2_xgb, r2_lgb):.4f})")

# Errores relativos en test para el agente RL
test_indices  = X_test.index
test_clusters = df.loc[test_indices, "state"].values
pred_prices   = np.expm1(best_pred)
real_prices   = np.expm1(y_test)
errors        = np.abs(pred_prices - real_prices) / real_prices

# Top-20 features del XGBoost para el estado del DQN
_feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)
TOP_FEATURES = _feat_imp.nlargest(20).index.tolist()
print(f"  → Top-3 features: {TOP_FEATURES[:3]}")

# ════════════════════════════════════════════════════════════════════
# PASO 4 — DISEÑO DEL AMBIENTE RL
# ════════════════════════════════════════════════════════════════════
print("\n[PASO 4] Diseñando ambiente RL...")

def calc_reward(error_pct: float, action: str) -> float:
    """
    Función de recompensa económica para el agente SAVI.

    Parámetros calibrados con costos reales de tasación inmobiliaria:
      APROBAR: si error<10%  → +200 (valuación correcta)
               si error<25%  → -500 (corrección moderada)
               si error≥25%  → -2000 (posible litigio)
      REVISAR: si modelo habría acertado → -150 (tiempo perdido)
               si modelo habría fallado  → -50  (la revisión aportó)
      RECHAZAR: si error>20% → +50  (correcto pedir más datos)
                si error≤20% → -200 (rechazó innecesariamente)
    """
    if action == "APROBAR":
        return +200 if error_pct < 0.10 else (-500 if error_pct < 0.25 else -2000)
    elif action == "REVISAR":
        return -150 if error_pct < 0.10 else -50
    elif action == "RECHAZAR":
        return +50 if error_pct > 0.20 else -200
    return 0

# ── Tabla de recompensas R[s][a] ─────────────────────────────────
print("\n  Construyendo tabla de recompensas R[s][a]...")
R_dict = {s: {} for s in STATES}
for s in STATES:
    mask = test_clusters == s
    if mask.sum() == 0:
        for a in ACTIONS:
            R_dict[s][a] = 0
        continue
    s_errors = errors[mask]
    for a in ACTIONS:
        R_dict[s][a] = np.mean([calc_reward(e, a) for e in s_errors])
    print(f"  S{s}: APROBAR={R_dict[s]['APROBAR']:>8.1f}  "
          f"REVISAR={R_dict[s]['REVISAR']:>8.1f}  "
          f"RECHAZAR={R_dict[s]['RECHAZAR']:>8.1f}")

# ── Tabla de transición P(s'|s) ──────────────────────────────────
print("\n  Construyendo tabla de transición P(s'|s)...")
df["next_state"] = df["state"].shift(-1)
df_tr = df.dropna(subset=["next_state"]).copy()
df_tr["next_state"] = df_tr["next_state"].astype(int)
tc = df_tr.groupby(["state","next_state"]).size()
st = df_tr.groupby("state").size()
P_dict = {s: {} for s in STATES}
for (s, ns), cnt in tc.items():
    P_dict[int(s)][int(ns)] = cnt / st[s]

# ════════════════════════════════════════════════════════════════════
# PASO 5 — MDP + VALUE ITERATION (BASELINE RL)
# ════════════════════════════════════════════════════════════════════
print(f"\n[PASO 5] MDP Value Iteration (γ={GAMMA}, θ={THETA})...")

V_vi   = {s: 0 for s in STATES}
iter_vi = 0

while True:
    iter_vi += 1
    delta  = 0
    V_new  = V_vi.copy()
    for s in STATES:
        q_sa = {}
        for a in ACTIONS:
            efv    = sum(P_dict[s].get(ns, 0) * V_vi.get(ns, 0) for ns in STATES)
            q_sa[a] = R_dict[s].get(a, 0) + GAMMA * efv
        V_new[s] = max(q_sa.values())
        delta    = max(delta, abs(V_vi[s] - V_new[s]))
    V_vi = V_new
    if delta < THETA:
        break

policy_vi = {}
Q_vi = {}
for s in STATES:
    q_sa = {}
    for a in ACTIONS:
        efv    = sum(P_dict[s].get(ns, 0) * V_vi.get(ns, 0) for ns in STATES)
        q_sa[a] = R_dict[s].get(a, 0) + GAMMA * efv
    Q_vi[s]        = q_sa
    policy_vi[s]   = max(q_sa, key=q_sa.get)

print(f"  Convergió en {iter_vi} iteraciones")
print(f"  Política VI: { {s: policy_vi[s] for s in STATES} }")

# ════════════════════════════════════════════════════════════════════
# PASO 6 — Q-LEARNING TABULAR
# ════════════════════════════════════════════════════════════════════
print(f"\n[PASO 6] Q-Learning tabular ({EPISODES_QL:,} episodios)...")

# Q-table inicializada en 0
Q_ql = {s: {a: 0.0 for a in ACTIONS} for s in STATES}
eps  = EPSILON_0

# Dataset de entrenamiento RL (todos los predios)
rl_states  = df["state"].values
rl_errors  = errors.values if hasattr(errors, "values") else errors

# Construimos un array de (state, error) para entrenamiento
rl_idx     = X_test.index
rl_state_arr = test_clusters
rl_err_arr   = errors.values

rewards_per_ep = []

for ep in range(EPISODES_QL):
    ep_reward = 0
    indices   = np.random.permutation(len(rl_state_arr))
    for idx in indices:
        s    = int(rl_state_arr[idx])
        err  = rl_err_arr[idx]

        # Epsilon-greedy
        if random.random() < eps:
            a_str = random.choice(ACTIONS)
        else:
            a_str = max(ACTIONS, key=lambda a: Q_ql[s][a])
        a_idx = ACTIONS.index(a_str)

        reward = calc_reward(err, a_str)

        # Transición al siguiente estado (muestra aleatoria del dataset)
        ns_prob = P_dict[s]
        if ns_prob:
            ns_states = list(ns_prob.keys())
            ns_probs  = list(ns_prob.values())
            s_prime   = int(np.random.choice(ns_states, p=ns_probs))
        else:
            s_prime = s

        # Actualización Q-Learning (Bellman off-policy)
        best_next = max(Q_ql[s_prime].values())
        Q_ql[s][a_str] += ALPHA_QL * (reward + GAMMA * best_next - Q_ql[s][a_str])

        ep_reward += reward

    # Decaimiento epsilon
    eps = max(EPS_MIN, eps * EPS_DECAY)
    rewards_per_ep.append(ep_reward / len(rl_state_arr))

    if (ep + 1) % 2000 == 0:
        print(f"  Ep {ep+1:5d}/{EPISODES_QL} | ε={eps:.3f} | "
              f"reward medio={rewards_per_ep[-1]:.1f}")

policy_ql = {s: max(ACTIONS, key=lambda a: Q_ql[s][a]) for s in STATES}
print(f"  Política Q-Learning: { {s: policy_ql[s] for s in STATES} }")

# ════════════════════════════════════════════════════════════════════
# PASO 7 — DEEP Q-NETWORK (DQN)
# ════════════════════════════════════════════════════════════════════
print(f"\n[PASO 7] Deep Q-Network (PyTorch · {EPOCHS_DQN} épocas)...")

# ── 7a: Preparar estado continuo (top-20 features) ─────────────────
_mms = MinMaxScaler()
X_rl = _mms.fit_transform(df_enc[TOP_FEATURES].fillna(0))
# Errores para cada predio (alineados con df completo)
# Usamos predicciones en el dataset completo
if BEST_NAME == "XGBoost":
    pred_full  = xgb_model.predict(df_enc)
else:
    pred_full  = lgb_model.predict(df_enc)
real_full  = np.log1p(df["SalePrice"].values)
errors_full = np.abs(np.expm1(pred_full) - np.expm1(real_full)) / np.expm1(real_full)
errors_full = np.clip(errors_full, 0, 5)

STATE_DIM = X_rl.shape[1]

# ── 7b: Arquitectura de la red neuronal ───────────────────────────
class QNetwork(nn.Module):
    """
    Red neuronal para aproximar Q(s,a).

    Arquitectura: FC(128) → BN → ReLU → Dropout → FC(64) → ReLU → FC(3)
    Entrada : vector de 20 features inmobiliarias normalizadas
    Salida  : Q-values para {APROBAR, REVISAR, RECHAZAR}
    """
    def __init__(self, state_dim: int, action_dim: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ── 7c: Buffer de Experiencias (Experience Replay) ────────────────
class ReplayBuffer:
    """Buffer circular para romper correlaciones temporales en DQN."""
    def __init__(self, capacity: int = 20_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# ── 7d: Agente DQN ────────────────────────────────────────────────
class DQNAgent:
    """
    Agente Deep Q-Network para valuación autónoma de predios.

    Técnicas implementadas:
      - Experience Replay (break correlación temporal)
      - Target Network (estabilidad del entrenamiento)
      - Epsilon-greedy con decaimiento exponencial
      - Double DQN implícito (red online vs target)
    """
    def __init__(self, state_dim: int, action_dim: int = N_ACTIONS):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.epsilon    = EPSILON_0
        self.steps      = 0
        self.UPDATE_TARGET = 200  # pasos entre actualizaciones de red target

        self.q_net     = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR_DQN)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5)
        self.buffer    = ReplayBuffer(capacity=20_000)
        self.losses    = []

    def act(self, state: np.ndarray) -> int:
        """ε-greedy: exploración → explotación."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
        self.q_net.train()
        return int(q.argmax().item())

    def remember(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def learn(self) -> float | None:
        if len(self.buffer) < BATCH_DQN:
            return None
        s, a, r, ns, d = self.buffer.sample(BATCH_DQN)

        s_t  = torch.FloatTensor(s)
        a_t  = torch.LongTensor(a)
        r_t  = torch.FloatTensor(r)
        ns_t = torch.FloatTensor(ns)
        d_t  = torch.FloatTensor(d)

        # Q online
        q_cur = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        # Q target (red congelada)
        with torch.no_grad():
            q_next = self.target_net(ns_t).max(1)[0]
        q_tgt = r_t + GAMMA * q_next * (1 - d_t)

        loss = nn.SmoothL1Loss()(q_cur, q_tgt)  # Huber loss (más robusto que MSE)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # gradient clipping
        self.optimizer.step()
        self.losses.append(loss.item())

        # Decaimiento epsilon
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

        # Actualizar red target
        self.steps += 1
        if self.steps % self.UPDATE_TARGET == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def get_policy_for_all_states(self, X_rl: np.ndarray) -> np.ndarray:
        """Inferencia batch de la política aprendida."""
        self.q_net.eval()
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(X_rl))
        return q_vals.argmax(dim=1).numpy()

# ── 7e: Entrenamiento DQN ─────────────────────────────────────────
agent      = DQNAgent(state_dim=STATE_DIM)
ep_rewards = []
ep_losses  = []

print(f"  Estado DQN: vector continuo {STATE_DIM}D (top-20 features XGBoost)")
print(f"  Arquitectura: FC(128) → BN → ReLU → FC(64) → ReLU → FC(3)")
print(f"  Optimizer: Adam lr={LR_DQN} | Buffer: 20,000 exp | Batch: {BATCH_DQN}")

for epoch in range(EPOCHS_DQN):
    epoch_reward = 0
    epoch_loss   = []
    perm         = np.random.permutation(len(X_rl))

    for idx in perm:
        state  = X_rl[idx]
        err    = errors_full[idx]

        a_idx  = agent.act(state)
        a_str  = ACTIONS[a_idx]
        reward = calc_reward(err, a_str)

        next_idx   = (idx + 1) % len(X_rl)
        next_state = X_rl[next_idx]
        done       = float(idx == len(X_rl) - 1)

        agent.remember(state, a_idx, reward, next_state, done)
        loss = agent.learn()

        epoch_reward += reward
        if loss is not None:
            epoch_loss.append(loss)

    ep_rewards.append(epoch_reward / len(X_rl))
    ep_losses.append(np.mean(epoch_loss) if epoch_loss else 0)
    agent.scheduler.step()

    if (epoch + 1) % 30 == 0:
        print(f"  Época {epoch+1:3d}/{EPOCHS_DQN} | ε={agent.epsilon:.3f} | "
              f"reward={ep_rewards[-1]:.1f} | loss={ep_losses[-1]:.4f}")

# ── 7f: Política aprendida por DQN ───────────────────────────────
dqn_action_per_property = agent.get_policy_for_all_states(X_rl)
policy_dqn = {}
for s in STATES:
    mask = df["state"].values == s
    if mask.sum() > 0:
        actions_in_state = dqn_action_per_property[mask]
        most_common_idx  = np.bincount(actions_in_state).argmax()
        policy_dqn[s]    = ACTIONS[most_common_idx]
    else:
        policy_dqn[s] = policy_vi[s]

print(f"\n  Política DQN: { {s: policy_dqn[s] for s in STATES} }")

# ════════════════════════════════════════════════════════════════════
# PASO 8 — COMPARATIVA DE LOS 3 MÉTODOS RL
# ════════════════════════════════════════════════════════════════════
print(f"\n[PASO 8] Comparativa de políticas RL...")

# Calcular reward esperado por método
def eval_policy(policy_dict, label=""):
    """Evalúa el reward total de una política sobre el dataset completo."""
    total_r = 0
    for i, (s, err) in enumerate(zip(df["state"].values, errors_full)):
        a   = policy_dict[int(s)]
        total_r += calc_reward(err, a)
    return total_r / len(df)

r_vi  = eval_policy(policy_vi,  "Value Iteration")
r_ql  = eval_policy(policy_ql,  "Q-Learning")
r_dqn = eval_policy(policy_dqn, "DQN")

# Concordancia entre métodos
agree_vi_ql  = sum(policy_vi[s] == policy_ql[s]  for s in STATES) / len(STATES) * 100
agree_vi_dqn = sum(policy_vi[s] == policy_dqn[s] for s in STATES) / len(STATES) * 100
agree_ql_dqn = sum(policy_ql[s] == policy_dqn[s] for s in STATES) / len(STATES) * 100

print("\n  ┌──── Comparativa de Métodos RL ─────────────────────────────────┐")
print(f"  │  {'Método':<22} {'Reward medio':>14}  {'Acuerdo con VI':>16}  │")
print("  ├──────────────────────────────────────────────────────────────┤")
print(f"  │  {'MDP Value Iteration':<22} {r_vi:>14.2f}  {'—':>16}  │")
print(f"  │  {'Q-Learning tabular':<22} {r_ql:>14.2f}  {agree_vi_ql:>14.1f}%  │")
print(f"  │  {'Deep Q-Network (DQN)':<22} {r_dqn:>14.2f}  {agree_vi_dqn:>14.1f}%  │")
print("  └──────────────────────────────────────────────────────────────┘")

# Distribución de acciones por método
for met_name, policy_dict in [("VI", policy_vi), ("Q-Learning", policy_ql), ("DQN", policy_dqn)]:
    dist = {a: 0 for a in ACTIONS}
    for s in STATES:
        n = (df["state"] == s).sum()
        dist[policy_dict[s]] += n
    total = sum(dist.values())
    pcts  = {a: dist[a]/total*100 for a in ACTIONS}
    print(f"  {met_name:<12}: APROBAR {pcts['APROBAR']:5.1f}% | "
          f"REVISAR {pcts['REVISAR']:5.1f}% | RECHAZAR {pcts['RECHAZAR']:5.1f}%")

# ════════════════════════════════════════════════════════════════════
# PASO 9 — POLÍTICA FINAL Y ANÁLISIS PROFUNDO
# ════════════════════════════════════════════════════════════════════
print(f"\n[PASO 9] Política óptima final y análisis...")

# Política final: consenso de los 3 métodos (DQN tiene prioridad en desempate)
policy_final = {}
for s in STATES:
    votes = [policy_vi[s], policy_ql[s], policy_dqn[s]]
    counts = {a: votes.count(a) for a in ACTIONS}
    max_votes = max(counts.values())
    candidates = [a for a, v in counts.items() if v == max_votes]
    policy_final[s] = policy_dqn[s] if policy_dqn[s] in candidates else candidates[0]

print("\n  POLÍTICA FINAL π*(s) — Consenso VI + Q-Learning + DQN:")
print(f"  {'Estado':<8} {'VI':<12} {'Q-Learn':<12} {'DQN':<12} {'FINAL':<12}  {'% dataset'}")
print("  " + "-"*65)
for s in STATES:
    n_s  = (df["state"] == s).sum()
    pct  = n_s / len(df) * 100
    mprice = df[df["state"] == s]["SalePrice"].mean()
    print(f"  S{s:<7} {policy_vi[s]:<12} {policy_ql[s]:<12} {policy_dqn[s]:<12} "
          f"{policy_final[s]:<12}  {pct:.1f}% | ${mprice:,.0f}")

# Las 3 reglas de oro
print(f"\n{'='*65}")
print("  LAS 3 REGLAS DE ORO DEL AGENTE SAVI v2.0")
print(f"{'='*65}")
aprobar_states  = [s for s in STATES if policy_final[s] == "APROBAR"]
revisar_states  = [s for s in STATES if policy_final[s] == "REVISAR"]
rechazar_states = [s for s in STATES if policy_final[s] == "RECHAZAR"]

if aprobar_states:
    pct_apr = sum((df["state"]==s).sum() for s in aprobar_states)/len(df)*100
    print(f"\n  REGLA 1 — Estados {aprobar_states} → APROBAR ({pct_apr:.1f}% del dataset)")
    print( "  El modelo es preciso en estos clusters. Automatización directa.")

if revisar_states:
    pct_rev = sum((df["state"]==s).sum() for s in revisar_states)/len(df)*100
    print(f"\n  REGLA 2 — Estados {revisar_states} → REVISAR ({pct_rev:.1f}% del dataset)")
    print( "  Alta heterogeneidad. El costo de equivocarse supera el de revisar.")

if rechazar_states:
    pct_rec = sum((df["state"]==s).sum() for s in rechazar_states)/len(df)*100
    print(f"\n  REGLA 3 — Estados {rechazar_states} → RECHAZAR ({pct_rec:.2f}% del dataset)")
    print( "  Datos insuficientes o outliers. Pedir más información es óptimo.")

print(f"\n{'='*65}")

# ════════════════════════════════════════════════════════════════════
# PASO 10 — VISUALIZACIONES FINALES
# ════════════════════════════════════════════════════════════════════
print(f"\n[PASO 10] Generando visualizaciones...")

_bg     = "#090E1A"
_panel  = "#0F1829"
_card   = "#0C1726"
_cyan   = "#00E5FF"
_green  = "#00FF9C"
_gold   = "#FFB300"
_purple = "#A78BFA"
_red    = "#FF4D6D"
_muted  = "#5A7A9B"
_text   = "#E2EBF7"
_pc     = {"APROBAR": _green, "REVISAR": _gold, "RECHAZAR": _red}

fig = plt.figure(figsize=(20, 14), facecolor=_bg)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)

# ── Subplot 1: Comparativa modelos supervisados ──────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(_panel)
models = ["XGBoost", "LightGBM"]
r2s    = [r2_xgb, r2_lgb]
colors = [_cyan, _purple]
bars = ax1.bar(models, r2s, color=colors, width=0.5, edgecolor=_card, linewidth=1.2)
for b, v in zip(bars, r2s):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.003, f"R²={v:.4f}",
             ha="center", va="bottom", color=_text, fontsize=9, fontweight="bold")
ax1.set_title("Comparativa Supervisado", color=_text, fontsize=10, pad=8)
ax1.set_ylim(0.85, 1.0)
ax1.tick_params(colors=_muted, labelsize=8)
for sp in ax1.spines.values(): sp.set_edgecolor(_card)
ax1.set_facecolor(_panel)

# ── Subplot 2: Silhouette comparativa clustering ─────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(_panel)
_c_labels = ["K-Means", "Jerárquico", "DBSCAN"]
_c_sils   = [sil_km, sil_hier, sil_db if not np.isnan(sil_db) else 0]
_c_cols   = [_cyan, _green, _gold]
bars2 = ax2.bar(_c_labels, _c_sils, color=_c_cols, width=0.5, edgecolor=_card, linewidth=1.2)
for b, v in zip(bars2, _c_sils):
    if v > 0:
        ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", color=_text, fontsize=9)
ax2.set_title("Comparativa Clustering\n(Silhouette)", color=_text, fontsize=10, pad=8)
ax2.tick_params(colors=_muted, labelsize=8)
for sp in ax2.spines.values(): sp.set_edgecolor(_card)

# ── Subplot 3: Reward por método RL ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(_panel)
_rl_names = ["MDP\nValue Iter.", "Q-Learning\nTabular", "Deep Q-\nNetwork"]
_rl_rewards = [r_vi, r_ql, r_dqn]
_rl_cols   = [_gold, _purple, _cyan]
bars3 = ax3.bar(_rl_names, _rl_rewards, color=_rl_cols, width=0.5, edgecolor=_card, linewidth=1.2)
for b, v in zip(bars3, _rl_rewards):
    ax3.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}",
             ha="center", va="bottom", color=_text, fontsize=9)
ax3.set_title("Reward Medio por Método RL", color=_text, fontsize=10, pad=8)
ax3.tick_params(colors=_muted, labelsize=8)
for sp in ax3.spines.values(): sp.set_edgecolor(_card)

# ── Subplot 4: V*(s) barras horizontales ─────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(_panel)
s_labels = [f"S{s}" for s in STATES]
v_vals   = [V_vi[s] for s in STATES]
v_cols   = [_pc[policy_final[s]] for s in STATES]
bars4    = ax4.barh(s_labels, [abs(v) for v in v_vals], color=v_cols,
                    edgecolor=_card, linewidth=1, height=0.55)
for b, v, s in zip(bars4, v_vals, STATES):
    ax4.text(b.get_width() + 3, b.get_y() + b.get_height()/2,
             f"V*={v:.1f}", va="center", color=_text, fontsize=8)
ax4.set_title("V*(s) Value Iteration", color=_text, fontsize=10, pad=8)
ax4.tick_params(colors=_muted, labelsize=8)
for sp in ax4.spines.values(): sp.set_edgecolor(_card)

# ── Subplot 5: Política final por estado ─────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(_panel)
pol_cols = [_pc[policy_final[s]] for s in STATES]
ax5.bar(s_labels, [1]*len(STATES), color=pol_cols, width=0.6, edgecolor=_card, linewidth=1.2)
for i, s in enumerate(STATES):
    pct = (df["state"] == s).sum() / len(df) * 100
    ax5.text(i, 0.5, f"{pct:.1f}%", ha="center", va="center", color=_bg,
             fontsize=10, fontweight="bold")
    ax5.text(i, 1.05, policy_final[s][:3], ha="center", va="bottom",
             color=pol_cols[i], fontsize=9, fontweight="bold")
ax5.set_yticks([])
ax5.set_ylim(0, 1.4)
ax5.set_title("Política Final π*(s)", color=_text, fontsize=10, pad=8)
ax5.tick_params(colors=_muted, labelsize=8)
for sp in ax5.spines.values(): sp.set_edgecolor(_card)
patches = [mpatches.Patch(color=v, label=k) for k, v in _pc.items()]
ax5.legend(handles=patches, loc="upper right", facecolor=_card,
           edgecolor=_card, labelcolor=_text, fontsize=7)

# ── Subplot 6: Q-Learning convergencia ───────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(_panel)
window   = 200
smooth_r = pd.Series(rewards_per_ep).rolling(window).mean()
ax6.plot(smooth_r, color=_gold, linewidth=1.5, label="reward (MA-200)")
ax6.axhline(y=smooth_r.iloc[-100:].mean(), color=_green,
            linewidth=1, linestyle="--", label="media final")
ax6.set_title("Convergencia Q-Learning", color=_text, fontsize=10, pad=8)
ax6.tick_params(colors=_muted, labelsize=8)
ax6.legend(facecolor=_card, edgecolor=_card, labelcolor=_text, fontsize=7)
for sp in ax6.spines.values(): sp.set_edgecolor(_card)

# ── Subplot 7: DQN reward convergencia ───────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor(_panel)
ax7.plot(ep_rewards, color=_cyan, linewidth=1.5, label="reward/época", alpha=0.7)
ma = pd.Series(ep_rewards).rolling(20).mean()
ax7.plot(ma, color=_green, linewidth=2, label="MA-20")
ax7.set_title("Convergencia DQN", color=_text, fontsize=10, pad=8)
ax7.tick_params(colors=_muted, labelsize=8)
ax7.legend(facecolor=_card, edgecolor=_card, labelcolor=_text, fontsize=7)
for sp in ax7.spines.values(): sp.set_edgecolor(_card)

# ── Subplot 8: DQN loss ──────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor(_panel)
ax8.plot(ep_losses, color=_red, linewidth=1.5, alpha=0.8)
ax8.set_title("DQN Loss (Huber)", color=_text, fontsize=10, pad=8)
ax8.tick_params(colors=_muted, labelsize=8)
for sp in ax8.spines.values(): sp.set_edgecolor(_card)

# ── Subplot 9: Distribución de acciones por método ───────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor(_panel)
_method_names = ["VI", "Q-Learn", "DQN"]
_method_pols  = [policy_vi, policy_ql, policy_dqn]
bar_width = 0.25
x_pos     = np.arange(len(_method_names))
for i, action in enumerate(ACTIONS):
    vals = []
    for pol in _method_pols:
        total_in_action = sum((df["state"]==s).sum() for s in STATES if pol[s] == action)
        vals.append(total_in_action / len(df) * 100)
    ax9.bar(x_pos + i * bar_width, vals, bar_width,
            color=_pc[action], label=action, edgecolor=_card, linewidth=0.8)
ax9.set_xticks(x_pos + bar_width)
ax9.set_xticklabels(_method_names, color=_muted, fontsize=8)
ax9.set_title("Distribución de Acciones\npor Método RL", color=_text, fontsize=10, pad=8)
ax9.tick_params(colors=_muted, labelsize=7)
ax9.legend(facecolor=_card, edgecolor=_card, labelcolor=_text, fontsize=7)
for sp in ax9.spines.values(): sp.set_edgecolor(_card)

fig.suptitle(
    "SAVI v2.0 — Parcial Final ML · Julian Rincon · Valeria Larea · "
    "Nicolás Garzón · Juan Niño · Universidad Sergio Arboleda 2026",
    color=_muted, fontsize=9, y=0.98)

plt.savefig("SAVI_v2_resultados.png", dpi=150, bbox_inches="tight",
            facecolor=_bg, edgecolor="none")
print("  Gráfica guardada: SAVI_v2_resultados.png")
plt.close()

# ════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  RESUMEN FINAL — SAVI v2.0")
print(f"{'='*65}")
print(f"\n  NO SUPERVISADO")
print(f"    K-Means k=6        Silhouette = {sil_km:.4f}  ← ELEGIDO")
print(f"    Jerárquico Ward    Silhouette = {sil_hier:.4f}")
print(f"\n  SUPERVISADO")
print(f"    XGBoost            R²={r2_xgb:.4f}  MAE=${mae_xgb:,.0f}  ← ELEGIDO")
print(f"    LightGBM           R²={r2_lgb:.4f}  MAE=${mae_lgb:,.0f}")
print(f"\n  REINFORCEMENT LEARNING")
print(f"    MDP Value Iter.    Convergencia: {iter_vi} iters  Reward: {r_vi:.1f}")
print(f"    Q-Learning         {EPISODES_QL:,} episodios     Reward: {r_ql:.1f}")
print(f"    DQN (PyTorch)      {EPOCHS_DQN} épocas         Reward: {r_dqn:.1f}")
print(f"\n  POLÍTICA FINAL:")
for s in STATES:
    n_s   = (df["state"] == s).sum()
    pct   = n_s / len(df) * 100
    print(f"    S{s}: {policy_final[s]:<10}  ({pct:.1f}% del mercado · "
          f"{n_s:,} predios · ${df[df['state']==s]['SalePrice'].mean():,.0f} precio medio)")

aprobar_pct  = sum((df["state"]==s).sum() for s in STATES if policy_final[s]=="APROBAR")  / len(df) * 100
revisar_pct  = sum((df["state"]==s).sum() for s in STATES if policy_final[s]=="REVISAR")  / len(df) * 100
rechazar_pct = sum((df["state"]==s).sum() for s in STATES if policy_final[s]=="RECHAZAR") / len(df) * 100
print(f"\n  AUTOMATIZACIÓN FINAL:")
print(f"    APROBAR  {aprobar_pct:.1f}% — valuación directa sin intervención")
print(f"    REVISAR  {revisar_pct:.1f}% — revisión humana selectiva")
print(f"    RECHAZAR {rechazar_pct:.2f}% — solicitar datos adicionales")

print(f"\n{'='*65}")
print("  SAVI v2.0 completado exitosamente.")
print("  Julian Rincon · Valeria Larea · Nicolás Garzón · Juan Niño")
print("  Universidad Sergio Arboleda · Corte 3 · 2026")
print(f"{'='*65}\n")
