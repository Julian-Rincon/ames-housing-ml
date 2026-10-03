# -*- coding: utf-8 -*-
"""
SAVI v2 · Cloud — utilidades compartidas entre el motor CPU (EC2) y el motor GPU (SageMaker).

Este módulo SOLO depende de la librería estándar + numpy para poder importarse
tanto en el EC2 (Amazon Linux + venv propio) como en el contenedor preconstruido
de PyTorch de SageMaker sin instalar nada extra.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL (antes vivía en el monolito como constantes sueltas)
# ════════════════════════════════════════════════════════════════════
SEED = 42

GAMMA = 0.95          # Factor de descuento RL
THETA = 1e-4          # Umbral de convergencia Value Iteration
ALPHA_QL = 0.1        # Learning rate inicial Q-Learning tabular
QL_ALPHA_HALF = 2000  # visitas a (s,a) con las que α cae a la mitad
EPSILON_0 = 1.0       # Exploración inicial
EPS_DECAY = 0.995     # Decaimiento epsilon (por EPISODIO / ÉPOCA, no por paso)
EPS_MIN = 0.05        # Epsilon mínimo
EPISODES_QL = 8000    # Episodios Q-Learning
N_CLUSTERS = 6        # Estados discretos del MDP (K-Means)

# División de las ventas para evaluar sin sesgo: los agentes aprenden en TRAIN,
# la regla de decisión final se elige en VAL y el resultado se reporta UNA vez en TEST.
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
SPLIT_FRACS = (0.70, 0.15, 0.15)

ACTIONS = ["APROBAR", "REVISAR", "RECHAZAR"]
N_ACTIONS = len(ACTIONS)
A_APROBAR, A_REVISAR, A_RECHAZAR = 0, 1, 2

# Nombres de artefactos que el EC2 deja en S3 y SageMaker consume.
# Mantenerlos aquí evita que los dos motores se desincronicen.
ARTIFACTS = {
    "train_states": "tensors/train_states.npy",        # (N_train, D) float32, MinMax [0,1]
    "train_errors": "tensors/train_errors.npy",        # (N_train,) error relativo out-of-fold
    "train_next": "tensors/train_next_idx.npy",        # (N_train,) índice del siguiente predio (orden cronológico)
    "train_clusters": "tensors/train_clusters.npy",    # (N_train,) estado discreto K-Means
    "train_split": "tensors/train_split.npy",          # (N_train,) 0=train 1=val 2=test (estratificado)
    "portfolio_states": "tensors/portfolio_states.npy",  # (N_port, D) cartera 2024 a decidir
    "portfolio_clusters": "tensors/portfolio_clusters.npy",
    "portfolio_meta": "portfolio/portfolio_meta.csv",  # PID, valor AVM, avalúo, etc.
    "policy_vi": "rl/policy_vi.json",
    "policy_ql": "rl/policy_ql.json",
    "q_table_ql": "rl/q_table_ql.json",
    "mdp": "rl/mdp_tables.json",
    "metrics_cpu": "metrics/metrics_cpu.json",
    "success": "_SUCCESS.json",                        # SIEMPRE el último archivo escrito
}


# ════════════════════════════════════════════════════════════════════
# LOGGING → stdout (CloudWatch captura stdout en EC2-agent, Lambda y SageMaker)
# ════════════════════════════════════════════════════════════════════
def get_logger(name: str = "savi", log_file: str | None = None) -> logging.Logger:
    """Logger con formato uniforme a nivel INFO; opcionalmente duplica a archivo."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def set_seed(seed: int = SEED) -> None:
    """Semilla global para reproducibilidad (python, numpy y torch si existe)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ════════════════════════════════════════════════════════════════════
# FUNCIÓN DE RECOMPENSA (idéntica en valores al monolito original)
# ════════════════════════════════════════════════════════════════════
def calc_reward(error_pct: float, action: int | str) -> float:
    """
    Recompensa económica del agente SAVI.

      APROBAR : error<10% → +200 | error<25% → -500 | error≥25% → -2000
      REVISAR : error<10% → -150 (tiempo perdido) | si no → -50 (la revisión aportó)
      RECHAZAR: error>20% → +50 (correcto pedir datos) | si no → -200
    """
    if isinstance(action, str):
        action = ACTIONS.index(action)
    if action == A_APROBAR:
        return 200.0 if error_pct < 0.10 else (-500.0 if error_pct < 0.25 else -2000.0)
    if action == A_REVISAR:
        return -150.0 if error_pct < 0.10 else -50.0
    if action == A_RECHAZAR:
        return 50.0 if error_pct > 0.20 else -200.0
    return 0.0


def reward_matrix(errors: np.ndarray) -> np.ndarray:
    """Versión vectorizada: devuelve (N, 3) con la recompensa de cada acción por predio."""
    e = np.asarray(errors, dtype=np.float64)
    r = np.empty((e.shape[0], N_ACTIONS), dtype=np.float32)
    r[:, A_APROBAR] = np.where(e < 0.10, 200.0, np.where(e < 0.25, -500.0, -2000.0))
    r[:, A_REVISAR] = np.where(e < 0.10, -150.0, -50.0)
    r[:, A_RECHAZAR] = np.where(e > 0.20, 50.0, -200.0)
    return r


# ════════════════════════════════════════════════════════════════════
# EVALUACIÓN SIN SESGO: split estratificado, cadena cronológica, bootstrap
# ════════════════════════════════════════════════════════════════════
def stratified_split(strata: np.ndarray, fracs=SPLIT_FRACS, seed: int = SEED) -> np.ndarray:
    """
    Asigna cada fila a TRAIN/VAL/TEST respetando la proporción dentro de cada estrato
    (cluster K-Means). Estratos con <3 filas van completos a TRAIN.
    """
    rng = np.random.default_rng(seed)
    split = np.full(len(strata), SPLIT_TRAIN, dtype=np.int64)
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        if len(idx) < 3:
            continue
        n_val = max(1, int(round(fracs[1] * len(idx))))
        n_test = max(1, int(round(fracs[2] * len(idx))))
        split[idx[:n_val]] = SPLIT_VAL
        split[idx[n_val:n_val + n_test]] = SPLIT_TEST
    return split


def next_within_split(split: np.ndarray) -> np.ndarray:
    """
    Para filas ya ordenadas cronológicamente: índice de la SIGUIENTE fila del mismo
    subconjunto. La última de cada subconjunto apunta a sí misma (done=1 en el DQN),
    así ninguna transición cruza de TRAIN a VAL/TEST.
    """
    nxt = np.arange(len(split))
    for g in np.unique(split):
        idx = np.flatnonzero(split == g)
        nxt[idx[:-1]] = idx[1:]
    return nxt


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Media e intervalo de confianza bootstrap (percentiles) de `values`."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


# ════════════════════════════════════════════════════════════════════
# POLÍTICAS
# ════════════════════════════════════════════════════════════════════
def consensus_policy(policy_vi: dict, policy_ql: dict, policy_dqn: dict) -> dict:
    """
    Votación mayoritaria VI + QL + DQN por estado. En empate (1-1-1) gana el DQN,
    igual que en el monolito original.
    """
    final = {}
    for s in policy_vi:
        votes = [policy_vi[s], policy_ql[s], policy_dqn[s]]
        counts = {a: votes.count(a) for a in ACTIONS}
        top = max(counts.values())
        candidates = [a for a, v in counts.items() if v == top]
        final[s] = policy_dqn[s] if policy_dqn[s] in candidates else candidates[0]
    return final


def eval_policy(policy: dict, clusters: np.ndarray, errors: np.ndarray) -> float:
    """Recompensa media de una política por-estado {int: acción} sobre (cluster, error)."""
    R = reward_matrix(errors)
    acts = np.array([ACTIONS.index(policy[int(s)]) for s in clusters])
    return float(R[np.arange(len(acts)), acts].mean())


def load_policy(path: str | Path) -> dict:
    """Carga una política desde JSON devolviendo claves int (JSON las guarda como str)."""
    return {int(k): v for k, v in load_json(path).items()}


# ════════════════════════════════════════════════════════════════════
# JSON (las claves de dict en JSON siempre son str → normalizamos)
# ════════════════════════════════════════════════════════════════════
def _to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_to_jsonable), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
