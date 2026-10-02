# -*- coding: utf-8 -*-
"""
SAVI v2 · Cloud — MOTOR GPU (SageMaker Training Job, contenedor PyTorch preconstruido).

Contrato SageMaker (script mode):
  · Hiperparámetros → argumentos CLI (--epochs 150 --batch-size 128 ...)
  · Canal de datos "processed" → /opt/ml/input/data/processed  (env SM_CHANNEL_PROCESSED)
      contiene los artefactos del motor CPU (tensores, políticas VI/QL, metadatos de cartera)
  · Modelo final → /opt/ml/model (env SM_MODEL_DIR) → SageMaker lo sube como model.tar.gz
  · Checkpoints  → /opt/ml/checkpoints (Spot: si la instancia se interrumpe, se reanuda)

Flujo:
  1. Carga estados continuos + errores out-of-fold del AVM.
  2. Entrena un Double DQN (online elige a', target evalúa Q(s',a')).
  3. Deriva la política DQN por estado y por predio.
  4. Consenso VI + QL + DQN (votación mayoritaria; empate → DQN).
  5. Guarda modelo, políticas, decisiones de la cartera 2024 y métricas.

Prueba local:
  SM_CHANNEL_PROCESSED=out/local_run SM_MODEL_DIR=out/model python savi_gpu_sagemaker.py --epochs 3
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils as U

log = U.get_logger("savi.gpu")


# ════════════════════════════════════════════════════════════════════
# RED, BUFFER Y AGENTE
# ════════════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    """FC(128) → BN → ReLU → Dropout → FC(64) → ReLU → FC(3)  (misma arquitectura del monolito)."""

    def __init__(self, state_dim: int, action_dim: int = U.N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Buffer circular de experiencias guardado como arrays (muestreo sin bucles Python)."""

    def __init__(self, capacity: int, state_dim: int):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.ns = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def push(self, s, a, r, ns, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.ns[self.i], self.d[self.i] = s, a, r, ns, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k: int):
        idx = np.random.randint(0, self.n, k)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]

    def __len__(self):
        return self.n


class DoubleDQNAgent:
    """
    Double DQN real (van Hasselt et al., 2016):
        a* = argmax_a Q_online(s', a)          ← la red online ELIGE
        y  = r + γ · Q_target(s', a*) · (1-d)  ← la red target EVALÚA
    El monolito usaba max_a Q_target(s', a) (DQN clásico con target net), que sobreestima Q.
    """

    def __init__(self, state_dim: int, hp, device: torch.device):
        self.hp, self.device = hp, device
        self.q = QNetwork(state_dim).to(device)
        self.target = QNetwork(state_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=hp.lr)
        self.sched = optim.lr_scheduler.StepLR(self.opt, step_size=max(1, hp.epochs // 3), gamma=0.5)
        self.buffer = ReplayBuffer(hp.buffer, state_dim)
        self.loss_fn = nn.SmoothL1Loss()
        self.epsilon = U.EPSILON_0
        self.steps = 0

    @torch.no_grad()
    def act_batch(self, states: np.ndarray) -> np.ndarray:
        """ε-greedy vectorizado sobre un bloque de estados."""
        self.q.eval()
        greedy = self.q(torch.from_numpy(states).to(self.device)).argmax(1).cpu().numpy()
        self.q.train()
        explore = np.random.rand(len(states)) < self.epsilon
        return np.where(explore, np.random.randint(0, U.N_ACTIONS, len(states)), greedy)

    def learn(self) -> float | None:
        if len(self.buffer) < self.hp.batch_size:
            return None
        s, a, r, ns, d = (torch.from_numpy(x).to(self.device) for x in self.buffer.sample(self.hp.batch_size))
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            self.q.eval()
            a_star = self.q(ns).argmax(1, keepdim=True)           # online elige
            self.q.train()
            q_next = self.target(ns).gather(1, a_star).squeeze(1)  # target evalúa
            y = r + self.hp.gamma * q_next * (1 - d)
        loss = self.loss_fn(q_sa, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.hp.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

    @torch.no_grad()
    def greedy_actions(self, states: np.ndarray, batch: int = 8192) -> np.ndarray:
        self.q.eval()
        out = [self.q(torch.from_numpy(states[i:i + batch]).to(self.device)).argmax(1).cpu().numpy()
               for i in range(0, len(states), batch)]
        return np.concatenate(out)


# ════════════════════════════════════════════════════════════════════
# CHECKPOINTS (Spot)
# ════════════════════════════════════════════════════════════════════
def save_ckpt(path: Path, agent: DoubleDQNAgent, epoch: int, curves: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "q": agent.q.state_dict(), "target": agent.target.state_dict(),
                "opt": agent.opt.state_dict(), "sched": agent.sched.state_dict(),
                "epsilon": agent.epsilon, "steps": agent.steps, "curves": curves},
               path / "dqn_ckpt.pt")


def load_ckpt(path: Path, agent: DoubleDQNAgent):
    f = path / "dqn_ckpt.pt"
    if not f.exists():
        return 0, {"reward": [], "loss": [], "epsilon": []}
    ck = torch.load(f, map_location=agent.device, weights_only=False)
    agent.q.load_state_dict(ck["q"])
    agent.target.load_state_dict(ck["target"])
    agent.opt.load_state_dict(ck["opt"])
    agent.sched.load_state_dict(ck["sched"])
    agent.epsilon, agent.steps = ck["epsilon"], ck["steps"]
    log.info("Checkpoint encontrado → reanudando desde la época %d (interrupción Spot)", ck["epoch"])
    return ck["epoch"], ck["curves"]


# ════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ════════════════════════════════════════════════════════════════════
def train_dqn(hp, data: dict, device: torch.device, ckpt_dir: Path):
    S, nxt, Rm = data["train_states"], data["train_next"], U.reward_matrix(data["train_errors"])
    n, dim = S.shape
    last = n - 1
    agent = DoubleDQNAgent(dim, hp, device)
    start, curves = load_ckpt(ckpt_dir, agent)
    # ε por ÉPOCA: llega a EPS_MIN al ~60 % del entrenamiento (antes decaía por paso y
    # quedaba en 0.05 tras ~600 de ~3M pasos → el agente prácticamente no exploraba)
    eps_decay = hp.eps_decay or U.EPS_MIN ** (1.0 / max(1, int(0.6 * hp.epochs)))
    log.info("DQN: %d estados × %dD | %d épocas | batch %d | lr %g | ε-decay/época %.4f | device %s",
             n, dim, hp.epochs, hp.batch_size, hp.lr, eps_decay, device)
    t0 = time.time()
    for epoch in range(start, hp.epochs):
        perm = np.random.permutation(n)
        ep_r, losses = 0.0, []
        for c in range(0, n, hp.act_chunk):
            idx = perm[c:c + hp.act_chunk]
            acts = agent.act_batch(S[idx])
            rewards = Rm[idx, acts]
            for i, a, r in zip(idx, acts, rewards):
                agent.buffer.push(S[i], a, r, S[nxt[i]], float(i == last))
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
            ep_r += float(rewards.sum())
        agent.epsilon = max(U.EPS_MIN, agent.epsilon * eps_decay)
        agent.sched.step()
        curves["reward"].append(ep_r / n)
        curves["loss"].append(float(np.mean(losses)) if losses else 0.0)
        curves["epsilon"].append(agent.epsilon)
        if (epoch + 1) % max(1, hp.epochs // 10) == 0 or epoch == hp.epochs - 1:
            log.info("  Época %3d/%d | ε=%.3f | reward=%.1f | loss=%.4f | %.0fs", epoch + 1, hp.epochs,
                     agent.epsilon, curves["reward"][-1], curves["loss"][-1], time.time() - t0)
        if (epoch + 1) % hp.ckpt_every == 0:
            save_ckpt(ckpt_dir, agent, epoch + 1, curves)
    return agent, curves


def per_cluster_policy(actions: np.ndarray, clusters: np.ndarray, fallback: dict) -> dict:
    """Acción mayoritaria del DQN dentro de cada estado discreto (como el monolito)."""
    pol = {}
    for s in range(U.N_CLUSTERS):
        m = clusters == s
        pol[s] = U.ACTIONS[int(np.bincount(actions[m], minlength=U.N_ACTIONS).argmax())] if m.any() else fallback[s]
    return pol


def per_property_consensus(dqn_actions, clusters, pol_vi, pol_ql) -> np.ndarray:
    """Consenso por predio: VI[s] + QL[s] + DQN(predio) — votación mayoritaria, empate → DQN."""
    out = np.empty(len(dqn_actions), dtype=np.int64)
    for k, (a_dqn, s) in enumerate(zip(dqn_actions, clusters)):
        votes = [U.ACTIONS.index(pol_vi[int(s)]), U.ACTIONS.index(pol_ql[int(s)]), int(a_dqn)]
        counts = np.bincount(votes, minlength=U.N_ACTIONS)
        top = counts.max()
        out[k] = a_dqn if counts[a_dqn] == top else int(np.argmax(counts))
    return out


def try_plot(curves: dict, path: Path) -> None:
    """Curvas de entrenamiento (opcional: el contenedor puede no traer matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib no disponible → se omite la figura")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, k in zip(ax, ["reward", "loss", "epsilon"]):
        a.plot(curves[k])
        a.set_title(f"DQN · {k}")
        a.set_xlabel("época")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(description="SAVI v2 · Double DQN (SageMaker)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=U.GAMMA)
    ap.add_argument("--target-update", type=int, default=200)
    ap.add_argument("--buffer", type=int, default=20_000)
    ap.add_argument("--eps-decay", type=float, default=0.0, help="0 = automático")
    ap.add_argument("--act-chunk", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--run-id", type=str, default="local")
    # Rutas estándar SageMaker (con valores por defecto para pruebas locales)
    ap.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_PROCESSED", "/opt/ml/input/data/processed"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    ap.add_argument("--checkpoint-dir", default=os.environ.get("SAVI_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    return ap.parse_args()


def main() -> None:
    hp = parse_args()
    U.set_seed()
    t0 = time.time()
    data_dir, model_dir = Path(hp.data_dir), Path(hp.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("═" * 60)
    log.info("SAVI v2 · MOTOR GPU · run_id=%s | device=%s %s", hp.run_id, device,
             torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    log.info("Canal de datos: %s → %s", data_dir, sorted(p.name for p in data_dir.iterdir()))

    A = U.ARTIFACTS
    data = {k: np.load(data_dir / A[k]) for k in
            ["train_states", "train_errors", "train_next", "train_clusters", "portfolio_states", "portfolio_clusters"]}
    pol_vi, pol_ql = U.load_policy(data_dir / A["policy_vi"]), U.load_policy(data_dir / A["policy_ql"])
    log.info("Políticas recibidas del EC2 → VI=%s | QL=%s", pol_vi, pol_ql)

    # ── 1-2: entrenamiento Double DQN ──
    agent, curves = train_dqn(hp, data, device, Path(hp.checkpoint_dir))

    # ── 3: política DQN ──
    tr_act = agent.greedy_actions(data["train_states"])
    pf_act = agent.greedy_actions(data["portfolio_states"])
    pol_dqn = per_cluster_policy(pf_act, data["portfolio_clusters"], pol_vi)
    log.info("Política DQN (por estado) → %s", pol_dqn)

    # ── 4: consenso ──
    pol_final = U.consensus_policy(pol_vi, pol_ql, pol_dqn)
    tr_cons = per_property_consensus(tr_act, data["train_clusters"], pol_vi, pol_ql)
    pf_cons = per_property_consensus(pf_act, data["portfolio_clusters"], pol_vi, pol_ql)

    Rm = U.reward_matrix(data["train_errors"])
    rows = np.arange(len(tr_act))
    cl, err = data["train_clusters"], data["train_errors"]
    metrics = {
        "run_id": hp.run_id, "device": str(device), "epochs": hp.epochs,
        "reward_vi": U.eval_policy(pol_vi, cl, err), "reward_ql": U.eval_policy(pol_ql, cl, err),
        "reward_dqn_state": U.eval_policy(pol_dqn, cl, err),
        "reward_dqn_property": float(Rm[rows, tr_act].mean()),
        "reward_consensus_state": U.eval_policy(pol_final, cl, err),
        "reward_consensus_property": float(Rm[rows, tr_cons].mean()),
        "reward_oracle": float(Rm.max(axis=1).mean()),
        "policy_vi": pol_vi, "policy_ql": pol_ql, "policy_dqn": pol_dqn, "policy_final": pol_final,
        "portfolio_action_share": {U.ACTIONS[a]: float(np.mean(pf_cons == a)) for a in range(U.N_ACTIONS)},
        "train_seconds": round(time.time() - t0, 1),
    }
    for k in [k for k in metrics if k.startswith("reward_")]:
        log.info("  %-28s %8.2f", k, metrics[k])
    log.info("POLÍTICA FINAL (consenso por estado) → %s", pol_final)
    log.info("Cartera 2024 → %s", {k: f"{100 * v:.1f}%" for k, v in metrics["portfolio_action_share"].items()})

    # ── 5: artefactos del modelo final ──
    torch.save({"state_dict": agent.q.state_dict(), "state_dim": int(data["train_states"].shape[1]),
                "actions": U.ACTIONS, "hyperparameters": vars(hp)}, model_dir / "dqn_model.pt")
    U.save_json(pol_dqn, model_dir / "policy_dqn.json")
    U.save_json(pol_final, model_dir / "policy_final.json")
    U.save_json(metrics, model_dir / "metrics_gpu.json")
    U.save_json(curves, model_dir / "training_curves.json")
    try_plot(curves, model_dir / "dqn_training.png")

    meta_path = data_dir / A["portfolio_meta"]
    with open(meta_path, newline="", encoding="utf-8") as fin, \
            open(model_dir / "portfolio_decisions.csv", "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        meta_rows = list(reader)
        if len(meta_rows) != len(pf_act):  # zip() truncaría en silencio si se desincronizan
            raise ValueError(f"portfolio_meta ({len(meta_rows)}) ≠ portfolio_states ({len(pf_act)})")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames + ["dqn_action", "final_action"])
        writer.writeheader()
        for row, a_dqn, a_fin in zip(meta_rows, pf_act, pf_cons):
            row.update(dqn_action=U.ACTIONS[int(a_dqn)], final_action=U.ACTIONS[int(a_fin)])
            writer.writerow(row)
    log.info("Modelo y decisiones guardados en %s (SageMaker → model.tar.gz en S3)", model_dir)
    log.info("MOTOR GPU completado en %.1fs", time.time() - t0)


if __name__ == "__main__":
    random.seed(U.SEED)
    main()
