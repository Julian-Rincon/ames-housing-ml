# -*- coding: utf-8 -*-
"""Tests for savi_gpu_sagemaker.py: ReplayBuffer, DoubleDQNAgent, consensus, checkpoints, e2e."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import savi_gpu_sagemaker as G
import utils as U

LOCAL_RUN = ROOT / "out" / "local_run"


def _hp(**overrides):
    base = dict(lr=1e-3, epochs=3, buffer=10, batch_size=1, gamma=0.9, target_update=100)
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ════════════════════════════════════════════════════════════════════
# ReplayBuffer — circular behaviour
# ════════════════════════════════════════════════════════════════════
class TestReplayBuffer:
    def test_len_caps_at_capacity(self):
        buf = G.ReplayBuffer(capacity=3, state_dim=2)
        for i in range(5):
            buf.push([i, i], 0, float(i), [i + 1, i + 1], 0.0)
        assert len(buf) == 3

    def test_overwrites_oldest_entries_circularly(self):
        buf = G.ReplayBuffer(capacity=3, state_dim=1)
        for i in range(5):  # writes indices 0,1,2,0,1 (wrap-around)
            buf.push([i], 0, float(i), [i + 1], 0.0)
        # after 5 pushes into capacity 3: slot0 <- i=3, slot1 <- i=4, slot2 <- i=2 (last full write)
        np.testing.assert_allclose(buf.r, [3.0, 4.0, 2.0])

    def test_sample_shapes(self):
        buf = G.ReplayBuffer(capacity=5, state_dim=2)
        for i in range(5):
            buf.push([i, i], i % 3, float(i), [i + 1, i + 1], float(i == 4))
        s, a, r, ns, d = buf.sample(4)
        assert s.shape == (4, 2)
        assert ns.shape == (4, 2)
        assert a.shape == (4,)
        assert r.shape == (4,)
        assert d.shape == (4,)


# ════════════════════════════════════════════════════════════════════
# DoubleDQNAgent.learn — online argmax + target evaluation
# ════════════════════════════════════════════════════════════════════
class _FakeQ(nn.Module):
    """Q(s) = [10,20,30]; Q_online(s') = [5,1,1] -> argmax a* = 0 (APROBAR)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        rows = []
        for row in x:
            if torch.allclose(row, torch.zeros_like(row)):
                rows.append([10.0, 20.0, 30.0])
            else:
                rows.append([5.0, 1.0, 1.0])
        return self.scale * torch.tensor(rows, dtype=torch.float32)


class _FakeTarget(nn.Module):
    """Q_target(s') = [2,9,2]: if Double-DQN picks a*=0 -> value 2 (NOT the max, 9)."""

    def forward(self, x):
        rows = [[2.0, 9.0, 2.0] for _ in x]
        return torch.tensor(rows, dtype=torch.float32)


class TestDoubleDQNLearn:
    def test_target_uses_online_argmax_not_target_max(self):
        hp = _hp(batch_size=1, gamma=0.9, target_update=10_000)
        agent = G.DoubleDQNAgent(state_dim=2, hp=hp, device=torch.device("cpu"))
        agent.q = _FakeQ()
        agent.target = _FakeTarget()
        agent.opt = torch.optim.Adam(agent.q.parameters(), lr=hp.lr)

        # s = [0,0] (marker for Q(s) branch), a=1 (REVISAR) -> q_sa = 20
        # ns = [1,1] (marker for Q_online(s') branch) -> a* = argmax([5,1,1]) = 0
        # Q_target(ns)[a*=0] = 2  (classic DQN would instead take max([2,9,2]) = 9)
        agent.buffer.push([0.0, 0.0], 1, 3.0, [1.0, 1.0], 0.0)

        captured = {}
        orig_loss = agent.loss_fn

        def spy_loss(q_sa, y):
            captured["q_sa"] = q_sa.detach().clone()
            captured["y"] = y.detach().clone()
            return orig_loss(q_sa, y)

        agent.loss_fn = spy_loss
        agent.learn()

        expected_y = 3.0 + 0.9 * 2.0  # r + gamma * Q_target(s', a*) * (1-d)
        wrong_classic_dqn_y = 3.0 + 0.9 * 9.0  # r + gamma * max_a Q_target(s', a)

        assert captured["y"].item() == pytest.approx(expected_y)
        assert captured["y"].item() != pytest.approx(wrong_classic_dqn_y)
        assert captured["q_sa"].item() == pytest.approx(20.0)

    def test_learn_returns_none_below_batch_size(self):
        hp = _hp(batch_size=5)
        agent = G.DoubleDQNAgent(state_dim=2, hp=hp, device=torch.device("cpu"))
        agent.buffer.push([0.0, 0.0], 0, 1.0, [0.0, 0.0], 0.0)
        assert agent.learn() is None


# ════════════════════════════════════════════════════════════════════
# per_property_consensus
# ════════════════════════════════════════════════════════════════════
class TestPerPropertyConsensus:
    def test_majority_wins(self):
        pol_vi = {0: "APROBAR"}
        pol_ql = {0: "APROBAR"}
        dqn_actions = np.array([U.A_RECHAZAR])
        clusters = np.array([0])
        out = G.per_property_consensus(dqn_actions, clusters, pol_vi, pol_ql)
        assert out[0] == U.A_APROBAR

    def test_tie_breaks_to_dqn(self):
        pol_vi = {0: "APROBAR"}
        pol_ql = {0: "REVISAR"}
        dqn_actions = np.array([U.A_RECHAZAR])
        clusters = np.array([0])
        out = G.per_property_consensus(dqn_actions, clusters, pol_vi, pol_ql)
        assert out[0] == U.A_RECHAZAR

    def test_vectorized_over_multiple_properties(self):
        pol_vi = {0: "APROBAR", 1: "RECHAZAR"}
        pol_ql = {0: "APROBAR", 1: "RECHAZAR"}
        dqn_actions = np.array([U.A_REVISAR, U.A_APROBAR])
        clusters = np.array([0, 1])
        out = G.per_property_consensus(dqn_actions, clusters, pol_vi, pol_ql)
        assert out[0] == U.A_APROBAR   # VI+QL majority
        assert out[1] == U.A_RECHAZAR  # VI+QL majority


# ════════════════════════════════════════════════════════════════════
# checkpoint save/load roundtrip
# ════════════════════════════════════════════════════════════════════
class TestCheckpointRoundtrip:
    def test_resumes_epoch_and_epsilon(self, tmp_path):
        hp = _hp(epochs=9)
        agent = G.DoubleDQNAgent(state_dim=3, hp=hp, device=torch.device("cpu"))
        agent.epsilon = 0.37
        agent.steps = 123
        curves = {"reward": [1.0, 2.0], "loss": [0.5], "epsilon": [1.0, 0.5]}

        ckpt_dir = tmp_path / "ckpt"
        G.save_ckpt(ckpt_dir, agent, epoch=5, curves=curves)

        agent2 = G.DoubleDQNAgent(state_dim=3, hp=hp, device=torch.device("cpu"))
        start_epoch, loaded_curves = G.load_ckpt(ckpt_dir, agent2)

        assert start_epoch == 5
        assert loaded_curves == curves
        assert agent2.epsilon == pytest.approx(0.37)
        assert agent2.steps == 123
        for p1, p2 in zip(agent.q.state_dict().values(), agent2.q.state_dict().values()):
            torch.testing.assert_close(p1, p2)

    def test_load_ckpt_missing_file_returns_defaults(self, tmp_path):
        hp = _hp()
        agent = G.DoubleDQNAgent(state_dim=3, hp=hp, device=torch.device("cpu"))
        start_epoch, curves = G.load_ckpt(tmp_path / "nonexistent", agent)
        assert start_epoch == 0
        assert curves == {"reward": [], "loss": [], "epsilon": []}


# ════════════════════════════════════════════════════════════════════
# END-TO-END main() on artifacts produced by the CPU run
# ════════════════════════════════════════════════════════════════════
@pytest.mark.slow
@pytest.mark.skipif(not (LOCAL_RUN / U.ARTIFACTS["train_states"]).exists(),
                     reason="out/local_run artifacts not present (run the CPU e2e test first)")
def test_main_end_to_end_small_epochs(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    ckpt_dir = tmp_path / "ckpt"
    argv = [
        "savi_gpu_sagemaker.py",
        "--epochs", "2",
        "--data-dir", str(LOCAL_RUN),
        "--model-dir", str(model_dir),
        "--checkpoint-dir", str(ckpt_dir),
        "--run-id", "test-gpu-e2e",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    G.main()

    assert (model_dir / "dqn_model.pt").exists()
    assert (model_dir / "policy_dqn.json").exists()
    assert (model_dir / "policy_final.json").exists()
    assert (model_dir / "metrics_gpu.json").exists()
    assert (model_dir / "portfolio_decisions.csv").exists()

    decisions = pd.read_csv(model_dir / "portfolio_decisions.csv")
    meta = pd.read_csv(LOCAL_RUN / U.ARTIFACTS["portfolio_meta"])
    assert len(decisions) == len(meta)
