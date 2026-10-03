# -*- coding: utf-8 -*-
"""Tests for utils.py: reward function, policies, JSON round-trips."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils as U


# ════════════════════════════════════════════════════════════════════
# calc_reward / reward_matrix
# ════════════════════════════════════════════════════════════════════
class TestCalcReward:
    @pytest.mark.parametrize("action", [U.A_APROBAR, "APROBAR"])
    def test_aprobar_below_10(self, action):
        assert U.calc_reward(0.05, action) == 200.0

    def test_aprobar_at_boundary_010_is_not_below(self):
        # error_pct == 0.10 is NOT < 0.10 -> falls into the 0.25 branch
        assert U.calc_reward(0.10, U.A_APROBAR) == -500.0

    def test_aprobar_just_below_010(self):
        assert U.calc_reward(0.0999999, U.A_APROBAR) == 200.0

    def test_aprobar_between_010_and_025(self):
        assert U.calc_reward(0.15, U.A_APROBAR) == -500.0

    def test_aprobar_at_boundary_025_is_not_below(self):
        assert U.calc_reward(0.25, U.A_APROBAR) == -2000.0

    def test_aprobar_just_below_025(self):
        assert U.calc_reward(0.2499999, U.A_APROBAR) == -500.0

    def test_aprobar_above_025(self):
        assert U.calc_reward(0.9, U.A_APROBAR) == -2000.0

    def test_revisar_below_010(self):
        assert U.calc_reward(0.05, U.A_REVISAR) == -150.0

    def test_revisar_at_010_boundary(self):
        assert U.calc_reward(0.10, U.A_REVISAR) == -50.0

    def test_revisar_above_010(self):
        assert U.calc_reward(0.50, U.A_REVISAR) == -50.0

    def test_rechazar_above_020(self):
        assert U.calc_reward(0.21, U.A_RECHAZAR) == 50.0

    def test_rechazar_at_020_boundary_is_not_above(self):
        # error_pct == 0.20 is NOT > 0.20 -> -200
        assert U.calc_reward(0.20, U.A_RECHAZAR) == -200.0

    def test_rechazar_below_020(self):
        assert U.calc_reward(0.05, U.A_RECHAZAR) == -200.0

    def test_action_by_string_equals_action_by_int(self):
        for name, idx in zip(U.ACTIONS, range(U.N_ACTIONS)):
            for e in [0.01, 0.10, 0.15, 0.20, 0.21, 0.25, 0.9]:
                assert U.calc_reward(e, name) == U.calc_reward(e, idx)

    def test_matches_original_monolith_semantics(self):
        """Reference re-implementation copied verbatim from SAVI_v2_ParcialFinal.py."""

        def calc_reward_monolith(error_pct, action):
            if action == "APROBAR":
                return +200 if error_pct < 0.10 else (-500 if error_pct < 0.25 else -2000)
            elif action == "REVISAR":
                return -150 if error_pct < 0.10 else -50
            elif action == "RECHAZAR":
                return +50 if error_pct > 0.20 else -200
            return 0

        errors = [0.0, 0.05, 0.0999, 0.10, 0.1001, 0.20, 0.2001, 0.24, 0.25, 0.26, 0.9, 2.0]
        for e in errors:
            for a in U.ACTIONS:
                assert U.calc_reward(e, a) == calc_reward_monolith(e, a)


class TestRewardMatrix:
    def test_matches_calc_reward_elementwise(self):
        errors = np.array([0.0, 0.05, 0.0999, 0.10, 0.1001, 0.20, 0.2001, 0.24, 0.25, 0.26, 0.9])
        R = U.reward_matrix(errors)
        assert R.shape == (len(errors), U.N_ACTIONS)
        for i, e in enumerate(errors):
            for a in range(U.N_ACTIONS):
                assert R[i, a] == U.calc_reward(float(e), a)

    def test_dtype_and_shape_empty(self):
        R = U.reward_matrix(np.array([]))
        assert R.shape == (0, U.N_ACTIONS)


# ════════════════════════════════════════════════════════════════════
# consensus_policy
# ════════════════════════════════════════════════════════════════════
class TestConsensusPolicy:
    def test_unanimous(self):
        vi = {0: "APROBAR"}
        ql = {0: "APROBAR"}
        dqn = {0: "APROBAR"}
        assert U.consensus_policy(vi, ql, dqn) == {0: "APROBAR"}

    def test_majority_2_1_vi_ql_agree(self):
        vi = {0: "APROBAR"}
        ql = {0: "APROBAR"}
        dqn = {0: "RECHAZAR"}
        assert U.consensus_policy(vi, ql, dqn)[0] == "APROBAR"

    def test_majority_2_1_vi_dqn_agree(self):
        vi = {0: "REVISAR"}
        ql = {0: "RECHAZAR"}
        dqn = {0: "REVISAR"}
        assert U.consensus_policy(vi, ql, dqn)[0] == "REVISAR"

    def test_tie_1_1_1_breaks_to_dqn(self):
        vi = {0: "APROBAR"}
        ql = {0: "REVISAR"}
        dqn = {0: "RECHAZAR"}
        assert U.consensus_policy(vi, ql, dqn)[0] == "RECHAZAR"

    def test_multi_state(self):
        vi = {0: "APROBAR", 1: "REVISAR"}
        ql = {0: "APROBAR", 1: "RECHAZAR"}
        dqn = {0: "RECHAZAR", 1: "REVISAR"}
        out = U.consensus_policy(vi, ql, dqn)
        assert out[0] == "APROBAR"   # 2-1 majority
        assert out[1] == "REVISAR"   # 1-1-1 tie -> DQN


# ════════════════════════════════════════════════════════════════════
# eval_policy
# ════════════════════════════════════════════════════════════════════
class TestEvalPolicy:
    def test_matches_manual_mean(self):
        policy = {0: "APROBAR", 1: "RECHAZAR"}
        clusters = np.array([0, 0, 1, 1])
        errors = np.array([0.05, 0.30, 0.05, 0.30])
        got = U.eval_policy(policy, clusters, errors)
        expected = np.mean([
            U.calc_reward(0.05, "APROBAR"),
            U.calc_reward(0.30, "APROBAR"),
            U.calc_reward(0.05, "RECHAZAR"),
            U.calc_reward(0.30, "RECHAZAR"),
        ])
        assert got == pytest.approx(expected)

    def test_single_state(self):
        policy = {3: "REVISAR"}
        clusters = np.array([3, 3, 3])
        errors = np.array([0.01, 0.50, 0.11])
        got = U.eval_policy(policy, clusters, errors)
        expected = np.mean([U.calc_reward(e, "REVISAR") for e in errors])
        assert got == pytest.approx(expected)


# ════════════════════════════════════════════════════════════════════
# save_json / load_json / load_policy round-trips
# ════════════════════════════════════════════════════════════════════
class TestJsonRoundtrip:
    def test_save_load_roundtrip_plain(self, tmp_path):
        obj = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        p = tmp_path / "out" / "x.json"
        U.save_json(obj, p)
        assert p.exists()
        assert U.load_json(p) == obj

    def test_save_json_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "c.json"
        U.save_json({"k": 1}, p)
        assert p.exists()

    def test_save_json_numpy_int_and_float(self, tmp_path):
        obj = {"i": np.int64(7), "f": np.float32(1.5)}
        p = tmp_path / "x.json"
        U.save_json(obj, p)
        loaded = U.load_json(p)
        assert loaded == {"i": 7, "f": 1.5}
        assert isinstance(loaded["i"], int)
        assert isinstance(loaded["f"], float)

    def test_save_json_numpy_array(self, tmp_path):
        obj = {"arr": np.array([1, 2, 3])}
        p = tmp_path / "x.json"
        U.save_json(obj, p)
        assert U.load_json(p) == {"arr": [1, 2, 3]}

    def test_save_json_unsupported_type_raises(self, tmp_path):
        class Unsupported:
            pass

        p = tmp_path / "x.json"
        with pytest.raises(TypeError):
            U.save_json({"bad": Unsupported()}, p)

    def test_load_policy_int_keys(self, tmp_path):
        p = tmp_path / "policy.json"
        p.write_text(json.dumps({"0": "APROBAR", "1": "REVISAR", "5": "RECHAZAR"}), encoding="utf-8")
        policy = U.load_policy(p)
        assert policy == {0: "APROBAR", 1: "REVISAR", 5: "RECHAZAR"}
        assert all(isinstance(k, int) for k in policy)

    def test_load_policy_roundtrip_with_save_json(self, tmp_path):
        original = {s: U.ACTIONS[s % U.N_ACTIONS] for s in range(U.N_CLUSTERS)}
        p = tmp_path / "policy_vi.json"
        U.save_json(original, p)
        loaded = U.load_policy(p)
        # keys come back as int, values unchanged
        assert loaded == original
        assert all(isinstance(k, int) for k in loaded)
