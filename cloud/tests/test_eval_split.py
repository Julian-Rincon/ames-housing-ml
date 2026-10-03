# -*- coding: utf-8 -*-
"""Split estratificado, cadenas cronológicas por subconjunto, bootstrap y reglas de decisión."""
import numpy as np
import pytest

import utils as U


def test_stratified_split_proportions_and_small_strata():
    strata = np.array([0] * 100 + [1] * 40 + [2] * 2)
    split = U.stratified_split(strata)
    for s, n in [(0, 100), (1, 40)]:
        counts = np.bincount(split[strata == s], minlength=3)
        assert counts.sum() == n
        assert abs(counts[U.SPLIT_VAL] - 0.15 * n) <= 1 and abs(counts[U.SPLIT_TEST] - 0.15 * n) <= 1
    assert (split[strata == 2] == U.SPLIT_TRAIN).all()  # <3 filas → todo a TRAIN


def test_stratified_split_is_deterministic():
    strata = np.random.default_rng(0).integers(0, 6, 500)
    assert (U.stratified_split(strata) == U.stratified_split(strata)).all()


def test_next_within_split_never_crosses_subsets():
    split = np.array([0, 1, 0, 2, 1, 0, 2])
    nxt = U.next_within_split(split)
    assert list(nxt) == [2, 4, 5, 6, 4, 5, 6]
    assert (split[nxt] == split).all()


def test_bootstrap_ci_contains_mean_and_degenerates():
    v = np.random.default_rng(1).normal(10, 2, 400)
    m, lo, hi = U.bootstrap_ci(v)
    assert lo < m < hi and abs(m - v.mean()) < 1e-12
    assert U.bootstrap_ci(np.zeros(50)) == (0.0, 0.0, 0.0)


def test_decision_rules_gated_bounds():
    torch = pytest.importorskip("torch")  # noqa: F841 (savi_gpu_sagemaker importa torch)
    import savi_gpu_sagemaker as G
    rng = np.random.default_rng(0)
    q_s, q_p = rng.normal(size=(60, 3)), rng.normal(size=(30, 3))
    cl_s, cl_p = rng.integers(0, U.N_CLUSTERS, 60), rng.integers(0, U.N_CLUSTERS, 30)
    pol = {s: "APROBAR" for s in range(U.N_CLUSTERS)}
    srt = np.sort(q_s, axis=1)
    rules, pol_dqn, pol_cons, taus = G.decision_rules(q_s, q_p, cl_s, cl_p, np.ones(60, bool), pol, pol,
                                                      srt[:, -1] - srt[:, -2])
    assert {"vi", "ql", "dqn_state", "consensus_state", "dqn_property", "consensus_property"} <= set(rules)
    assert set(taus) == {f"gated_q{p}" for p in (10, 25, 50, 75, 90)}
    assert all(taus[f"gated_q{a}"] <= taus[f"gated_q{b}"] for a, b in [(10, 25), (25, 50), (50, 75), (75, 90)])
    # Umbral mínimo ⇒ casi siempre decide el DQN; consenso con VI=QL=APROBAR ⇒ APROBAR
    assert (rules["consensus_property"][0] == 0).all()
    g10 = rules["gated_q10"][0]
    assert (g10 == rules["dqn_property"][0]).mean() > 0.8
    for a_s, a_p in rules.values():
        assert a_s.shape == (60,) and a_p.shape == (30,)
