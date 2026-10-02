# -*- coding: utf-8 -*-
"""Tests for savi_cpu_pipeline.py: feature parsing, HPI math, MDP/VI/QL, loaders, e2e."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import savi_cpu_pipeline as C
import utils as U

DATA_INPUT = ROOT / "data" / "input" / "AmesHousing.txt"
DATA_REFERENCE = ROOT / "data" / "reference"


# ════════════════════════════════════════════════════════════════════
# helpers to build a minimal synthetic assessor roll xlsx
# ════════════════════════════════════════════════════════════════════
ROLL_COLUMNS = [
    "PARCERL NUMBER", "SUBDIVISION", "CLASSIFICATION", "MAIN STYLE", "YEAR BUILT",
    "GRADE", "CONDITION", "TOTAL LIVING AREA", "BASEMENT TYPE", "ATTIC TYPE",
    "TOTAL ROOMS ABOVE GRADE", "TOTAL ROOMS BELOW GRADE", "TOTAL BEDROOMS ABOVE GRADE",
    "TOTAL BEDROOMS BELOW GRADE", "# OF ADDITIONS", "BASEMENT FINISH", "# OF PORCHES",
    "# OF DECKS/PATIOS", "# OF PLUMBING FIXTURES", "# OF FIREPLACES", "GARAGE TYPE",
    "2024 LAND VALUE", "2024 TOTAL VALUE",
]


def _roll_row(pid, **overrides):
    row = {
        "PARCERL NUMBER": pid, "SUBDIVISION": "TESTSUB", "CLASSIFICATION": "Residential",
        "MAIN STYLE": "1 Story Frame", "YEAR BUILT": 1990, "GRADE": "3", "CONDITION": "Normal",
        "TOTAL LIVING AREA": 1500, "BASEMENT TYPE": "Full", "ATTIC TYPE": None,
        "TOTAL ROOMS ABOVE GRADE": 6, "TOTAL ROOMS BELOW GRADE": 2,
        "TOTAL BEDROOMS ABOVE GRADE": 3, "TOTAL BEDROOMS BELOW GRADE": 0,
        "# OF ADDITIONS": 0, "BASEMENT FINISH": "Yes", "# OF PORCHES": 1,
        "# OF DECKS/PATIOS": 0, "# OF PLUMBING FIXTURES": 8, "# OF FIREPLACES": 1,
        "GARAGE TYPE": "Att Frame", "2024 LAND VALUE": 50000, "2024 TOTAL VALUE": 200000,
    }
    row.update(overrides)
    return row


def _write_roll(tmp_path, rows) -> Path:
    df = pd.DataFrame(rows, columns=ROLL_COLUMNS)
    p = tmp_path / "residential-properties-with-detail-2024.xlsx"
    df.to_excel(p, index=False)
    return p


# ════════════════════════════════════════════════════════════════════
# load_assessor_roll — feature parsing
# ════════════════════════════════════════════════════════════════════
class TestLoadAssessorRollStories:
    @pytest.mark.parametrize("style,expected", [
        ("1 1/2 Story Frame", 1.5),
        ("1 3/4 Story Frame", 1.75),
        ("2 Story Frame", 2.0),
        ("1 Story Frame", 1.0),
    ])
    def test_stories_parsed(self, tmp_path, style, expected):
        p = _write_roll(tmp_path, [_roll_row("1", **{"MAIN STYLE": style})])
        a = C.load_assessor_roll(p)
        assert a.iloc[0]["stories"] == pytest.approx(expected)

    def test_split_level_defaults_stories_to_1(self, tmp_path):
        p = _write_roll(tmp_path, [_roll_row("1", **{"MAIN STYLE": "Split Level"})])
        a = C.load_assessor_roll(p)
        assert a.iloc[0]["stories"] == pytest.approx(1.0)
        assert a.iloc[0]["is_split"] == 1

    def test_condo_townhouse_brick_flags(self, tmp_path):
        rows = [
            _roll_row("1", **{"MAIN STYLE": "2 Story Condo"}),
            _roll_row("2", **{"MAIN STYLE": "1 Story Townhouse"}),
            _roll_row("3", **{"MAIN STYLE": "1 Story Brick"}),
        ]
        p = _write_roll(tmp_path, rows)
        a = C.load_assessor_roll(p).set_index("PID")
        assert a.loc[a.index[0], "is_condo"] == 1
        assert a.loc[a.index[1], "is_townhouse"] == 1
        assert a.loc[a.index[2], "is_brick"] == 1


class TestLoadAssessorRollGrade:
    @pytest.mark.parametrize("grade,base,mod", [
        ("4+10", 4, 10),
        ("3-5", 3, -5),
        ("3", 3, 0),
    ])
    def test_grade_parsed(self, tmp_path, grade, base, mod):
        p = _write_roll(tmp_path, [_roll_row("1", **{"GRADE": grade})])
        a = C.load_assessor_roll(p)
        assert a.iloc[0]["grade_base"] == base
        assert a.iloc[0]["grade_mod"] == mod


class TestLoadAssessorRollDedupe:
    def test_dedupe_by_pid_and_n_garages_count(self, tmp_path):
        # Same PID (two rows -> multiple garages on the parcel), each with a garage type set.
        rows = [
            _roll_row("1", **{"GARAGE TYPE": "Att Frame"}),
            _roll_row("1", **{"GARAGE TYPE": "Det Frame"}),
            _roll_row("2", **{"GARAGE TYPE": "Att Frame"}),
        ]
        p = _write_roll(tmp_path, rows)
        a = C.load_assessor_roll(p)
        # one row per PID
        assert a["PID"].is_unique
        assert len(a) == 2
        pid1 = "1".zfill(10)
        pid2 = "2".zfill(10)
        assert a.set_index("PID").loc[pid1, "n_garages"] == 2
        assert a.set_index("PID").loc[pid2, "n_garages"] == 1

    def test_dedupe_keeps_first_row_values(self, tmp_path):
        rows = [
            _roll_row("1", **{"GARAGE TYPE": "Att Frame", "YEAR BUILT": 1950}),
            _roll_row("1", **{"GARAGE TYPE": "Det Frame", "YEAR BUILT": 1999}),
        ]
        p = _write_roll(tmp_path, rows)
        a = C.load_assessor_roll(p)
        assert len(a) == 1
        assert a.iloc[0]["year_built"] == 1950  # "first" kept


# ════════════════════════════════════════════════════════════════════
# to_today — HPI math
# ════════════════════════════════════════════════════════════════════
class TestToToday:
    def test_scales_by_hpi_ratio(self):
        hpi = {(2020, 1): 200.0, (2024, 1): 300.0}
        latest = (2024, 1)
        values = [100.0, 50.0]
        years = [2020, 2020]
        quarters = [1, 1]
        out = C.to_today(values, years, quarters, hpi, latest)
        expected = np.array([100.0, 50.0]) * 300.0 / 200.0
        np.testing.assert_allclose(out, expected)

    def test_same_quarter_as_latest_is_unchanged(self):
        hpi = {(2024, 1): 300.0}
        latest = (2024, 1)
        out = C.to_today([123.0], [2024], [1], hpi, latest)
        np.testing.assert_allclose(out, [123.0])

    def test_missing_quarter_raises_keyerror(self):
        hpi = {(2024, 1): 300.0}
        with pytest.raises(KeyError):
            C.to_today([100.0], [1999], [1], hpi, (2024, 1))


# ════════════════════════════════════════════════════════════════════
# build_mdp
# ════════════════════════════════════════════════════════════════════
class TestBuildMdp:
    def test_transition_rows_sum_to_one(self, monkeypatch):
        monkeypatch.setattr(U, "N_CLUSTERS", 3)
        clusters = np.array([0, 1, 0, 1, 2, 0, 1])
        errors = np.array([0.01, 0.5, 0.02, 0.4, 0.9, 0.03, 0.45])
        R, P = C.build_mdp(clusters, errors)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(3))

    def test_absorbing_state_for_cluster_without_data(self, monkeypatch):
        monkeypatch.setattr(U, "N_CLUSTERS", 3)
        # cluster 2 never appears
        clusters = np.array([0, 1, 0, 1, 0, 1])
        errors = np.array([0.01, 0.5, 0.02, 0.4, 0.03, 0.45])
        R, P = C.build_mdp(clusters, errors)
        np.testing.assert_allclose(P[2], [0.0, 0.0, 1.0])  # identity row -> absorbing
        np.testing.assert_allclose(R[2], [0.0, 0.0, 0.0])  # no data -> zero reward row

    def test_reward_matches_mean_per_cluster(self, monkeypatch):
        monkeypatch.setattr(U, "N_CLUSTERS", 2)
        clusters = np.array([0, 0, 1, 1])
        errors = np.array([0.01, 0.30, 0.01, 0.30])
        R, P = C.build_mdp(clusters, errors)
        expected_row = np.mean(U.reward_matrix(errors[:2]), axis=0)
        np.testing.assert_allclose(R[0], expected_row)
        np.testing.assert_allclose(R[1], expected_row)


# ════════════════════════════════════════════════════════════════════
# value_iteration on a tiny hand-made MDP with a known answer
# ════════════════════════════════════════════════════════════════════
class TestValueIteration:
    def test_two_absorbing_states_known_answer(self, monkeypatch):
        monkeypatch.setattr(U, "N_CLUSTERS", 2)
        monkeypatch.setattr(U, "GAMMA", 0.9)
        monkeypatch.setattr(U, "THETA", 1e-6)
        # State 0: APROBAR is clearly best (10); state 1: RECHAZAR is clearly best (5).
        R = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        P = np.eye(2)  # both states absorbing (self-loop)
        V, Q, policy, it = C.value_iteration(R, P)
        assert policy[0] == "APROBAR"
        assert policy[1] == "RECHAZAR"
        # analytic fixed point: V = r_best / (1 - gamma) for an absorbing self-loop
        assert V[0] == pytest.approx(10.0 / (1 - 0.9), rel=1e-3)
        assert V[1] == pytest.approx(5.0 / (1 - 0.9), rel=1e-3)


# ════════════════════════════════════════════════════════════════════
# q_learning converges to the same policy as VI on a small deterministic case
# ════════════════════════════════════════════════════════════════════
class TestQLearning:
    def test_converges_to_vi_policy(self, monkeypatch):
        monkeypatch.setattr(U, "N_CLUSTERS", 2)
        monkeypatch.setattr(U, "GAMMA", 0.9)
        monkeypatch.setattr(U, "THETA", 1e-6)
        monkeypatch.setattr(U, "EPISODES_QL", 400)  # small for speed
        monkeypatch.setattr(U, "EPS_DECAY", 0.97)
        monkeypatch.setattr(U, "EPS_MIN", 0.05)

        rng = np.random.default_rng(0)
        n = 200
        clusters = rng.integers(0, 2, n)
        errors = np.where(clusters == 0, 0.01, 0.80)  # cluster0 -> approve great; cluster1 -> reject great
        P = np.eye(2)

        R, _ = C.build_mdp(clusters, errors)
        _, _, vi_policy, _ = C.value_iteration(R, P)
        _, ql_policy, _ = C.q_learning(clusters, errors, P)

        assert ql_policy == vi_policy


# ════════════════════════════════════════════════════════════════════
# load_decock_sales
# ════════════════════════════════════════════════════════════════════
class TestLoadDecockSales:
    def test_rejects_csv_without_pid(self, tmp_path):
        p = tmp_path / "combined.csv"
        pd.DataFrame({"SalePrice": [100000], "YrSold": [2008], "MoSold": [5]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="no trae PID"):
            C.load_decock_sales(p)

    def test_accepts_csv_with_pid(self, tmp_path):
        p = tmp_path / "sales.csv"
        pd.DataFrame({
            "PID": ["1"], "SalePrice": [100000.0], "YrSold": [2008], "MoSold": [5],
            "SaleCondition": ["Normal"], "GrLivArea": [1500],
        }).to_csv(p, index=False)
        out = C.load_decock_sales(p)
        assert out.iloc[0]["PID"] == "0000000001"
        assert out.iloc[0]["sale_price"] == 100000.0


# ════════════════════════════════════════════════════════════════════
# END-TO-END on real data
# ════════════════════════════════════════════════════════════════════
@pytest.mark.slow
@pytest.mark.skipif(not DATA_INPUT.exists() or not DATA_REFERENCE.exists(),
                     reason="real Ames data not present")
def test_run_end_to_end_real_data(tmp_path):
    out_dir = tmp_path / "out"
    metrics = C.run(
        input_uri=str(DATA_INPUT),
        reference=str(DATA_REFERENCE),
        output=str(out_dir),
        run_id="test-e2e",
    )

    A = U.ARTIFACTS
    for key, rel in A.items():
        assert (out_dir / rel).exists(), f"missing artifact {key} -> {rel}"

    success = json.loads((out_dir / A["success"]).read_text(encoding="utf-8"))
    assert success["status"] == "SUCCEEDED"
    assert success["run_id"] == "test-e2e"

    train_states = np.load(out_dir / A["train_states"])
    train_errors = np.load(out_dir / A["train_errors"])
    train_next = np.load(out_dir / A["train_next"])
    train_clusters = np.load(out_dir / A["train_clusters"])
    portfolio_states = np.load(out_dir / A["portfolio_states"])
    portfolio_meta = pd.read_csv(out_dir / A["portfolio_meta"])

    n_train = train_states.shape[0]
    assert train_errors.shape[0] == n_train
    assert train_next.shape[0] == n_train
    assert train_clusters.shape[0] == n_train
    assert portfolio_states.shape[0] == len(portfolio_meta)
    assert portfolio_states.shape[1] == train_states.shape[1]  # same state dim

    assert train_states.min() >= 0.0 and train_states.max() <= 1.0
    assert portfolio_states.min() >= 0.0 and portfolio_states.max() <= 1.0

    # train_next must be a valid index array into train_states
    assert train_next.min() >= 0
    assert train_next.max() < n_train

    best = metrics["avm_selected"]
    assert metrics["supervised"][best]["r2_usd"] > 0.85
