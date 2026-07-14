"""Dataset contract tests — run in CI on every PR (SPEC C0).

These enforce the freeze: if datasets/* drifts or the length-only control
reaches 50%, the build fails.
"""

from __future__ import annotations

import pytest

from gateway.datasets import load_cache_traps, load_complexity_labels
from scripts.length_baseline import COMMITTED, GATE, TOLERANCE, run_baseline
from scripts.validate_datasets import (
    check_cache_traps,
    check_complexity_labels,
    check_group_leakage,
)


@pytest.fixture(scope="module")
def traps():
    return load_cache_traps()


@pytest.fixture(scope="module")
def labels():
    return load_complexity_labels()


def test_cache_traps_contract(traps):
    assert check_cache_traps(traps) == []


def test_complexity_labels_contract(labels):
    assert check_complexity_labels(labels) == []


def test_zero_group_leakage_across_splits(labels):
    assert check_group_leakage(labels) == []


def test_trap_dev_test_prompt_disjoint(traps):
    dev = {p for r in traps if r["split"] == "dev" for p in (r["prompt_a"], r["prompt_b"])}
    test = {p for r in traps if r["split"] == "test" for p in (r["prompt_a"], r["prompt_b"])}
    assert not dev & test


@pytest.fixture(scope="module")
def baseline_results():
    return run_baseline()


class TestLengthBaselineGate:
    """The release gate: length alone must not predict tier at >= 50%."""

    def test_below_gate(self, baseline_results):
        results = baseline_results
        for name, acc in results.items():
            assert acc < GATE, f"{name} length-only baseline hit {acc:.1%} (gate {GATE:.0%})"

    def test_reproduces_committed_numbers(self, baseline_results):
        for name, acc in baseline_results.items():
            assert acc == pytest.approx(COMMITTED[name], abs=TOLERANCE), (
                f"{name} baseline {acc:.3%} drifted from committed {COMMITTED[name]:.1%}"
            )
