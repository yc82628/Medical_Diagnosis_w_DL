"""Shared test setup."""
import pytest

import quality_model


@pytest.fixture(autouse=True)
def _classical_quality_gate_by_default(tmp_path, monkeypatch):
    """
    A trained quality model in checkpoints/ must not change what the suite
    measures: tests see the classical gate unless they inject a learned one.
    """
    monkeypatch.setenv("DR_QUALITY_CHECKPOINT", str(tmp_path / "no_quality_model.pt"))
    quality_model.reset_default_quality_gate()
    yield
    quality_model.reset_default_quality_gate()
