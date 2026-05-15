"""Tests for DataClassifier."""

from __future__ import annotations

from boed_agent.classifier import DataClassifier


def test_empty_data_is_homogeneous():
    result = DataClassifier(mode="raw").classify([])
    assert result.homogeneous is True
    assert result.cluster_labels == []


def test_homogeneous_single_cluster():
    data = [[1.0, 1.0]] * 10
    result = DataClassifier(mode="raw").classify(data)
    assert result.homogeneous is True


def test_raw_mode_raises_warning():
    data = [[0.0], [0.1]]
    result = DataClassifier(mode="raw").classify(data)
    assert any("homogeneity depends" in w for w in result.warnings)


def test_simulator_aware_without_simulator_warns():
    data = [[0.0], [0.1]]
    result = DataClassifier(mode="simulator_aware").classify(data, simulator=None)
    assert any("no simulator supplied" in w for w in result.warnings)
