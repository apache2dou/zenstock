import pytest

from zenstock.chanlun.segments import apply_fx_auxiliary_operation


def test_apply_fx_auxiliary_operation_finds_top_divergence_without_gap():
    features = [(15.0, 8.0), (20.0, 12.0), (18.0, 10.0)]

    result = apply_fx_auxiliary_operation(features, "up")

    assert result["std_features"] == [(15.0, 8.0), (20.0, 12.0), (18.0, 10.0)]
    assert result["fx_idx"] == 1
    assert result["fx_type"] == "顶分型"
    assert result["has_gap"] is False


def test_apply_fx_auxiliary_operation_finds_bottom_divergence_with_gap():
    features = [(20.0, 12.0), (10.0, 5.0), (12.0, 8.0)]

    result = apply_fx_auxiliary_operation(features, "down")

    assert result["fx_idx"] == 1
    assert result["fx_type"] == "底分型"
    assert result["has_gap"] is True


def test_apply_fx_auxiliary_operation_returns_none_when_no_divergence():
    features = [(10.0, 8.0), (11.0, 7.0), (10.5, 8.3)]

    result = apply_fx_auxiliary_operation(features, "up")

    assert result["fx_idx"] is None
    assert result["fx_type"] is None
    assert result["has_gap"] is False
