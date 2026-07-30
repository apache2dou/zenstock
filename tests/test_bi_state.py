"""缠论"走势结构的两重表里关系"（第91-92课）状态机与多级别诊断的测试。"""

import pytest

from zenstock.chanlun.bi_state import (
    BiState,
    classify_disease,
    compute_bi_state,
    is_valid_transition,
)


# ==================== 单级别状态计算 ====================

class TestComputeBiState:
    def test_up_bi_extending(self):
        """最后一笔向上，且无未完成顶分型 → (1, 1)"""
        state = compute_bi_state(last_bi_direction="up", fx_forming=False)
        assert state == BiState.UP_EXTENDING

    def test_down_bi_extending(self):
        state = compute_bi_state(last_bi_direction="down", fx_forming=True)
        assert state == BiState.DOWN_FX_FORMING

    def test_up_fx_forming(self):
        state = compute_bi_state(last_bi_direction="up", fx_forming=True)
        assert state == BiState.UP_FX_FORMING

    def test_down_extending(self):
        state = compute_bi_state(last_bi_direction="down", fx_forming=False)
        assert state == BiState.DOWN_EXTENDING


# ==================== 状态转移合法性 ====================

class TestTransition:
    def test_up_extending_can_only_go_to_up_fx(self):
        assert is_valid_transition(BiState.UP_EXTENDING, BiState.UP_FX_FORMING) is True
        assert is_valid_transition(BiState.UP_EXTENDING, BiState.DOWN_EXTENDING) is False
        assert is_valid_transition(BiState.UP_EXTENDING, BiState.DOWN_FX_FORMING) is False

    def test_down_extending_can_only_go_to_down_fx(self):
        assert is_valid_transition(BiState.DOWN_EXTENDING, BiState.DOWN_FX_FORMING) is True
        assert is_valid_transition(BiState.DOWN_EXTENDING, BiState.UP_EXTENDING) is False

    def test_fx_has_two_possible_next(self):
        assert is_valid_transition(BiState.UP_FX_FORMING, BiState.UP_EXTENDING) is True
        assert is_valid_transition(BiState.UP_FX_FORMING, BiState.DOWN_EXTENDING) is True
        assert is_valid_transition(BiState.UP_FX_FORMING, BiState.DOWN_FX_FORMING) is False

        assert is_valid_transition(BiState.DOWN_FX_FORMING, BiState.DOWN_EXTENDING) is True
        assert is_valid_transition(BiState.DOWN_FX_FORMING, BiState.UP_EXTENDING) is True


# ==================== 多级别病情诊断 ====================

class TestDiseaseClassification:
    def test_worst_case_down(self):
        """周线(-1,1) + 日线(-1,1) = 第1恶劣"""
        result = classify_disease(
            BiState.DOWN_EXTENDING, BiState.DOWN_EXTENDING
        )
        assert result["rank"] == 1
        assert "最恶劣" in result["label"]

    def test_second_worst(self):
        """周线(-1,1) + 日线(-1,0) = 第2恶劣"""
        result = classify_disease(
            BiState.DOWN_EXTENDING, BiState.DOWN_FX_FORMING
        )
        assert result["rank"] == 2

    def test_third_worst(self):
        """周线(-1,0) + 日线(-1,1) = 第3恶劣"""
        result = classify_disease(
            BiState.DOWN_FX_FORMING, BiState.DOWN_EXTENDING
        )
        assert result["rank"] == 3

    def test_turning_point(self):
        """周线(-1,0) + 日线(-1,0) = 第4（可能出现转机）"""
        result = classify_disease(
            BiState.DOWN_FX_FORMING, BiState.DOWN_FX_FORMING
        )
        assert result["rank"] == 4
        assert "转机" in result["label"]

    def test_up_healthy(self):
        """大级别向上延伸 + 小级别向上延伸 = 健康"""
        result = classify_disease(
            BiState.UP_EXTENDING, BiState.UP_EXTENDING
        )
        assert result["rank"] is None  # 下跌行情排序不适用
        assert result["health"] == "健康"
