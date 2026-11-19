"""核心数据清洗、预处理与防数据泄漏的回归测试。"""

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    MAX_VALID_YEAR,
    MIN_VALID_YEAR,
    TARGET_COLUMN,
)
from src.data_loader import clean_data
from src.modeling import build_models
from src.preprocessing import build_preprocessor, split_features_and_target


def make_raw_row(year=2010, sales=2.5):
    """创建一条字段完整的模拟原始数据。"""
    return {
        "Rank": 1,
        "Name": "Test Game",
        "Platform": "PS3",
        "Year": year,
        "Genre": "Action",
        "Publisher": "Test Publisher",
        "NA_Sales": sales / 2,
        "EU_Sales": sales / 4,
        "JP_Sales": sales / 8,
        "Other_Sales": sales / 8,
        TARGET_COLUMN: sales,
    }


def test_clean_data_removes_invalid_year_and_duplicates():
    """清洗阶段应移除异常年份和重复记录。"""
    valid = make_raw_row()
    data = pd.DataFrame(
        [valid, valid.copy(), make_raw_row(year=2099)]
    )

    cleaned = clean_data(data)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Year"] == 2010


def test_split_never_contains_leaky_columns():
    """区域销量和 Rank 等泄漏字段不得进入训练或测试特征。"""
    rows = [make_raw_row(year=2000 + index) for index in range(10)]
    data = pd.DataFrame(rows)

    x_train, x_test, _, _ = split_features_and_target(data)

    assert list(x_train.columns) == FEATURE_COLUMNS
    assert not set(x_train.columns).intersection(LEAKY_COLUMNS)
    assert not set(x_test.columns).intersection(LEAKY_COLUMNS)


def test_preprocessor_output_contains_no_missing_values():
    """预处理结果不应出现 NaN，且行数与输入一致。"""
    rows = []
    for index in range(30):
        row = make_raw_row(year=2000 + index % 10)
        row["Publisher"] = "Publisher {}".format(index % 4)
        rows.append(row)
    data = clean_data(pd.DataFrame(rows))
    x_train, _, _, _ = split_features_and_target(data, test_size=0.33)

    transformed = build_preprocessor().fit_transform(x_train)

    assert transformed.shape[0] == len(x_train)
    assert not np.isnan(transformed).any()


def test_prediction_script_requires_valid_year():
    """年份边界与配置保持一致，便于预测脚本和 README 同步。"""
    assert MIN_VALID_YEAR == 1980
    assert MAX_VALID_YEAR == 2016


def test_mvp_uses_beginner_friendly_models():
    """MVP 只包含均值基线、线性回归和随机森林三个模型。"""
    assert list(build_models()) == [
        "mean_baseline",
        "linear_regression_log_target",
        "random_forest_log_target",
    ]
