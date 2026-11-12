"""读取并审计电子游戏销量数据集。"""

from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from src.config import TARGET_COLUMN

SOURCE_COLUMNS = [
    # 原始 CSV 必须包含这些列，否则数据审计结果将不完整。
    "Rank",
    "Name",
    "Platform",
    "Year",
    "Genre",
    "Publisher",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    TARGET_COLUMN,
]


def load_raw_data(path: Union[str, Path]) -> pd.DataFrame:
    """读取原始 CSV，并检查必要字段是否完整。"""
    data_path = Path(path)
    # 在读取前给出明确错误，避免后续出现难以定位的路径问题。
    if not data_path.exists():
        raise FileNotFoundError("Dataset not found: {}".format(data_path))

    data = pd.read_csv(data_path)
    # 使用集合差集一次找出全部缺失列，便于用户集中修复数据。
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: {}".format(
                ", ".join(missing_columns)
            )
        )
    return data


def audit_data(data: pd.DataFrame) -> Dict[str, Any]:
    """生成可写入 JSON 的原始数据审计摘要。"""
    # 转为数值后再统计，避免把无法解析的异常文本误认为正常数值。
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    years = pd.to_numeric(data["Year"], errors="coerce")

    # 区域销量之和通常等于全球销量，用于检查字段之间是否存在明显不一致。
    regional_sum = data[
        ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    ].sum(axis=1, min_count=4)

    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in data.isna().sum().items()
            if count > 0
        },
        "unique_values": {
            "Platform": int(data["Platform"].nunique(dropna=True)),
            "Genre": int(data["Genre"].nunique(dropna=True)),
            "Publisher": int(data["Publisher"].nunique(dropna=True)),
        },
        "year_range": {
            "min": float(years.min()),
            "max": float(years.max()),
        },
        "target_summary": {
            metric: float(value)
            for metric, value in target.describe().items()
        },
        "max_regional_sum_difference": float(
            (regional_sum - target).abs().max()
        ),
    }
