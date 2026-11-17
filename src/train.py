"""训练均值基线与线性回归，并保存可复现的实验产物。"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.modeling import train_and_evaluate
from src.preprocessing import split_features_and_target


def run_training(
    data: pd.DataFrame,
    report_dir: Union[str, Path] = REPORTS_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """训练模型，并把指标与实验设置写入 metrics.json。"""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据划分与预处理都受随机种子控制，便于复现实验。
    x_train, x_test, y_train, y_test = split_features_and_target(
        data, test_size=test_size, random_state=random_state
    )
    metrics, _ = train_and_evaluate(x_train, x_test, y_train, y_test)

    summary: Dict[str, Any] = {
        "dataset": {
            "rows_used": int(len(data)),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "experiment": {
            "target": TARGET_COLUMN,
            "features": FEATURE_COLUMNS,
            "excluded_leaky_columns": LEAKY_COLUMNS,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "target_transform": (
                "log1p for linear regression, none for mean baseline"
            ),
        },
        "metrics": metrics,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary
