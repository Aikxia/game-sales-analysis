"""训练候选模型，并保存可复现的实验产物。"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.modeling import (
    calculate_permutation_importance,
    train_and_evaluate,
)
from src.preprocessing import split_features_and_target


def run_training(
    data: pd.DataFrame,
    report_dir: Union[str, Path] = REPORTS_DIR,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """训练全部模型，保存指标、特征重要性和随机森林模型。"""
    output_dir = Path(report_dir)
    table_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    # 数据划分与预处理都受随机种子控制，便于复现实验。
    x_train, x_test, y_train, y_test = split_features_and_target(
        data, test_size=test_size, random_state=random_state
    )
    metrics, models = train_and_evaluate(
        x_train,
        x_test,
        y_train,
        y_test,
        random_state=random_state,
    )

    # 项目最终选择随机森林作为可调用模型，并保留全部模型指标用于比较。
    selected_model = models["random_forest_log_target"]
    importance = calculate_permutation_importance(
        selected_model, x_test, y_test, random_state=random_state
    )
    importance.to_csv(table_dir / "feature_importance.csv", index=False)
    # 压缩模型，避免随机森林的重复数组让文件体积过大。
    joblib.dump(selected_model, model_file, compress=3)

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
                "log1p for linear regression and random forest, "
                "none for mean baseline"
            ),
        },
        "metrics": metrics,
        "selected_model": "random_forest_log_target",
        "model_path": "models/random_forest_log_target.joblib",
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary
