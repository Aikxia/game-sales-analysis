"""无数据泄漏的预处理与训练集/测试集划分。"""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

# 分类字段需要编码，Year 是当前唯一的连续数值字段。
CATEGORICAL_FEATURES = ["Platform", "Genre", "Publisher"]
NUMERIC_FEATURES = ["Year"]


def build_preprocessor() -> ColumnTransformer:
    """构建只能在训练集上拟合的预处理流程。"""
    # 分类流水线先填补最常见的类别，再做 One-Hot 编码。
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    # 合并低频类别，降低线性回归的共线性和数值不稳定性。
                    min_frequency=50,
                    sparse_output=False,
                ),
            ),
        ]
    )

    # 数值流水线先填补中位数，再标准化以改善线性模型稳定性。
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        # 只处理明确选择的特征，其余列一律丢弃，防止误入模型。
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def split_features_and_target(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """创建可复现的数据划分，并阻止泄漏字段进入特征。"""
    # 运行前再次检查，避免修改配置后不小心选中泄漏字段。
    forbidden_columns = set(FEATURE_COLUMNS) & set(LEAKY_COLUMNS)
    if forbidden_columns:
        raise ValueError(
            "Leaky columns selected as features: {}".format(
                ", ".join(sorted(forbidden_columns))
            )
        )

    # 确认清洗结果包含模型所需的全部字段。
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(
        data.columns
    )
    if missing_columns:
        raise ValueError(
            "Cleaned data is missing columns: {}".format(
                ", ".join(sorted(missing_columns))
            )
        )

    # 显式复制特征和目标，避免训练过程意外修改清洗后的主数据。
    features = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].copy()

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
