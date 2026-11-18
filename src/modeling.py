"""训练并评估均值基线、普通线性回归与随机森林。"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor

ModelDict = Dict[str, object]


def build_models(random_state: int = RANDOM_STATE) -> ModelDict:
    """创建均值基线、普通线性回归和随机森林三个模型。"""
    # 销量呈长尾分布，先对目标做 log1p 变换，预测后再还原到原始尺度。
    linear_regression = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", LinearRegression()),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # 随机森林能捕捉类别特征与销量之间的非线性关系，作为主模型。
    random_forest = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=250,
                        min_samples_leaf=2,
                        max_features=0.7,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return {
        # 均值基线不学习特征，用来判断其他模型是否真正提供了预测价值。
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression_log_target": linear_regression,
        "random_forest_log_target": random_forest,
    }


def evaluate_predictions(
    actual: pd.Series, predicted: np.ndarray
) -> Dict[str, float]:
    """在原始销量尺度上计算回归指标。"""
    # RMSE 由 MSE 开方得到，使误差单位与销量单位一致。
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def train_and_evaluate(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, Dict[str, float]], ModelDict]:
    """训练全部候选模型，并返回指标和已拟合模型。"""
    metrics: Dict[str, Dict[str, float]] = {}
    fitted_models: ModelDict = {}

    for name, model in build_models(random_state=random_state).items():
        # 所有模型使用完全相同的训练集和测试集，保证比较公平。
        model.fit(x_train, y_train)
        # 销量不可能为负，因此把极端负预测统一截断为 0。
        predictions = np.maximum(model.predict(x_test), 0)
        metrics[name] = evaluate_predictions(y_test, predictions)
        fitted_models[name] = model

    return metrics, fitted_models


def calculate_permutation_importance(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """通过置换重要性衡量各原始特征对测试集 MAE 的影响。"""
    # 每次打乱一个原始特征后观察误差变化，结果比树模型内部重要性更直观。
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
