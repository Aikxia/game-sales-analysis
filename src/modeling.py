"""训练并评估均值基线与普通线性回归。"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

ModelDict = Dict[str, object]


def build_models() -> ModelDict:
    """创建均值基线和普通线性回归两个模型。"""
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

    return {
        # 均值基线不学习特征，用来判断其他模型是否真正提供了预测价值。
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression_log_target": linear_regression,
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
) -> Tuple[Dict[str, Dict[str, float]], ModelDict]:
    """训练全部候选模型，并返回指标和已拟合模型。"""
    metrics: Dict[str, Dict[str, float]] = {}
    fitted_models: ModelDict = {}

    for name, model in build_models().items():
        # 所有模型使用完全相同的训练集和测试集，保证比较公平。
        model.fit(x_train, y_train)
        # 销量不可能为负，因此把极端负预测统一截断为 0。
        predictions = np.maximum(model.predict(x_test), 0)
        metrics[name] = evaluate_predictions(y_test, predictions)
        fitted_models[name] = model

    return metrics, fitted_models
