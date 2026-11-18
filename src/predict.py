"""根据一个游戏配置预测全球销量。"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH,
    MAX_VALID_YEAR,
    MIN_VALID_YEAR,
)


def parse_args() -> argparse.Namespace:
    # 四个必要输入与训练阶段使用的四个特征完全一致。
    parser = argparse.ArgumentParser(
        description="Predict global sales with the trained random forest."
    )
    parser.add_argument("--platform", required=True)
    parser.add_argument("--genre", required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 预测前检查年份和模型文件，尽早给出可理解的错误信息。
    if not MIN_VALID_YEAR <= args.year <= MAX_VALID_YEAR:
        raise ValueError(
            "Year must be between {} and {}.".format(
                MIN_VALID_YEAR, MAX_VALID_YEAR
            )
        )
    if not args.model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `py -m src.main` first."
        )

    # 加载训练阶段保存的随机森林模型。
    model = joblib.load(args.model_path)
    # 构造单行 DataFrame，确保列名与训练数据保持一致。
    features = pd.DataFrame(
        [
            {
                "Platform": args.platform,
                "Year": args.year,
                "Genre": args.genre,
                "Publisher": args.publisher,
            }
        ]
    )
    # 销量不能为负数，因此对模型输出执行下界截断。
    prediction = max(float(model.predict(features)[0]), 0.0)
    print(
        json.dumps(
            {
                "input": features.iloc[0].to_dict(),
                "predicted_global_sales_millions": round(prediction, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
