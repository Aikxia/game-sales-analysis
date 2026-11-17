"""从项目根目录运行完整的数据分析与模型训练流程。"""

import argparse
import json
from pathlib import Path

from src.config import DEFAULT_DATA_PATH, FIGURES_DIR, REPORTS_DIR
from src.data_loader import audit_data, clean_data, load_raw_data
from src.eda import generate_eda
from src.train import run_training


def parse_args() -> argparse.Namespace:
    # 命令行参数允许更换数据路径，或在重复训练时跳过绘图。
    parser = argparse.ArgumentParser(
        description="Audit, clean, explore and model the video game sales dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw vgsales.csv file.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Train models without regenerating EDA charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 第一步：读取原始数据并保存数据质量审计结果。
    raw_data = load_raw_data(args.data_path)
    audit = audit_data(raw_data)
    with (reports_dir / "data_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    # 第二步：清洗数据；后续建模统一使用这份清洗结果。
    cleaned_data = clean_data(raw_data)
    cleaning_summary = {
        "rows_before": int(len(raw_data)),
        "rows_after": int(len(cleaned_data)),
        "rows_removed": int(len(raw_data)) - int(len(cleaned_data)),
        "columns": list(cleaned_data.columns),
    }
    with (reports_dir / "data_cleaning.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(cleaning_summary, handle, indent=2, ensure_ascii=False)

    # 第三步：生成探索性分析图表；重复训练时可跳过以节省时间。
    eda_summary = None
    if not args.skip_eda:
        eda_summary = generate_eda(cleaned_data, FIGURES_DIR)
        with (reports_dir / "eda_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(eda_summary, handle, indent=2, ensure_ascii=False)

    # 第四步：训练基线模型和线性回归，保存指标文件。
    training_summary = run_training(cleaned_data, report_dir=reports_dir)

    print(
        json.dumps(
            {
                "data_audit": audit,
                "data_cleaning": cleaning_summary,
                "eda_summary": eda_summary,
                "training": training_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
