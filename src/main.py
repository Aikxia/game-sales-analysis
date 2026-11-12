"""从项目根目录运行数据分析流程的第一步：读取原始数据并生成数据审计报告。"""

import argparse
import json
from pathlib import Path

from src.config import DEFAULT_DATA_PATH, REPORTS_DIR
from src.data_loader import audit_data, load_raw_data


def parse_args() -> argparse.Namespace:
    # 命令行参数允许在不修改代码的情况下更换数据路径。
    parser = argparse.ArgumentParser(
        description="Audit the raw video game sales dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw vgsales.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始数据并保存数据质量审计结果。
    raw_data = load_raw_data(args.data_path)
    audit = audit_data(raw_data)
    with (reports_dir / "data_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
