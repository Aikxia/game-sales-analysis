"""项目共享配置。"""

from pathlib import Path

# 从当前文件位置反推项目根目录，保证从不同工作目录启动时路径仍然正确。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据与报告的默认位置。
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "vgsales.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 目标销量列，数据审计时用于统计。
TARGET_COLUMN = "Global_Sales"
