"""项目共享配置。"""

from pathlib import Path

# 从当前文件位置反推项目根目录，保证从不同工作目录启动时路径仍然正确。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据与报告的默认位置。
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "vgsales.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# 目标销量列，数据审计与建模时使用。
TARGET_COLUMN = "Global_Sales"

# 清洗与建模使用的特征列：均为游戏发行时即可获得的属性。
FEATURE_COLUMNS = ["Platform", "Year", "Genre", "Publisher"]

# 这些字段直接来自销量本身，不能作为预测 Global_Sales 的特征。
LEAKY_COLUMNS = [
    "Rank",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
]

# 固定随机种子和测试集比例，保证每次运行得到相同的划分和结果。
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 可信的发行年份范围，用于过滤明显错误的极端年份。
MIN_VALID_YEAR = 1980
MAX_VALID_YEAR = 2016
