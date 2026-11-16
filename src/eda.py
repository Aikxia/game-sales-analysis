"""生成三张核心探索性分析图表。"""

from pathlib import Path
from typing import Dict, Union

import matplotlib

# 使用非交互式后端，保证服务器或命令行环境也能稳定保存图片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import TARGET_COLUMN


def generate_eda(
    data: pd.DataFrame, output_dir: Union[str, Path]
) -> Dict[str, Union[str, float]]:
    """生成并保存年份趋势、类型销量和平台销量图。"""
    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # 按发行年份汇总全球销量，并找出销量最高的年份用于标注。
    yearly_sales = (
        data.groupby("Year", as_index=False)[TARGET_COLUMN].sum().sort_values(
            "Year"
        )
    )
    peak_row = yearly_sales.loc[yearly_sales[TARGET_COLUMN].idxmax()]

    # 图一：展示全球销量随发行年份的变化趋势。
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(
        data=yearly_sales,
        x="Year",
        y=TARGET_COLUMN,
        marker="o",
        linewidth=2.2,
        color="#2a6f97",
        ax=ax,
    )
    ax.set_title("Global Video Game Sales by Release Year")
    ax.set_xlabel("Release year")
    ax.set_ylabel("Total global sales (millions of units)")
    ax.annotate(
        "Peak: {} ({:.1f}m)".format(
            int(peak_row["Year"]), float(peak_row[TARGET_COLUMN])
        ),
        xy=(peak_row["Year"], peak_row[TARGET_COLUMN]),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "sales_by_year.png", dpi=180)
    plt.close(fig)

    # 图二：按累计全球销量排序，只展示前十个游戏类型。
    genre_sales = (
        data.groupby("Genre")[TARGET_COLUMN]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        x=genre_sales.values,
        y=genre_sales.index,
        palette="Blues_r",
        ax=ax,
    )
    ax.set_title("Top 10 Genres by Total Global Sales")
    ax.set_xlabel("Total global sales (millions of units)")
    ax.set_ylabel("Genre")
    fig.tight_layout()
    fig.savefig(figure_dir / "sales_by_genre.png", dpi=180)
    plt.close(fig)

    # 图三：按累计全球销量排序，只展示前十个游戏平台。
    platform_sales = (
        data.groupby("Platform")[TARGET_COLUMN]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        x=platform_sales.values,
        y=platform_sales.index,
        palette="Greens_r",
        ax=ax,
    )
    ax.set_title("Top 10 Platforms by Total Global Sales")
    ax.set_xlabel("Total global sales (millions of units)")
    ax.set_ylabel("Platform")
    fig.tight_layout()
    fig.savefig(figure_dir / "sales_by_platform.png", dpi=180)
    plt.close(fig)

    # 返回关键数值，便于主流程同时保存 JSON 摘要。
    return {
        "peak_sales_year": float(peak_row["Year"]),
        "peak_year_sales": float(peak_row[TARGET_COLUMN]),
        "top_genre": str(genre_sales.idxmax()),
        "top_genre_sales": float(genre_sales.max()),
        "top_platform": str(platform_sales.idxmax()),
        "top_platform_sales": float(platform_sales.max()),
    }
