import itertools
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from stockstats import StockDataFrame as Sdf

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

FILE_PATHS = [
    BASE_DIR.parent / "Dataset/HSI (2015-3-31 to 2026-3-31) Data.xlsx",
    BASE_DIR.parent / "Dataset/HSTECH (2015-3-31 to 2026-3-31) Data.xlsx",
    BASE_DIR.parent / "Dataset/HSCEI (2015-3-31 to 2026-3-31) Data.xlsx",
    BASE_DIR.parent / "Dataset/HSI ESG (2015-3-31 to 2026-3-31) Data.xlsx",
    BASE_DIR.parent / "Dataset/Hong Kong 1-Month Bond Yield  Data.xlsx",
]
TICKER_NAMES = ["HSI", "HSTECH", "HSCEI", "HSIESG", "HK_BOND"]

TRAIN_START = "2019-05-15"
TRAIN_END = "2024-12-31"
TRADE_START = "2025-01-01"
TRADE_END = "2026-03-31"
WARMUP_DAYS = 30


def data_split(df: pd.DataFrame, start: str, end: str, target_date_col: str = "date") -> pd.DataFrame:
    out = df[(df[target_date_col] >= start) & (df[target_date_col] < end)].copy()
    out = out.sort_values([target_date_col, "tic"], ignore_index=True)
    out.index = out[target_date_col].factorize()[0]
    return out


def convert_volume(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        text = val.upper().replace(",", "").strip()
        if text in {"-", "", "N/A", "NONE"}:
            return 0.0
        try:
            if text.endswith("B"):
                return float(text[:-1]) * 1e9
            if text.endswith("M"):
                return float(text[:-1]) * 1e6
            if text.endswith("K"):
                return float(text[:-1]) * 1e3
            return float(text)
        except ValueError:
            return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def load_single_file(file_path: Path, tic: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [col.lower().strip() for col in df.columns]
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if tic == "HK_BOND":
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * 100.0

    df["volume"] = df["volume"].apply(convert_volume).fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])

    if tic != "HK_BOND":
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df["tic"] = tic
    df["esg_score"] = 1.0 if tic == "HSIESG" else 0.0
    print(f"{tic}: {len(df)} rows | {df['date'].min()} -> {df['date'].max()}")
    return df


def add_indicators_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for tic, sub in df.groupby("tic", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True).copy()
        stock = Sdf.retype(sub.copy())
        for indicator in INDICATORS:
            try:
                sub[indicator] = stock[indicator].values
            except Exception:
                sub[indicator] = pd.NA
        sub["tic"] = tic
        result.append(sub)
    return pd.concat(result, ignore_index=True)


def trim_warmup(df: pd.DataFrame, warmup_days: int) -> pd.DataFrame:
    trimmed = []
    for tic, sub in df.sort_values(["tic", "date"]).groupby("tic", sort=False):
        keep = sub.iloc[warmup_days:] if len(sub) > warmup_days else sub.iloc[0:0]
        keep = keep.copy()
        keep["tic"] = tic
        trimmed.append(keep)
    return pd.concat(trimmed, ignore_index=True)


def build_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(df["tic"].unique().tolist())
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    full = pd.DataFrame(list(itertools.product(dates, tickers)), columns=["date", "tic"])
    merged = full.merge(df, on=["date", "tic"], how="left")
    merged = merged.sort_values(["tic", "date"]).reset_index(drop=True)

    value_cols = ["open", "high", "low", "close", "volume", "esg_score"] + [
        col for col in INDICATORS if col in merged.columns
    ]
    merged[value_cols] = merged.groupby("tic")[value_cols].ffill()
    merged = merged.dropna(subset=["close"]).reset_index(drop=True)

    non_bond_mask = merged["tic"] != "HK_BOND"
    merged = merged[
        (~non_bond_mask)
        | (
            (merged["open"] > 0)
            & (merged["high"] > 0)
            & (merged["low"] > 0)
            & (merged["close"] > 0)
        )
    ].reset_index(drop=True)
    return merged


def plot_line_by_ticker(df: pd.DataFrame, y_col: str, title: str, filename: str) -> None:
    plt.figure(figsize=(14, 7))
    for tic in sorted(df["tic"].unique()):
        temp = df[df["tic"] == tic]
        plt.plot(temp["date"], temp[y_col], label=tic)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


def plot_train_trade_split(df: pd.DataFrame, filename: str) -> None:
    plt.figure(figsize=(14, 7))
    for tic in sorted(df["tic"].unique()):
        temp = df[df["tic"] == tic]
        plt.plot(temp["date"], temp["close"], label=tic)
    plt.axvline(pd.to_datetime(TRAIN_END), color="red", linestyle="--", label="Train End")
    plt.axvline(pd.to_datetime(TRADE_START), color="green", linestyle="--", label="Trade Start")
    plt.title("Train / Trade Split on Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


def plot_close_subplots(df: pd.DataFrame, filename: str) -> None:
    tickers = sorted(df["tic"].unique())
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 4 * len(tickers)), sharex=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, tic in zip(axes, tickers):
        temp = df[df["tic"] == tic]
        ax.plot(temp["date"], temp["close"])
        ax.set_title(f"{tic} Close Price")
        ax.set_ylabel("Close")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


def main() -> None:
    df_list = []
    for file_path, tic in zip(FILE_PATHS, TICKER_NAMES):
        df_single = load_single_file(file_path, tic)
        if not df_single.empty:
            df_list.append(df_single)
    if not df_list:
        raise ValueError("No valid data loaded.")

    df_raw = pd.concat(df_list, ignore_index=True).sort_values(["tic", "date"]).reset_index(drop=True)
    processed = add_indicators_per_ticker(df_raw)
    processed = trim_warmup(processed, WARMUP_DAYS)
    processed_full = build_full_grid(processed)

    train = data_split(processed_full, TRAIN_START, TRAIN_END)
    trade = data_split(processed_full, TRADE_START, TRADE_END)

    train.to_csv(BASE_DIR / "train_data.csv", index=False)
    trade.to_csv(BASE_DIR / "trade_data.csv", index=False)

    plot_line_by_ticker(processed_full, "close", "Close Price Trends", "close_prices.png")
    plot_line_by_ticker(processed_full, "volume", "Volume Trends", "volume.png")
    for indicator in ["macd", "rsi_30", "cci_30", "dx_30"]:
        if indicator in processed_full.columns and processed_full[indicator].notna().any():
            plot_line_by_ticker(
                processed_full.dropna(subset=[indicator]),
                indicator,
                f"{indicator.upper()} Trends",
                f"{indicator}.png",
            )
    plot_train_trade_split(processed_full, "train_trade_split.png")
    plot_close_subplots(processed_full, "close_subplots.png")

    print("Saved:")
    print(f"  {BASE_DIR / 'train_data.csv'}")
    print(f"  {BASE_DIR / 'trade_data.csv'}")
    print(f"  {PLOT_DIR}")


if __name__ == "__main__":
    main()
