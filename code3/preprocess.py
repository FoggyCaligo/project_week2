from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

TARGET_COL = "daily_rental_count"
DATE_ORIGIN_FILE = "date_origin.txt"

SPLIT_PRESET_2025 = {
    "TRAIN_END": "2024-12-31",
    "VAL_START": "2025-01-01",
    "VAL_END": "2025-02-28",
    "TEST_START": "2025-03-01",
    "TEST_END": "2025-04-30",
}

SPLIT_PRESET_2024 = {
    "TRAIN_END": "2024-09-30",
    "VAL_START": "2024-10-01",
    "VAL_END": "2024-11-30",
    "TEST_START": "2024-12-01",
    "TEST_END": "2024-12-31",
}

SCRIPT_DIR = Path(__file__).resolve().parent


def find_base_dir() -> Path:
    candidates = [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]
    for candidate in candidates:
        if (candidate / "dataset").exists():
            return candidate
    return SCRIPT_DIR


BASE_DIR = find_base_dir()
RAW_BIKE_DIR = BASE_DIR / "dataset" / "raw_data" / "bike"
RAW_WEATHER_DIR = BASE_DIR / "dataset" / "raw_data" / "weather"
REFORMED_DIR = BASE_DIR / "dataset" / "reformed_data"
MODEL_OUTPUT_DIR = BASE_DIR / "dataset" / "model_output"
REPORT_DIR = BASE_DIR / "dataset" / "report"

REFORMED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------

def log(message: str):
    print(message)


def normalize_colname(col: str) -> str:
    col = str(col).strip()
    col = col.replace("\n", "")
    col = re.sub(r"\s+", "", col)
    return col


def find_col_by_keywords(columns, keywords):
    normalized = {c: normalize_colname(c) for c in columns}
    for original, normed in normalized.items():
        if all(keyword in normed for keyword in keywords):
            log(f"find_col_by_keywords 완료: {keywords} -> {original}")
            return original
    log(f"find_col_by_keywords 완료: {keywords} -> None")
    return None


def collect_files(data_dir: Path):
    files = []
    for pattern in ["*.csv", "*.xlsx", "*.xls"]:
        files.extend(glob.glob(str(data_dir / pattern)))
    files = sorted(files)
    log(f"collect_files 완료: {data_dir} / {len(files)}개")
    return files


def detect_csv_header_and_encoding(path: str):
    encodings = [
        "cp949",
        "utf-8-sig",
        "utf-8",
        "euc-kr",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
    ]
    errors_list = []
    for enc in encodings:
        try:
            header_df = pd.read_csv(path, encoding=enc, nrows=0)
            log(f"detect_csv_header_and_encoding 완료: {path} / encoding={enc}")
            return header_df.columns.tolist(), enc
        except Exception as e:
            errors_list.append(f"{enc}: {e}")
    raise ValueError(
        f"CSV 헤더 읽기 실패: {path}\n시도한 인코딩:\n" + "\n".join(errors_list)
    )


def read_weather_table_flexible(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
        log(f"read_weather_table_flexible 완료: {path}")
        return df

    if ext == ".csv":
        encodings = [
            "cp949",
            "utf-8-sig",
            "utf-8",
            "euc-kr",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
        ]
        errors_list = []
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                log(f"read_weather_table_flexible 완료: {path} / encoding={enc}")
                return df
            except Exception as e:
                errors_list.append(f"{enc}: {e}")

        raise ValueError(
            f"날씨 CSV 파일 읽기 실패: {path}\n시도한 인코딩:\n" + "\n".join(errors_list)
        )

    raise ValueError(f"지원하지 않는 파일 형식: {path}")


# ------------------------------------------------------------
# 데이터 로딩 / 전처리
# ------------------------------------------------------------

def load_bike_daily_from_files(bike_dir: Path) -> pd.DataFrame:
    files = collect_files(bike_dir)
    if not files:
        raise FileNotFoundError(f"따릉이 파일을 찾지 못했습니다: {bike_dir}")

    daily_parts = []

    for path in files:
        log(f"[bike] 처리 중: {path}")
        ext = os.path.splitext(path)[1].lower()

        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
            rental_dt_col = find_col_by_keywords(df.columns, ["대여일시"])
            if rental_dt_col is None:
                raise KeyError(
                    f"{path} 에서 '대여일시' 컬럼을 찾지 못했습니다.\n현재 컬럼: {df.columns.tolist()}"
                )

            temp = df[[rental_dt_col]].copy()
            temp[rental_dt_col] = pd.to_datetime(temp[rental_dt_col], errors="coerce")
            temp = temp.dropna(subset=[rental_dt_col]).copy()
            temp["date"] = temp[rental_dt_col].dt.floor("D")
            daily = temp.groupby("date").size().reset_index(name=TARGET_COL)
            daily_parts.append(daily)
            continue

        if ext == ".csv":
            columns, enc = detect_csv_header_and_encoding(path)
            rental_dt_col = find_col_by_keywords(columns, ["대여일시"])
            if rental_dt_col is None:
                raise KeyError(
                    f"{path} 에서 '대여일시' 컬럼을 찾지 못했습니다.\n현재 컬럼: {columns}"
                )

            chunk_parts = []
            for chunk in pd.read_csv(
                path,
                encoding=enc,
                usecols=[rental_dt_col],
                chunksize=200000,
                low_memory=False,
            ):
                chunk[rental_dt_col] = pd.to_datetime(chunk[rental_dt_col], errors="coerce")
                chunk = chunk.dropna(subset=[rental_dt_col]).copy()
                chunk["date"] = chunk[rental_dt_col].dt.floor("D")
                daily_chunk = chunk.groupby("date").size().reset_index(name=TARGET_COL)
                chunk_parts.append(daily_chunk)

            if chunk_parts:
                daily = pd.concat(chunk_parts, ignore_index=True)
                daily = (
                    daily.groupby("date", as_index=False)[TARGET_COL]
                    .sum()
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                daily_parts.append(daily)
            continue

        raise ValueError(f"지원하지 않는 파일 형식: {path}")

    if not daily_parts:
        raise ValueError("따릉이 일별 데이터가 하나도 생성되지 않았습니다.")

    daily_bike = pd.concat(daily_parts, ignore_index=True)
    daily_bike = (
        daily_bike.groupby("date", as_index=False)[TARGET_COL]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    log("load_bike_daily_from_files 완료")
    return daily_bike


def load_weather_raw(weather_dir: Path) -> pd.DataFrame:
    files = collect_files(weather_dir)
    if not files:
        raise FileNotFoundError(f"날씨 파일을 찾지 못했습니다: {weather_dir}")

    dfs = []
    for path in files:
        log(f"[weather] 읽는 중: {path}")
        df = read_weather_table_flexible(path)
        df.columns = [str(c).strip() for c in df.columns]
        df["__source_file"] = os.path.basename(path)
        dfs.append(df)

    weather_raw = pd.concat(dfs, ignore_index=True)
    weather_raw = weather_raw.drop_duplicates().reset_index(drop=True)
    log("load_weather_raw 완료")
    return weather_raw


def preprocess_weather_daily(weather_raw: pd.DataFrame) -> pd.DataFrame:
    weather = weather_raw.copy()
    original_cols = weather.columns.tolist()

    col_date = find_col_by_keywords(original_cols, ["일시"])
    col_avg_temp = find_col_by_keywords(original_cols, ["평균기온", "°C"])
    col_min_temp = find_col_by_keywords(original_cols, ["최저기온", "°C"])
    col_max_temp = find_col_by_keywords(original_cols, ["최고기온", "°C"])
    col_precip_dur = find_col_by_keywords(original_cols, ["강수", "계속시간"])
    col_daily_precip = find_col_by_keywords(original_cols, ["일강수량"])
    col_avg_wind = find_col_by_keywords(original_cols, ["평균", "풍속"])
    col_avg_humidity = find_col_by_keywords(original_cols, ["평균", "상대습도"])

    rename_map = {}
    if col_date:
        rename_map[col_date] = "date"
    if col_avg_temp:
        rename_map[col_avg_temp] = "avg_temp_c"
    if col_min_temp:
        rename_map[col_min_temp] = "min_temp_c"
    if col_max_temp:
        rename_map[col_max_temp] = "max_temp_c"
    if col_precip_dur:
        rename_map[col_precip_dur] = "precip_duration_hr"
    if col_daily_precip:
        rename_map[col_daily_precip] = "daily_precip_mm"
    if col_avg_wind:
        rename_map[col_avg_wind] = "avg_wind_speed_m_s"
    if col_avg_humidity:
        rename_map[col_avg_humidity] = "avg_rel_humidity_pct"

    weather = weather.rename(columns=rename_map)
    required = [
        "date",
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "precip_duration_hr",
        "daily_precip_mm",
        "avg_wind_speed_m_s",
        "avg_rel_humidity_pct",
    ]
    missing = [c for c in required if c not in weather.columns]
    if missing:
        raise KeyError(f"날씨 필수 컬럼이 없습니다: {missing}\n현재 컬럼: {weather.columns.tolist()}")

    weather = weather[required].copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather = weather.dropna(subset=["date"]).copy()

    for c in required:
        if c != "date":
            weather[c] = pd.to_numeric(weather[c], errors="coerce")

    weather["precip_duration_hr"] = weather["precip_duration_hr"].fillna(0)
    weather["daily_precip_mm"] = weather["daily_precip_mm"].fillna(0)

    fill_cols = [
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "avg_wind_speed_m_s",
        "avg_rel_humidity_pct",
    ]
    for c in fill_cols:
        weather[c] = weather[c].interpolate(limit_direction="both")

    weather = (
        weather.sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .reset_index(drop=True)
    )
    log("preprocess_weather_daily 완료")
    return weather


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def make_features(daily_bike: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(daily_bike, weather_daily, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    date_origin = df["date"].min()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_rainy"] = (df["daily_precip_mm"].fillna(0) > 0).astype(int)
    df["temp_range_c"] = df["max_temp_c"] - df["min_temp_c"]
    df["trend_idx"] = (df["date"] - date_origin).dt.days.astype(int)

    df = add_cyclical_features(df)

    df["temp_x_rain"] = df["avg_temp_c"] * df["is_rainy"]
    df["humidity_x_rain"] = df["avg_rel_humidity_pct"] * df["is_rainy"]
    df["precip_x_humidity"] = df["daily_precip_mm"] * df["avg_rel_humidity_pct"]
    df["wind_x_rain"] = df["avg_wind_speed_m_s"] * df["is_rainy"]

    def month_to_season(m):
        if m in [12, 1, 2]:
            return "winter"
        if m in [3, 4, 5]:
            return "spring"
        if m in [6, 7, 8]:
            return "summer"
        return "autumn"

    df["season"] = df["month"].apply(month_to_season)
    (REFORMED_DIR / DATE_ORIGIN_FILE).write_text(str(pd.to_datetime(date_origin).date()), encoding="utf-8")

    log("make_features 완료")
    return df


def choose_split_preset(df: pd.DataFrame):
    max_date = df["date"].max()
    if max_date >= pd.to_datetime(SPLIT_PRESET_2025["TEST_END"]):
        log("choose_split_preset 완료: 2025 preset 선택")
        return SPLIT_PRESET_2025
    if max_date >= pd.to_datetime(SPLIT_PRESET_2024["TEST_END"]):
        log("choose_split_preset 완료: 2024 preset 선택")
        return SPLIT_PRESET_2024
    raise ValueError(
        f"분할 가능한 날짜 범위가 부족합니다. max_date={max_date}\n"
        f"2024 preset test end={SPLIT_PRESET_2024['TEST_END']}"
    )


def split_by_date(df: pd.DataFrame, split_cfg: dict):
    train_end = pd.to_datetime(split_cfg["TRAIN_END"])
    val_start = pd.to_datetime(split_cfg["VAL_START"])
    val_end = pd.to_datetime(split_cfg["VAL_END"])
    test_start = pd.to_datetime(split_cfg["TEST_START"])
    test_end = pd.to_datetime(split_cfg["TEST_END"])

    train_df = df.loc[df["date"] <= train_end].copy()
    val_df = df.loc[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
    test_df = df.loc[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"분할 결과가 비었습니다.\n"
            f"train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}\n"
            f"전체 date 범위: {df['date'].min()} ~ {df['date'].max()}\n"
            f"split_cfg: {split_cfg}"
        )

    log(f"split_by_date 완료: {split_cfg}")
    return train_df, val_df, test_df


def get_feature_columns() -> List[str]:
    return [
        "trend_idx",
        "avg_temp_c",
        "temp_range_c",
        "precip_duration_hr",
        "daily_precip_mm",
        "avg_wind_speed_m_s",
        "avg_rel_humidity_pct",
        "is_rainy",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "temp_x_rain",
        "humidity_x_rain",
        "precip_x_humidity",
        "wind_x_rain",
    ]


def make_model_matrices(train_df, val_df, test_df):
    feature_cols = get_feature_columns()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET_COL].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    log("make_model_matrices 완료")
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_preprocessing() -> Dict:
    log("\n==============================")
    log("1) 전처리 시작")
    log("==============================")

    daily_bike = load_bike_daily_from_files(RAW_BIKE_DIR)
    weather_raw = load_weather_raw(RAW_WEATHER_DIR)
    weather_daily = preprocess_weather_daily(weather_raw)
    merged = make_features(daily_bike, weather_daily)

    split_cfg = choose_split_preset(merged)
    train_df, val_df, test_df = split_by_date(merged, split_cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = make_model_matrices(train_df, val_df, test_df)

    daily_bike.to_csv(REFORMED_DIR / "daily_bike.csv", index=False, encoding="utf-8-sig")
    weather_daily.to_csv(REFORMED_DIR / "weather_daily.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(REFORMED_DIR / "merged_features.csv", index=False, encoding="utf-8-sig")

    train_df.to_csv(REFORMED_DIR / "train_df.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(REFORMED_DIR / "val_df.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(REFORMED_DIR / "test_df.csv", index=False, encoding="utf-8-sig")

    X_train.to_csv(REFORMED_DIR / "X_train.csv", index=False, encoding="utf-8-sig")
    X_val.to_csv(REFORMED_DIR / "X_val.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(REFORMED_DIR / "X_test.csv", index=False, encoding="utf-8-sig")

    y_train.to_csv(REFORMED_DIR / "y_train.csv", index=False, encoding="utf-8-sig")
    y_val.to_csv(REFORMED_DIR / "y_val.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv(REFORMED_DIR / "y_test.csv", index=False, encoding="utf-8-sig")

    result_summary = {
        "daily_bike_rows": len(daily_bike),
        "weather_daily_rows": len(weather_daily),
        "merged_rows": len(merged),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "merged_start": str(merged["date"].min()),
        "merged_end": str(merged["date"].max()),
        "split_cfg": split_cfg,
    }

    log(f"전처리 완료. 저장 위치: {REFORMED_DIR}")
    return result_summary


def ensure_preprocessed_files():
    required_files = [
        REFORMED_DIR / "X_train.csv",
        REFORMED_DIR / "X_val.csv",
        REFORMED_DIR / "X_test.csv",
        REFORMED_DIR / "y_train.csv",
        REFORMED_DIR / "y_val.csv",
        REFORMED_DIR / "y_test.csv",
        REFORMED_DIR / "merged_features.csv",
        REFORMED_DIR / DATE_ORIGIN_FILE,
    ]
    if all(path.exists() for path in required_files):
        return
    log("전처리 결과 파일이 없어 자동으로 전처리를 실행합니다.")
    run_preprocessing()


def load_preprocessed_split_data():
    ensure_preprocessed_files()

    X_train = pd.read_csv(REFORMED_DIR / "X_train.csv")
    X_val = pd.read_csv(REFORMED_DIR / "X_val.csv")
    X_test = pd.read_csv(REFORMED_DIR / "X_test.csv")

    y_train = pd.read_csv(REFORMED_DIR / "y_train.csv").iloc[:, 0]
    y_val = pd.read_csv(REFORMED_DIR / "y_val.csv").iloc[:, 0]
    y_test = pd.read_csv(REFORMED_DIR / "y_test.csv").iloc[:, 0]

    train_df = pd.read_csv(REFORMED_DIR / "train_df.csv")
    val_df = pd.read_csv(REFORMED_DIR / "val_df.csv")
    test_df = pd.read_csv(REFORMED_DIR / "test_df.csv")
    train_df["date"] = pd.to_datetime(train_df["date"])
    val_df["date"] = pd.to_datetime(val_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    date_origin = pd.to_datetime((REFORMED_DIR / DATE_ORIGIN_FILE).read_text(encoding="utf-8").strip())

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "date_origin": date_origin,
    }


def build_prediction_input_row(
    feature_columns,
    date_origin,
    target_date,
    avg_temp_c,
    min_temp_c,
    max_temp_c,
    precip_duration_hr,
    daily_precip_mm,
    avg_wind_speed_m_s,
    avg_rel_humidity_pct,
):
    ts = pd.to_datetime(target_date)
    temp_range_c = float(max_temp_c - min_temp_c)
    is_rainy = int(daily_precip_mm > 0)
    dayofweek = int(ts.dayofweek)
    month = int(ts.month)
    dayofyear = int(ts.dayofyear)

    row = {
        "trend_idx": int((ts - pd.to_datetime(date_origin)).days),
        "avg_temp_c": float(avg_temp_c),
        "temp_range_c": temp_range_c,
        "precip_duration_hr": float(precip_duration_hr),
        "daily_precip_mm": float(daily_precip_mm),
        "avg_wind_speed_m_s": float(avg_wind_speed_m_s),
        "avg_rel_humidity_pct": float(avg_rel_humidity_pct),
        "is_rainy": is_rainy,
        "is_weekend": int(dayofweek >= 5),
        "is_month_start": int(ts.is_month_start),
        "is_month_end": int(ts.is_month_end),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dow_sin": np.sin(2 * np.pi * dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * dayofweek / 7),
        "doy_sin": np.sin(2 * np.pi * dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * dayofyear / 365.25),
        "temp_x_rain": float(avg_temp_c) * is_rainy,
        "humidity_x_rain": float(avg_rel_humidity_pct) * is_rainy,
        "precip_x_humidity": float(daily_precip_mm) * float(avg_rel_humidity_pct),
        "wind_x_rain": float(avg_wind_speed_m_s) * is_rainy,
    }

    return pd.DataFrame([{col: row.get(col, 0) for col in feature_columns}], columns=feature_columns)


def load_default_prediction_values():
    defaults = {
        "target_date": pd.Timestamp.today().date(),
        "avg_temp_c": 15.0,
        "min_temp_c": 10.0,
        "max_temp_c": 20.0,
        "precip_duration_hr": 0.0,
        "daily_precip_mm": 0.0,
        "avg_wind_speed_m_s": 2.5,
        "avg_rel_humidity_pct": 60.0,
    }

    merged_path = REFORMED_DIR / "merged_features.csv"
    if not merged_path.exists():
        return defaults

    try:
        df = pd.read_csv(merged_path)
        df["date"] = pd.to_datetime(df["date"])
        last_row = df.iloc[-1]
        defaults["target_date"] = last_row["date"].date()
        defaults["avg_temp_c"] = float(last_row["avg_temp_c"])
        defaults["min_temp_c"] = float(last_row["min_temp_c"])
        defaults["max_temp_c"] = float(last_row["max_temp_c"])
        defaults["precip_duration_hr"] = float(last_row["precip_duration_hr"])
        defaults["daily_precip_mm"] = float(last_row["daily_precip_mm"])
        defaults["avg_wind_speed_m_s"] = float(last_row["avg_wind_speed_m_s"])
        defaults["avg_rel_humidity_pct"] = float(last_row["avg_rel_humidity_pct"])
    except Exception:
        pass

    return defaults


def main():
    summary = run_preprocessing()
    print("\n[전처리 요약]")
    for k, v in summary.items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
