"""
============================================================
제출용 단일 파이썬 파일 (개선 버전)
- 전처리
- 날짜 + 날씨 기반 파생변수 생성
- 여러 회귀 모델 비교 후 Validation R² 기준 최적 모델 자동 선택
- Train / Validation / Test 성능 평가
- 상관관계 히트맵 생성
- ydata-profiling HTML 보고서 생성
- Streamlit 예측 UI 지원

핵심 개선 사항
- 선형회귀 1개만 쓰지 않고 Ridge / RandomForest / GradientBoosting 비교
- 다중공선성이 큰 변수 조합을 정리
- month, dayofweek, dayofyear는 주기(sin/cos) 인코딩 사용
- trend_idx, 비/온도 상호작용 파생변수 추가
- log1p 타깃 변환으로 극단값 영향 완화

실행 방법
1) 일반 파이썬 실행
   python merged_submission_date_weather_only_improved.py

2) Streamlit 예측 UI 실행
   streamlit run merged_submission_date_weather_only_improved.py

필요 패키지 예시
   pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling openpyxl streamlit
============================================================
"""

from __future__ import annotations

import glob
import math
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport

try:
    import streamlit as st
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = None
except Exception:
    st = None
    get_script_run_ctx = None

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
pd.options.display.float_format = "{:.4f}".format

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


def is_running_in_streamlit() -> bool:
    if st is None or get_script_run_ctx is None:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


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

    # 상호작용 변수
    df["temp_x_rain"] = df["avg_temp_c"] * df["is_rainy"]
    df["humidity_x_rain"] = df["avg_rel_humidity_pct"] * df["is_rainy"]
    df["precip_x_humidity"] = df["daily_precip_mm"] * df["avg_rel_humidity_pct"]
    df["wind_x_rain"] = df["avg_wind_speed_m_s"] * df["is_rainy"]

    # 해석용 season은 남기되, 모델 직접 입력에는 넣지 않음
    def month_to_season(m):
        if m in [12, 1, 2]:
            return "winter"
        if m in [3, 4, 5]:
            return "spring"
        if m in [6, 7, 8]:
            return "summer"
        return "autumn"

    df["season"] = df["month"].apply(month_to_season)

    # 날짜 origin 저장
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
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET_COL].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    log("make_model_matrices 완료")
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_preprocessing():
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


# ------------------------------------------------------------
# 모델 학습 / 비교 / 평가
# ------------------------------------------------------------

def get_candidate_specs() -> List[Dict]:
    return [
        {
            "name": "linear_regression_log",
            "factory": lambda: LinearRegression(),
            "use_log_target": True,
        },
        {
            "name": "ridge_alpha_1_log",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
            "use_log_target": True,
        },
        {
            "name": "ridge_alpha_10_log",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]),
            "use_log_target": True,
        },
        {
            "name": "random_forest_log",
            "factory": lambda: RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "use_log_target": True,
        },
        {
            "name": "gradient_boosting_log",
            "factory": lambda: GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            ),
            "use_log_target": True,
        },
    ]


def fit_candidate_model(spec: Dict, X: pd.DataFrame, y: pd.Series):
    model = spec["factory"]()
    if spec["use_log_target"]:
        model.fit(X, np.log1p(y))
    else:
        model.fit(X, y)
    return model


def predict_candidate_model(spec: Dict, model, X: pd.DataFrame) -> np.ndarray:
    pred = model.predict(X)
    if spec["use_log_target"]:
        pred = np.expm1(pred)
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    return pred


def calc_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    safe_true = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / safe_true)) * 100
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape) if not math.isnan(mape) else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
    }


def extract_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    inner = model.named_steps["model"] if isinstance(model, Pipeline) else model

    if hasattr(inner, "coef_"):
        importance = np.abs(np.ravel(inner.coef_))
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "signed_value": np.ravel(inner.coef_),
            "importance_type": "abs_coefficient",
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(inner, "feature_importances_"):
        importance = np.asarray(inner.feature_importances_, dtype=float)
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
            "signed_value": importance,
            "importance_type": "feature_importance",
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    return pd.DataFrame({
        "feature": feature_names,
        "importance": np.nan,
        "signed_value": np.nan,
        "importance_type": "not_supported",
    })


def evaluate_model_candidates():
    data = load_preprocessed_split_data()
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    rows = []
    fitted_models = {}

    for spec in get_candidate_specs():
        model = fit_candidate_model(spec, X_train, y_train)
        fitted_models[spec["name"]] = (spec, model)

        train_pred = predict_candidate_model(spec, model, X_train)
        val_pred = predict_candidate_model(spec, model, X_val)
        test_pred = predict_candidate_model(spec, model, X_test)

        train_metrics = calc_regression_metrics(y_train, train_pred)
        val_metrics = calc_regression_metrics(y_val, val_pred)
        test_metrics = calc_regression_metrics(y_test, test_pred)

        rows.append({
            "model_name": spec["name"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "test_mape": test_metrics["mape"],
            "val_bias": val_metrics["bias"],
            "test_bias": test_metrics["bias"],
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(
        by=["val_r2", "val_mae"],
        ascending=[False, True],
    ).reset_index(drop=True)
    comparison_df.to_csv(MODEL_OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    best_name = comparison_df.iloc[0]["model_name"]
    best_spec, best_eval_model = fitted_models[best_name]

    feature_importance_df = extract_feature_importance(best_eval_model, X_train.columns.tolist())
    feature_importance_df.to_csv(MODEL_OUTPUT_DIR / "best_model_feature_importance.csv", index=False, encoding="utf-8-sig")

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    best_inference_model = fit_candidate_model(best_spec, X_train_val, y_train_val)

    return {
        "comparison_df": comparison_df,
        "best_model_name": best_name,
        "best_spec": best_spec,
        "best_eval_model": best_eval_model,
        "best_inference_model": best_inference_model,
        "feature_importance_df": feature_importance_df,
        **data,
    }


def save_prediction_results(prefix_df: pd.DataFrame, y_true, y_pred, filename: str):
    out = pd.DataFrame({
        "date": prefix_df["date"],
        "actual": y_true,
        "predicted": y_pred,
    })
    out.to_csv(MODEL_OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    return out


def save_scatter_plot(y_true, y_pred, title: str, filename: str, show_plot: bool):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.55)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("실제 대여건수")
    plt.ylabel("예측 대여건수")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / filename, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_timeseries_plot(result_df: pd.DataFrame, title: str, filename: str, show_plot: bool):
    plt.figure(figsize=(12, 6))
    plt.plot(result_df["date"], result_df["actual"], label="실제 대여건수")
    plt.plot(result_df["date"], result_df["predicted"], label="예측 대여건수")
    plt.xlabel("날짜")
    plt.ylabel("대여건수")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / filename, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def run_best_model_pipeline(show_plots: bool = True):
    bundle = evaluate_model_candidates()
    best_spec = bundle["best_spec"]
    best_eval_model = bundle["best_eval_model"]
    best_model_name = bundle["best_model_name"]

    X_train = bundle["X_train"]
    X_val = bundle["X_val"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_val = bundle["y_val"]
    y_test = bundle["y_test"]

    train_pred = predict_candidate_model(best_spec, best_eval_model, X_train)
    val_pred = predict_candidate_model(best_spec, best_eval_model, X_val)
    test_pred = predict_candidate_model(best_spec, best_eval_model, X_test)

    train_metrics = calc_regression_metrics(y_train, train_pred)
    val_metrics = calc_regression_metrics(y_val, val_pred)
    test_metrics = calc_regression_metrics(y_test, test_pred)

    train_result = save_prediction_results(bundle["train_df"], y_train, train_pred, "train_prediction_result.csv")
    val_result = save_prediction_results(bundle["val_df"], y_val, val_pred, "val_prediction_result.csv")
    test_result = save_prediction_results(bundle["test_df"], y_test, test_pred, "test_prediction_result.csv")

    save_scatter_plot(y_val, val_pred, f"Validation 실제값 vs 예측값 산점도 ({best_model_name})", "val_scatter.png", show_plots)
    save_scatter_plot(y_test, test_pred, f"Test 실제값 vs 예측값 산점도 ({best_model_name})", "test_scatter.png", show_plots)
    save_timeseries_plot(val_result, f"Validation 날짜별 실제값 vs 예측값 ({best_model_name})", "val_timeseries.png", show_plots)
    save_timeseries_plot(test_result, f"Test 날짜별 실제값 vs 예측값 ({best_model_name})", "test_timeseries.png", show_plots)

    summary_df = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "validation", **val_metrics},
        {"split": "test", **test_metrics},
    ])
    summary_df.insert(0, "model_name", best_model_name)
    summary_df.to_csv(MODEL_OUTPUT_DIR / "best_model_metrics.csv", index=False, encoding="utf-8-sig")

    return {
        "model_name": best_model_name,
        "r2_train": train_metrics["r2"],
        "r2_val": val_metrics["r2"],
        "r2_test": test_metrics["r2"],
        "mae_val": val_metrics["mae"],
        "mae_test": test_metrics["mae"],
        "rmse_val": val_metrics["rmse"],
        "rmse_test": test_metrics["rmse"],
        "mape_val": val_metrics["mape"],
        "mape_test": test_metrics["mape"],
        "bias_val": val_metrics["bias"],
        "bias_test": test_metrics["bias"],
        "comparison_df": bundle["comparison_df"],
        "feature_importance_df": bundle["feature_importance_df"],
    }


def train_model_for_inference():
    bundle = evaluate_model_candidates()
    metrics_path = MODEL_OUTPUT_DIR / "best_model_metrics.csv"
    if not metrics_path.exists():
        run_best_model_pipeline(show_plots=False)

    metrics_df = pd.read_csv(metrics_path)
    val_row = metrics_df.loc[metrics_df["split"] == "validation"].iloc[0]
    test_row = metrics_df.loc[metrics_df["split"] == "test"].iloc[0]
    train_row = metrics_df.loc[metrics_df["split"] == "train"].iloc[0]

    return {
        "model": bundle["best_inference_model"],
        "model_name": bundle["best_model_name"],
        "feature_columns": bundle["X_train"].columns.tolist(),
        "r2_train": float(train_row["r2"]),
        "r2_val": float(val_row["r2"]),
        "r2_test": float(test_row["r2"]),
        "mae_val": float(val_row["mae"]),
        "mae_test": float(test_row["mae"]),
        "comparison_df": bundle["comparison_df"],
        "top_importance": bundle["feature_importance_df"].head(20),
        "date_origin": bundle["date_origin"],
    }


# ------------------------------------------------------------
# 리포트
# ------------------------------------------------------------

def run_heatmap(show_plot: bool = True):
    ensure_preprocessed_files()
    df = pd.read_csv(REFORMED_DIR / "merged_features.csv")
    cols = [
        TARGET_COL,
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "precip_duration_hr",
        "daily_precip_mm",
        "avg_wind_speed_m_s",
        "avg_rel_humidity_pct",
        "trend_idx",
        "dayofweek",
        "weekofyear",
        "is_weekend",
        "is_rainy",
        "temp_range_c",
        "temp_x_rain",
        "humidity_x_rain",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("따릉이 수요 예측 변수 상관관계 Heatmap")
    plt.tight_layout()
    heatmap_path = REPORT_DIR / "ddarungi_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close()

    log(f"히트맵 저장 완료: {heatmap_path}")
    return heatmap_path


def run_profiling_report():
    ensure_preprocessed_files()
    df = pd.read_csv(REFORMED_DIR / "merged_features.csv")
    profile = ProfileReport(
        df,
        title="Ddarungi Demand Prediction Profiling Report (Improved Date + Weather Model)",
        explorative=True,
    )
    output_file = REPORT_DIR / "ddarungi_profiling_report.html"
    profile.to_file(output_file)
    log(f"프로파일링 보고서 생성 완료: {output_file}")
    return output_file


def run_full_pipeline(show_plots: bool = True):
    preprocess_summary = run_preprocessing()
    metrics = run_best_model_pipeline(show_plots=show_plots)
    heatmap_path = run_heatmap(show_plot=show_plots)
    profiling_path = run_profiling_report()

    return {
        "preprocess_summary": preprocess_summary,
        "metrics": metrics,
        "heatmap_path": heatmap_path,
        "profiling_path": profiling_path,
    }


# ------------------------------------------------------------
# Streamlit 예측 UI용 입력 행 생성
# ------------------------------------------------------------

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

    input_df = pd.DataFrame([{col: row.get(col, 0) for col in feature_columns}], columns=feature_columns)
    return input_df


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


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

def run_streamlit_prediction_app():
    st.set_page_config(page_title="따릉이 수요 예측 UI", layout="wide")
    st.title("따릉이 대여건수 예측 UI")
    st.caption("날짜와 날씨 정보만 입력하면, 여러 후보 모델 중 성능이 가장 좋은 모델로 예측합니다.")
    st.info(
        "이 버전은 lag/rolling 없이 날짜+날씨만 사용합니다. 대신 주기 인코딩, 상호작용 변수, 모델 비교를 추가했습니다."
    )

    with st.expander("현재 경로 및 데이터 상태", expanded=False):
        st.write(f"BASE_DIR: {BASE_DIR}")
        st.write(f"따릉이 원본 폴더: {RAW_BIKE_DIR}")
        st.write(f"날씨 원본 폴더: {RAW_WEATHER_DIR}")
        st.write(f"전처리 결과 폴더: {REFORMED_DIR}")
        st.write(f"모델 출력 폴더: {MODEL_OUTPUT_DIR}")
        st.write(f"리포트 폴더: {REPORT_DIR}")

    left_top, right_top = st.columns([1, 1])

    with left_top:
        if st.button("전처리 데이터 준비 / 재생성"):
            with st.spinner("전처리 실행 중입니다..."):
                summary = run_preprocessing()
            st.success("전처리가 완료되었습니다.")
            st.json(summary)

    with right_top:
        if st.button("부가 리포트 생성 (히트맵 + HTML)"):
            with st.spinner("리포트 생성 중입니다..."):
                heatmap_path = run_heatmap(show_plot=False)
                profiling_path = run_profiling_report()
            st.success("리포트 생성이 완료되었습니다.")
            if Path(heatmap_path).exists():
                st.image(str(heatmap_path), caption="상관관계 히트맵", use_container_width=True)
            st.write(f"프로파일링 HTML 저장 위치: {profiling_path}")

    defaults = load_default_prediction_values()

    st.subheader("예측 입력값")
    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            target_date = st.date_input("예측 날짜", value=defaults["target_date"])
            avg_temp_c = st.number_input("평균기온(°C)", value=float(defaults["avg_temp_c"]), step=0.1)
            min_temp_c = st.number_input("최저기온(°C)", value=float(defaults["min_temp_c"]), step=0.1)
            max_temp_c = st.number_input("최고기온(°C)", value=float(defaults["max_temp_c"]), step=0.1)
            precip_duration_hr = st.number_input(
                "강수 계속시간(hr)",
                min_value=0.0,
                value=float(defaults["precip_duration_hr"]),
                step=0.1,
                help="비가 내린 총 시간을 입력합니다.",
            )

        with right:
            daily_precip_mm = st.number_input("일강수량(mm)", min_value=0.0, value=float(defaults["daily_precip_mm"]), step=0.1)
            avg_wind_speed_m_s = st.number_input("평균 풍속(m/s)", min_value=0.0, value=float(defaults["avg_wind_speed_m_s"]), step=0.1)
            avg_rel_humidity_pct = st.number_input("평균 상대습도(%)", min_value=0.0, max_value=100.0, value=float(defaults["avg_rel_humidity_pct"]), step=0.1)

        submitted = st.form_submit_button("최적 모델로 예측하기")

    if submitted:
        try:
            if min_temp_c > max_temp_c:
                st.error("최저기온은 최고기온보다 클 수 없습니다.")
                st.stop()

            with st.spinner("모델을 준비하고 예측 중입니다..."):
                model_info = train_model_for_inference()
                input_df = build_prediction_input_row(
                    feature_columns=model_info["feature_columns"],
                    date_origin=model_info["date_origin"],
                    target_date=target_date,
                    avg_temp_c=avg_temp_c,
                    min_temp_c=min_temp_c,
                    max_temp_c=max_temp_c,
                    precip_duration_hr=precip_duration_hr,
                    daily_precip_mm=daily_precip_mm,
                    avg_wind_speed_m_s=avg_wind_speed_m_s,
                    avg_rel_humidity_pct=avg_rel_humidity_pct,
                )
                predicted_value = float(model_info["model"].predict(input_df)[0])
                if "log" in model_info["model_name"]:
                    predicted_value = float(np.expm1(predicted_value))
                predicted_value = max(0.0, predicted_value)

            st.success("예측이 완료되었습니다.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("예측 대여건수", f"{predicted_value:,.0f}")
            m2.metric("선택 모델", model_info["model_name"])
            m3.metric("Validation R²", f"{model_info['r2_val']:.4f}")
            m4.metric("Test R²", f"{model_info['r2_test']:.4f}")

            with st.expander("모델 입력값(파생변수 포함) 보기", expanded=True):
                preview = input_df.copy()
                preview.insert(0, "target_date", str(pd.to_datetime(target_date).date()))
                st.dataframe(preview, use_container_width=True)

            with st.expander("모델 비교 결과 보기", expanded=False):
                st.dataframe(model_info["comparison_df"], use_container_width=True)

            with st.expander("상위 중요 변수 보기", expanded=False):
                st.dataframe(model_info["top_importance"], use_container_width=True)

            with st.expander("해석 시 주의점", expanded=False):
                st.markdown(
                    """
                    - 이 예측은 **날짜 + 날씨 기반 모델**입니다.
                    - 성능을 높이기 위해 주기 인코딩, 상호작용 변수, 모델 비교를 추가했습니다.
                    - 그래도 공휴일, 행사, 정책 변화, 돌발 이벤트는 반영하지 못할 수 있습니다.
                    """
                )
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")


# ------------------------------------------------------------
# CLI 실행
# ------------------------------------------------------------

def main_cli():
    log("개선 버전 단일 제출용 파이썬 파일 실행 시작")
    log(f"BASE_DIR: {BASE_DIR}")

    result = run_full_pipeline(show_plots=True)
    metrics = result["metrics"]

    log("\n모든 작업이 완료되었습니다.")
    log(f"- 전처리 결과: {REFORMED_DIR}")
    log(f"- 모델 결과: {MODEL_OUTPUT_DIR}")
    log(f"- 리포트 결과: {REPORT_DIR}")
    log(f"- 선택 모델: {metrics['model_name']}")
    log(f"- Validation R²: {metrics['r2_val']:.4f}")
    log(f"- Test R²: {metrics['r2_test']:.4f}")
    log(f"- Validation MAE: {metrics['mae_val']:.2f}")
    log(f"- Test MAE: {metrics['mae_test']:.2f}")


if __name__ == "__main__":
    if is_running_in_streamlit():
        run_streamlit_prediction_app()
    else:
        main_cli()
