"""
============================================================
제출용 단일 파이썬 파일
- 전처리
- 선형회귀 학습 / 평가
- 상관관계 히트맵 생성
- ydata-profiling HTML 보고서 생성
- Streamlit 예측 UI 지원

실행 방법
1) 일반 파이썬 실행
   python merged_submission_streamlit_predict_ui.py

2) Streamlit 예측 UI 실행
   streamlit run merged_submission_streamlit_predict_ui.py

필요 패키지 예시
   pip install pandas matplotlib seaborn scikit-learn ydata-profiling openpyxl streamlit
============================================================
"""

# ============================================================
# 라이브러리 import
# ============================================================
import os
import glob
import re
from pathlib import Path
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from ydata_profiling import ProfileReport

# Streamlit은 선택적으로 사용한다.
# 일반 python 실행에서는 없어도 되도록 처리한다.
try:
    import streamlit as st
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = None
except Exception:
    st = None
    get_script_run_ctx = None


# ============================================================
# 전역 설정
# ============================================================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
pd.options.display.float_format = "{:.2f}".format

TARGET_COL = "daily_rental_count"

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


# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent


def find_base_dir() -> Path:
    """
    dataset 폴더가 존재하는 상위 경로를 자동 탐색한다.
    """
    candidates = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR.parent.parent,
    ]

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


# ============================================================
# 유틸리티
# ============================================================
def log(message: str):
    """
    일반 실행 / Streamlit 실행 모두에서 사용할 공통 로그 함수.
    """
    print(message)


def is_running_in_streamlit() -> bool:
    """
    현재 streamlit run 으로 실행 중인지 판별한다.
    """
    if st is None or get_script_run_ctx is None:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def normalize_colname(col: str) -> str:
    """
    컬럼명 비교를 쉽게 하기 위해 공백과 줄바꿈을 제거한다.
    """
    col = str(col).strip()
    col = col.replace("\n", "")
    col = re.sub(r"\s+", "", col)
    return col


def find_col_by_keywords(columns, keywords):
    """
    전달된 컬럼 목록에서 특정 키워드들을 모두 포함하는 컬럼명을 찾는다.
    """
    normalized = {c: normalize_colname(c) for c in columns}

    for original, normed in normalized.items():
        if all(keyword in normed for keyword in keywords):
            log(f"find_col_by_keywords 완료: {keywords} -> {original}")
            return original

    log(f"find_col_by_keywords 완료: {keywords} -> None")
    return None


def collect_files(data_dir: Path):
    """
    폴더 안의 csv / xlsx / xls 파일을 모두 수집한다.
    """
    files = []
    for pattern in ["*.csv", "*.xlsx", "*.xls"]:
        files.extend(glob.glob(str(data_dir / pattern)))

    files = sorted(files)
    log(f"collect_files 완료: {data_dir} / {len(files)}개")
    return files


def detect_csv_header_and_encoding(path: str):
    """
    CSV 파일 헤더를 읽으면서 인코딩을 자동 탐색한다.
    """
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
        f"CSV 헤더 읽기 실패: {path}\n"
        f"시도한 인코딩:\n" + "\n".join(errors_list)
    )


def read_weather_table_flexible(path: str) -> pd.DataFrame:
    """
    날씨 파일을 확장자에 따라 유연하게 읽는다.
    """
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
            f"날씨 CSV 파일 읽기 실패: {path}\n"
            f"시도한 인코딩:\n" + "\n".join(errors_list)
        )

    raise ValueError(f"지원하지 않는 파일 형식: {path}")


# ============================================================
# 1. 전처리 함수들
# ============================================================
def load_bike_daily_from_files(bike_dir: Path) -> pd.DataFrame:
    """
    따릉이 원본 파일들을 읽어 일별 대여건수 데이터로 변환한다.
    """
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
                    f"{path} 에서 '대여일시' 컬럼을 찾지 못했습니다.\n"
                    f"현재 컬럼: {df.columns.tolist()}"
                )

            temp = df[[rental_dt_col]].copy()
            temp[rental_dt_col] = pd.to_datetime(temp[rental_dt_col], errors="coerce")
            temp = temp.dropna(subset=[rental_dt_col]).copy()
            temp["date"] = temp[rental_dt_col].dt.floor("D")

            daily = temp.groupby("date").size().reset_index(name="daily_rental_count")

            log(
                f"[bike daily range] {os.path.basename(path)} -> "
                f"{daily['date'].min()} ~ {daily['date'].max()} / rows={len(daily)}"
            )

            daily_parts.append(daily)
            del df, temp, daily
            continue

        if ext == ".csv":
            columns, enc = detect_csv_header_and_encoding(path)
            rental_dt_col = find_col_by_keywords(columns, ["대여일시"])
            if rental_dt_col is None:
                raise KeyError(
                    f"{path} 에서 '대여일시' 컬럼을 찾지 못했습니다.\n"
                    f"현재 컬럼: {columns}"
                )

            chunk_parts = []

            for chunk in pd.read_csv(
                path,
                encoding=enc,
                usecols=[rental_dt_col],
                chunksize=200000,
                low_memory=False
            ):
                chunk[rental_dt_col] = pd.to_datetime(chunk[rental_dt_col], errors="coerce")
                chunk = chunk.dropna(subset=[rental_dt_col]).copy()
                chunk["date"] = chunk[rental_dt_col].dt.floor("D")

                daily_chunk = (
                    chunk.groupby("date")
                    .size()
                    .reset_index(name="daily_rental_count")
                )
                chunk_parts.append(daily_chunk)

            if not chunk_parts:
                log(f"[warning] chunk 결과 없음: {path}")
                continue

            daily = pd.concat(chunk_parts, ignore_index=True)
            daily = (
                daily.groupby("date", as_index=False)["daily_rental_count"]
                .sum()
                .sort_values("date")
                .reset_index(drop=True)
            )

            log(
                f"[bike daily range] {os.path.basename(path)} -> "
                f"{daily['date'].min()} ~ {daily['date'].max()} / rows={len(daily)}"
            )

            daily_parts.append(daily)
            del chunk_parts, daily
            continue

        raise ValueError(f"지원하지 않는 파일 형식: {path}")

    if not daily_parts:
        raise ValueError("따릉이 일별 데이터가 하나도 생성되지 않았습니다.")

    daily_bike = pd.concat(daily_parts, ignore_index=True)
    daily_bike = (
        daily_bike.groupby("date", as_index=False)["daily_rental_count"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    log("load_bike_daily_from_files 완료")
    return daily_bike


def load_weather_raw(weather_dir: Path) -> pd.DataFrame:
    """
    날씨 원본 파일들을 모두 읽어서 하나의 DataFrame으로 결합한다.
    """
    files = collect_files(weather_dir)
    if not files:
        raise FileNotFoundError(f"날씨 파일을 찾지 못했습니다: {weather_dir}")

    dfs = []

    for path in files:
        log(f"[weather] 읽는 중: {path}")
        try:
            df = read_weather_table_flexible(path)
            df.columns = [str(c).strip() for c in df.columns]
            df["__source_file"] = os.path.basename(path)
            dfs.append(df)
        except Exception as e:
            raise RuntimeError(f"날씨 파일 읽기 실패: {path}\n원인: {e}") from e

    weather_raw = pd.concat(dfs, ignore_index=True)
    weather_raw = weather_raw.drop_duplicates().reset_index(drop=True)

    log("load_weather_raw 완료")
    return weather_raw


def preprocess_weather_daily(weather_raw: pd.DataFrame) -> pd.DataFrame:
    """
    날씨 원본 테이블에서 필요한 일별 변수만 추출하고 정리한다.
    """
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
        raise KeyError(
            f"날씨 필수 컬럼이 없습니다: {missing}\n"
            f"현재 컬럼: {weather.columns.tolist()}"
        )

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


def make_features(daily_bike: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    """
    따릉이 데이터와 날씨 데이터를 결합하고, 모델 학습에 사용할 파생변수를 생성한다.
    """
    df = pd.merge(
        daily_bike,
        weather_daily,
        on="date",
        how="inner"
    ).sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_rainy"] = (df["daily_precip_mm"].fillna(0) > 0).astype(int)

    def month_to_season(m):
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "spring"
        elif m in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(month_to_season)

    # 과거값 기반 파생변수
    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_7"] = df[TARGET_COL].shift(7)
    df["rolling_mean_7"] = df[TARGET_COL].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df[TARGET_COL].shift(1).rolling(7).std()
    df["rolling_mean_14"] = df[TARGET_COL].shift(1).rolling(14).mean()

    df["temp_range_c"] = df["max_temp_c"] - df["min_temp_c"]

    df = df.dropna().reset_index(drop=True)

    log("make_features 완료")
    return df


def choose_split_preset(df: pd.DataFrame):
    """
    데이터 최대 날짜 범위를 보고 분할 preset을 자동 선택한다.
    """
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
    """
    날짜 기준으로 train / validation / test 데이터를 분할한다.
    """
    train_end = pd.to_datetime(split_cfg["TRAIN_END"])
    val_start = pd.to_datetime(split_cfg["VAL_START"])
    val_end = pd.to_datetime(split_cfg["VAL_END"])
    test_start = pd.to_datetime(split_cfg["TEST_START"])
    test_end = pd.to_datetime(split_cfg["TEST_END"])

    train_mask = df["date"] <= train_end
    val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
    test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"분할 결과가 비었습니다.\n"
            f"train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}\n"
            f"전체 date 범위: {df['date'].min()} ~ {df['date'].max()}\n"
            f"split_cfg: {split_cfg}"
        )

    log(f"split_by_date 완료: {split_cfg}")
    return train_df, val_df, test_df


def make_model_matrices(train_df, val_df, test_df):
    """
    모델 학습용 X, y 행렬을 만든다.
    계절 변수는 one-hot encoding 한다.
    """
    feature_cols = [
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "precip_duration_hr",
        "daily_precip_mm",
        "avg_wind_speed_m_s",
        "avg_rel_humidity_pct",
        "month",
        "day",
        "dayofweek",
        "weekofyear",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_rainy",
        "temp_range_c",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "season",
    ]

    all_df = pd.concat([train_df, val_df, test_df], axis=0).copy()
    all_df = pd.get_dummies(all_df, columns=["season"], drop_first=False)

    y = all_df[TARGET_COL].copy()
    X = all_df.drop(columns=["date", TARGET_COL])

    season_dummy_cols = [c for c in X.columns if c.startswith("season_")]
    final_feature_cols = [c for c in X.columns if c in feature_cols or c in season_dummy_cols]
    X = X[final_feature_cols].copy()

    train_len = len(train_df)
    val_len = len(val_df)

    X_train = X.iloc[:train_len].reset_index(drop=True)
    y_train = y.iloc[:train_len].reset_index(drop=True)

    X_val = X.iloc[train_len:train_len + val_len].reset_index(drop=True)
    y_val = y.iloc[train_len:train_len + val_len].reset_index(drop=True)

    X_test = X.iloc[train_len + val_len:].reset_index(drop=True)
    y_test = y.iloc[train_len + val_len:].reset_index(drop=True)

    log("make_model_matrices 완료")
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_preprocessing():
    """
    전처리 전체 파이프라인을 실행하고 CSV를 저장한다.
    """
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


# ============================================================
# 2. 선형회귀 학습 / 평가 / 예측용 모델 준비
# ============================================================
def ensure_preprocessed_files():
    """
    예측 또는 학습 전에 필요한 전처리 결과 파일이 없으면 자동으로 생성한다.
    """
    required_files = [
        REFORMED_DIR / "X_train.csv",
        REFORMED_DIR / "X_val.csv",
        REFORMED_DIR / "X_test.csv",
        REFORMED_DIR / "y_train.csv",
        REFORMED_DIR / "y_val.csv",
        REFORMED_DIR / "y_test.csv",
        REFORMED_DIR / "merged_features.csv",
    ]

    if all(path.exists() for path in required_files):
        return

    log("전처리 결과 파일이 없어 자동으로 전처리를 실행합니다.")
    run_preprocessing()


def train_model_for_inference():
    """
    예측용 모델을 준비한다.

    동작 방식:
    1) X_train으로 성능 평가용 모델을 학습
    2) Validation/Test R² 계산
    3) 실제 예측용 모델은 X_train + X_val 로 다시 학습
       (조금 더 많은 데이터를 사용하기 위함)
    """
    ensure_preprocessed_files()

    X_train = pd.read_csv(REFORMED_DIR / "X_train.csv")
    X_val = pd.read_csv(REFORMED_DIR / "X_val.csv")
    X_test = pd.read_csv(REFORMED_DIR / "X_test.csv")

    y_train = pd.read_csv(REFORMED_DIR / "y_train.csv").iloc[:, 0]
    y_val = pd.read_csv(REFORMED_DIR / "y_val.csv").iloc[:, 0]
    y_test = pd.read_csv(REFORMED_DIR / "y_test.csv").iloc[:, 0]

    # 평가용 모델
    eval_model = LinearRegression()
    eval_model.fit(X_train, y_train)

    r2_train = eval_model.score(X_train, y_train)
    r2_val = eval_model.score(X_val, y_val)
    r2_test = eval_model.score(X_test, y_test)

    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": eval_model.coef_
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)
    coef_df.to_csv(MODEL_OUTPUT_DIR / "linear_regression_coefficients.csv", index=False, encoding="utf-8-sig")

    # 실제 예측용 모델: train + val
    X_infer = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_infer = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    inference_model = LinearRegression()
    inference_model.fit(X_infer, y_infer)

    return {
        "model": inference_model,
        "feature_columns": X_train.columns.tolist(),
        "r2_train": float(r2_train),
        "r2_val": float(r2_val),
        "r2_test": float(r2_test),
        "intercept": float(eval_model.intercept_),
        "top_coefficients": coef_df[["feature", "coefficient"]].head(20),
    }


def run_linear_regression(show_plots: bool = True):
    """
    원래 제출 흐름을 유지하기 위한 학습 / 평가 / 시각화 함수.
    """
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

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    r2_train = model.score(X_train, y_train)
    r2_val = model.score(X_val, y_val)
    r2_test = model.score(X_test, y_test)

    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)
    coef_df.to_csv(MODEL_OUTPUT_DIR / "linear_regression_coefficients.csv", index=False, encoding="utf-8-sig")

    train_result = pd.DataFrame({
        "date": train_df["date"],
        "actual": y_train,
        "predicted": y_train_pred
    })
    val_result = pd.DataFrame({
        "date": val_df["date"],
        "actual": y_val,
        "predicted": y_val_pred
    })
    test_result = pd.DataFrame({
        "date": test_df["date"],
        "actual": y_test,
        "predicted": y_test_pred
    })

    train_result.to_csv(MODEL_OUTPUT_DIR / "train_prediction_result.csv", index=False, encoding="utf-8-sig")
    val_result.to_csv(MODEL_OUTPUT_DIR / "val_prediction_result.csv", index=False, encoding="utf-8-sig")
    test_result.to_csv(MODEL_OUTPUT_DIR / "test_prediction_result.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    min_val = min(y_val.min(), y_val_pred.min())
    max_val = max(y_val.max(), y_val_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("실제 대여건수")
    plt.ylabel("예측 대여건수")
    plt.title("Validation 실제값 vs 예측값 산점도")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / "val_scatter.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("실제 대여건수")
    plt.ylabel("예측 대여건수")
    plt.title("Test 실제값 vs 예측값 산점도")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / "test_scatter.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(val_result["date"], val_result["actual"], label="실제 대여건수")
    plt.plot(val_result["date"], val_result["predicted"], label="예측 대여건수")
    plt.xlabel("날짜")
    plt.ylabel("대여건수")
    plt.title("Validation 날짜별 실제값 vs 예측값")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / "val_timeseries.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(test_result["date"], test_result["actual"], label="실제 대여건수")
    plt.plot(test_result["date"], test_result["predicted"], label="예측 대여건수")
    plt.xlabel("날짜")
    plt.ylabel("대여건수")
    plt.title("Test 날짜별 실제값 vs 예측값")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_OUTPUT_DIR / "test_timeseries.png", dpi=150)
    if show_plots:
        plt.show()
    else:
        plt.close()

    return {
        "r2_train": float(r2_train),
        "r2_val": float(r2_val),
        "r2_test": float(r2_test),
        "intercept": float(model.intercept_),
        "top_coefficients": coef_df[["feature", "coefficient"]].head(20),
    }


# ============================================================
# 3. 히트맵 / 프로파일링
# ============================================================
def run_heatmap(show_plot: bool = True):
    """
    merged_features.csv를 바탕으로 상관관계 히트맵을 생성하고 저장한다.
    """
    ensure_preprocessed_files()

    df = pd.read_csv(REFORMED_DIR / "merged_features.csv")

    cols = [
        "daily_rental_count",
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "precip_duration_hr",
        "daily_precip_mm",
        "avg_wind_speed_m_s",
        "avg_rel_humidity_pct",
        "is_weekend",
        "is_rainy",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_mean_14",
    ]

    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
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
    """
    merged_features.csv를 바탕으로 ydata-profiling HTML 리포트를 생성한다.
    """
    ensure_preprocessed_files()

    df = pd.read_csv(REFORMED_DIR / "merged_features.csv")

    profile = ProfileReport(
        df,
        title="Ddarungi Demand Prediction Profiling Report",
        explorative=True
    )

    output_file = REPORT_DIR / "ddarungi_profiling_report.html"
    profile.to_file(output_file)

    log(f"프로파일링 보고서 생성 완료: {output_file}")
    return output_file


def run_full_pipeline(show_plots: bool = True):
    """
    일반 python 실행 시 전체 파이프라인을 순서대로 수행한다.
    """
    preprocess_summary = run_preprocessing()
    metrics = run_linear_regression(show_plots=show_plots)
    heatmap_path = run_heatmap(show_plot=show_plots)
    profiling_path = run_profiling_report()

    return {
        "preprocess_summary": preprocess_summary,
        "metrics": metrics,
        "heatmap_path": heatmap_path,
        "profiling_path": profiling_path,
    }


# ============================================================
# 4. 예측 입력 데이터 생성
# ============================================================
def infer_season_from_month(month: int) -> str:
    """
    월(month)로부터 계절 문자열을 반환한다.
    """
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def build_prediction_input_row(
    feature_columns,
    target_date,
    avg_temp_c,
    min_temp_c,
    max_temp_c,
    precip_duration_hr,
    daily_precip_mm,
    avg_wind_speed_m_s,
    avg_rel_humidity_pct,
    lag_1,
    lag_7,
    rolling_mean_7,
    rolling_std_7,
    rolling_mean_14,
):
    """
    Streamlit 입력값을 모델 입력 형식 1행 DataFrame으로 변환한다.

    사용자가 직접 입력한 값:
    - 날짜
    - 날씨 변수
    - lag / rolling 관련 변수

    날짜로부터 자동 파생되는 값:
    - month, day, dayofweek, weekofyear
    - is_weekend, is_month_start, is_month_end
    - is_rainy, temp_range_c, season
    """
    ts = pd.to_datetime(target_date)

    row = {col: 0 for col in feature_columns}

    # 직접 입력값
    direct_values = {
        "avg_temp_c": avg_temp_c,
        "min_temp_c": min_temp_c,
        "max_temp_c": max_temp_c,
        "precip_duration_hr": precip_duration_hr,
        "daily_precip_mm": daily_precip_mm,
        "avg_wind_speed_m_s": avg_wind_speed_m_s,
        "avg_rel_humidity_pct": avg_rel_humidity_pct,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7,
        "rolling_mean_14": rolling_mean_14,
        "month": int(ts.month),
        "day": int(ts.day),
        "dayofweek": int(ts.dayofweek),
        "weekofyear": int(ts.isocalendar().week),
        "is_weekend": int(ts.dayofweek >= 5),
        "is_month_start": int(ts.is_month_start),
        "is_month_end": int(ts.is_month_end),
        "is_rainy": int(daily_precip_mm > 0),
        "temp_range_c": float(max_temp_c - min_temp_c),
    }

    for key, value in direct_values.items():
        if key in row:
            row[key] = value

    # season one-hot 처리
    season = infer_season_from_month(int(ts.month))
    season_col = f"season_{season}"
    if season_col in row:
        row[season_col] = 1

    input_df = pd.DataFrame([row], columns=feature_columns)
    return input_df, season


def load_default_prediction_values():
    """
    Streamlit 입력창의 기본값을 준비한다.
    merged_features.csv가 있으면 최근 데이터 기준으로 기본값을 만든다.
    """
    defaults = {
        "target_date": pd.Timestamp.today().date(),
        "avg_temp_c": 15.0,
        "min_temp_c": 10.0,
        "max_temp_c": 20.0,
        "precip_duration_hr": 0.0,
        "daily_precip_mm": 0.0,
        "avg_wind_speed_m_s": 2.5,
        "avg_rel_humidity_pct": 60.0,
        "lag_1": 50000.0,
        "lag_7": 48000.0,
        "rolling_mean_7": 49000.0,
        "rolling_std_7": 5000.0,
        "rolling_mean_14": 49500.0,
    }

    merged_path = REFORMED_DIR / "merged_features.csv"
    if not merged_path.exists():
        return defaults

    try:
        df = pd.read_csv(merged_path)
        df["date"] = pd.to_datetime(df["date"])
        last_row = df.iloc[-1]

        defaults["target_date"] = (last_row["date"] + timedelta(days=1)).date()
        defaults["avg_temp_c"] = float(last_row["avg_temp_c"])
        defaults["min_temp_c"] = float(last_row["min_temp_c"])
        defaults["max_temp_c"] = float(last_row["max_temp_c"])
        defaults["precip_duration_hr"] = float(last_row["precip_duration_hr"])
        defaults["daily_precip_mm"] = float(last_row["daily_precip_mm"])
        defaults["avg_wind_speed_m_s"] = float(last_row["avg_wind_speed_m_s"])
        defaults["avg_rel_humidity_pct"] = float(last_row["avg_rel_humidity_pct"])
        defaults["lag_1"] = float(last_row["lag_1"])
        defaults["lag_7"] = float(last_row["lag_7"])
        defaults["rolling_mean_7"] = float(last_row["rolling_mean_7"])
        defaults["rolling_std_7"] = float(last_row["rolling_std_7"])
        defaults["rolling_mean_14"] = float(last_row["rolling_mean_14"])
    except Exception:
        pass

    return defaults


# ============================================================
# 5. Streamlit 예측 UI
# ============================================================
def run_streamlit_prediction_app():
    """
    Streamlit UI:
    - 날짜 / 날씨 / lag 값 입력
    - 버튼 클릭 시 모델 학습 후 예측값 출력
    """
    st.set_page_config(page_title="따릉이 수요 예측 UI", layout="wide")
    st.title("따릉이 대여건수 예측 UI")
    st.caption("입력값을 넣고 버튼을 누르면 선형회귀 모델을 학습한 뒤 예측값을 출력합니다.")

    st.info(
        "이 모델은 lag_1, lag_7, rolling_mean_7, rolling_std_7, rolling_mean_14 같은 과거 대여량 기반 변수를 사용합니다. "
        "따라서 미래 날짜를 예측하려면 최근 대여량 정보를 함께 넣어야 합니다."
    )

    with st.expander("현재 경로 및 데이터 상태", expanded=False):
        st.write(f"BASE_DIR: {BASE_DIR}")
        st.write(f"따릉이 원본 폴더: {RAW_BIKE_DIR}")
        st.write(f"날씨 원본 폴더: {RAW_WEATHER_DIR}")
        st.write(f"전처리 결과 폴더: {REFORMED_DIR}")
        st.write(f"모델 출력 폴더: {MODEL_OUTPUT_DIR}")
        st.write(f"리포트 폴더: {REPORT_DIR}")

        required_files = [
            REFORMED_DIR / "X_train.csv",
            REFORMED_DIR / "X_val.csv",
            REFORMED_DIR / "X_test.csv",
            REFORMED_DIR / "merged_features.csv",
        ]
        status_df = pd.DataFrame({
            "file": [p.name for p in required_files],
            "exists": [p.exists() for p in required_files]
        })
        st.dataframe(status_df, use_container_width=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        if st.button("전처리 데이터 준비 / 재생성"):
            with st.spinner("전처리 실행 중입니다..."):
                summary = run_preprocessing()
            st.success("전처리가 완료되었습니다.")
            st.json(summary)

    with col_b:
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
            precip_duration_hr = st.number_input("강수 계속시간(hr)", min_value=0.0, value=float(defaults["precip_duration_hr"]), step=0.1)
            daily_precip_mm = st.number_input("일강수량(mm)", min_value=0.0, value=float(defaults["daily_precip_mm"]), step=0.1)
            avg_wind_speed_m_s = st.number_input("평균 풍속(m/s)", min_value=0.0, value=float(defaults["avg_wind_speed_m_s"]), step=0.1)
            avg_rel_humidity_pct = st.number_input("평균 상대습도(%)", min_value=0.0, max_value=100.0, value=float(defaults["avg_rel_humidity_pct"]), step=0.1)

        with right:
            lag_1 = st.number_input("전일 대여건수 (lag_1)", min_value=0.0, value=float(defaults["lag_1"]), step=100.0)
            lag_7 = st.number_input("7일 전 대여건수 (lag_7)", min_value=0.0, value=float(defaults["lag_7"]), step=100.0)
            rolling_mean_7 = st.number_input("직전 7일 평균 대여건수", min_value=0.0, value=float(defaults["rolling_mean_7"]), step=100.0)
            rolling_std_7 = st.number_input("직전 7일 표준편차", min_value=0.0, value=float(defaults["rolling_std_7"]), step=100.0)
            rolling_mean_14 = st.number_input("직전 14일 평균 대여건수", min_value=0.0, value=float(defaults["rolling_mean_14"]), step=100.0)

        submitted = st.form_submit_button("모델 학습 후 예측하기")

    if submitted:
        try:
            with st.spinner("모델을 준비하고 예측 중입니다..."):
                model_info = train_model_for_inference()

                input_df, inferred_season = build_prediction_input_row(
                    feature_columns=model_info["feature_columns"],
                    target_date=target_date,
                    avg_temp_c=avg_temp_c,
                    min_temp_c=min_temp_c,
                    max_temp_c=max_temp_c,
                    precip_duration_hr=precip_duration_hr,
                    daily_precip_mm=daily_precip_mm,
                    avg_wind_speed_m_s=avg_wind_speed_m_s,
                    avg_rel_humidity_pct=avg_rel_humidity_pct,
                    lag_1=lag_1,
                    lag_7=lag_7,
                    rolling_mean_7=rolling_mean_7,
                    rolling_std_7=rolling_std_7,
                    rolling_mean_14=rolling_mean_14,
                )

                predicted_value = float(model_info["model"].predict(input_df)[0])

            st.success("예측이 완료되었습니다.")

            m1, m2, m3 = st.columns(3)
            m1.metric("예측 대여건수", f"{predicted_value:,.0f}")
            m2.metric("Validation R²", f'{model_info["r2_val"]:.4f}')
            m3.metric("Test R²", f'{model_info["r2_test"]:.4f}')

            with st.expander("모델 입력값(파생변수 포함) 보기", expanded=True):
                derived_preview = input_df.copy()
                derived_preview.insert(0, "season_label", inferred_season)
                derived_preview.insert(0, "target_date", str(pd.to_datetime(target_date).date()))
                st.dataframe(derived_preview, use_container_width=True)

            with st.expander("상위 회귀계수 보기", expanded=False):
                st.dataframe(model_info["top_coefficients"], use_container_width=True)

            with st.expander("해석 시 주의점", expanded=False):
                st.markdown(
                    """
                    - 이 예측은 선형회귀 기반입니다.
                    - 날짜 자체보다도 `lag_1`, `lag_7`, `rolling_mean_7` 같은 최근 대여량 입력의 영향이 큽니다.
                    - 따라서 미래 예측에서는 최근 실제 대여량 정보를 얼마나 정확히 넣느냐가 중요합니다.
                    """
                )

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")


# ============================================================
# 6. 일반 CLI 실행
# ============================================================
def main_cli():
    """
    일반 python 실행 시 전체 파이프라인을 수행한다.
    """
    log("단일 제출용 파이썬 파일 실행 시작")
    log(f"BASE_DIR: {BASE_DIR}")

    result = run_full_pipeline(show_plots=True)

    log("\n모든 작업이 완료되었습니다.")
    log(f"- 전처리 결과: {REFORMED_DIR}")
    log(f"- 모델 결과: {MODEL_OUTPUT_DIR}")
    log(f"- 리포트 결과: {REPORT_DIR}")
    log(f"- Validation R²: {result['metrics']['r2_val']:.4f}")
    log(f"- Test R²: {result['metrics']['r2_test']:.4f}")


# ============================================================
# 실행 진입점
# ============================================================
if __name__ == "__main__":
    if is_running_in_streamlit():
        run_streamlit_prediction_app()
    else:
        main_cli()
