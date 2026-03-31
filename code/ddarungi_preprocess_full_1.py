import os
import glob
import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BIKE_DIR = BASE_DIR / "dataset" / "raw_data" / "bike"
WEATHER_DIR = BASE_DIR / "dataset" / "raw_data" / "weather"
OUTPUT_DIR = BASE_DIR / "dataset" / "reformed_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def normalize_colname(col: str) -> str:
    col = str(col).strip()
    col = col.replace("\n", "")
    col = re.sub(r"\s+", "", col)
    return col


def find_col_by_keywords(columns, keywords):
    norm_cols = {c: normalize_colname(c) for c in columns}
    for original, normed in norm_cols.items():
        if all(k in normed for k in keywords):
            print(f"find_col_by_keywords 완료: {keywords} -> {original}")
            return original

    print(f"find_col_by_keywords 완료: {keywords} -> None")
    return None


def collect_files(data_dir: Path):
    files = []
    for pattern in ["*.csv", "*.xlsx", "*.xls"]:
        files.extend(glob.glob(str(data_dir / pattern)))

    files = sorted(files)
    print(f"collect_files 완료: {data_dir} / {len(files)}개")
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
            print(f"detect_csv_header_and_encoding 완료: {path} / encoding={enc}")
            return header_df.columns.tolist(), enc
        except Exception as e:
            errors_list.append(f"{enc}: {e}")

    raise ValueError(
        f"CSV 헤더 읽기 실패: {path}\n"
        f"시도한 인코딩:\n" + "\n".join(errors_list)
    )


def read_weather_table_flexible(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
        print(f"read_weather_table_flexible 완료: {path}")
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
                print(f"read_weather_table_flexible 완료: {path} / encoding={enc}")
                return df
            except Exception as e:
                errors_list.append(f"{enc}: {e}")

        raise ValueError(
            f"날씨 CSV 파일 읽기 실패: {path}\n"
            f"시도한 인코딩:\n" + "\n".join(errors_list)
        )

    raise ValueError(f"지원하지 않는 파일 형식: {path}")


def load_bike_daily_from_files(bike_dir: Path) -> pd.DataFrame:
    files = collect_files(bike_dir)
    if not files:
        raise FileNotFoundError(f"따릉이 파일을 찾지 못했습니다: {bike_dir}")

    daily_parts = []

    for path in files:
        print(f"[bike] 처리 중: {path}")
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

            daily = (
                temp.groupby("date")
                .size()
                .reset_index(name="daily_rental_count")
            )

            print(
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
                print(f"[warning] chunk 결과 없음: {path}")
                continue

            daily = pd.concat(chunk_parts, ignore_index=True)
            daily = (
                daily.groupby("date", as_index=False)["daily_rental_count"]
                .sum()
                .sort_values("date")
                .reset_index(drop=True)
            )

            print(
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

    print(daily_bike.head())
    print(daily_bike.tail())
    print("daily_bike date range:", daily_bike["date"].min(), "~", daily_bike["date"].max())
    print("load_bike_daily_from_files 완료")
    return daily_bike


def load_weather_raw(weather_dir: Path) -> pd.DataFrame:
    files = collect_files(weather_dir)
    if not files:
        raise FileNotFoundError(f"날씨 파일을 찾지 못했습니다: {weather_dir}")

    dfs = []
    for path in files:
        print(f"[weather] 읽는 중: {path}")
        try:
            df = read_weather_table_flexible(path)
            df.columns = [str(c).strip() for c in df.columns]
            df["__source_file"] = os.path.basename(path)
            dfs.append(df)
        except Exception as e:
            raise RuntimeError(f"날씨 파일 읽기 실패: {path}\n원인: {e}") from e

    weather_raw = pd.concat(dfs, ignore_index=True)
    weather_raw = weather_raw.drop_duplicates().reset_index(drop=True)

    print("load_weather_raw 완료")
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

    print("preprocess_weather_daily 완료")
    return weather


def make_features(daily_bike: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
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

    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_7"] = df[TARGET_COL].shift(7)
    df["rolling_mean_7"] = df[TARGET_COL].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df[TARGET_COL].shift(1).rolling(7).std()
    df["rolling_mean_14"] = df[TARGET_COL].shift(1).rolling(14).mean()
    df["temp_range_c"] = df["max_temp_c"] - df["min_temp_c"]

    df = df.dropna().reset_index(drop=True)

    print("make_features 완료")
    return df


def choose_split_preset(df: pd.DataFrame):
    max_date = df["date"].max()
    if max_date >= pd.to_datetime(SPLIT_PRESET_2025["TEST_END"]):
        split_cfg = SPLIT_PRESET_2025
        print("choose_split_preset 완료: 2025 preset 선택")
        return split_cfg

    if max_date >= pd.to_datetime(SPLIT_PRESET_2024["TEST_END"]):
        split_cfg = SPLIT_PRESET_2024
        print("choose_split_preset 완료: 2024 preset 선택")
        return split_cfg

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

    print(f"split_by_date 완료: {split_cfg}")
    return train_df, val_df, test_df


def make_model_matrices(train_df, val_df, test_df):
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

    print("make_model_matrices 완료")
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    print("전처리 시작")

    daily_bike = load_bike_daily_from_files(BIKE_DIR)
    weather_raw = load_weather_raw(WEATHER_DIR)

    print("[daily_bike_raw_result]", daily_bike.shape)
    print("[weather_raw]", weather_raw.shape)

    weather_daily = preprocess_weather_daily(weather_raw)

    print("[daily_bike]", daily_bike.shape, daily_bike["date"].min(), daily_bike["date"].max())
    print("[weather_daily]", weather_daily.shape, weather_daily["date"].min(), weather_daily["date"].max())

    merged = make_features(daily_bike, weather_daily)

    print("[merged]", merged.shape)
    print("date range:", merged["date"].min(), "~", merged["date"].max())
    print(merged.head())

    split_cfg = choose_split_preset(merged)
    train_df, val_df, test_df = split_by_date(merged, split_cfg)

    print("[train]", train_df.shape, train_df["date"].min(), train_df["date"].max())
    print("[val]", val_df.shape, val_df["date"].min(), val_df["date"].max())
    print("[test]", test_df.shape, test_df["date"].min(), test_df["date"].max())

    X_train, y_train, X_val, y_val, X_test, y_test = make_model_matrices(train_df, val_df, test_df)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    daily_bike.to_csv(OUTPUT_DIR / "daily_bike.csv", index=False, encoding="utf-8-sig")
    weather_daily.to_csv(OUTPUT_DIR / "weather_daily.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUTPUT_DIR / "merged_features.csv", index=False, encoding="utf-8-sig")

    train_df.to_csv(OUTPUT_DIR / "train_df.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "val_df.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "test_df.csv", index=False, encoding="utf-8-sig")

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False, encoding="utf-8-sig")
    X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False, encoding="utf-8-sig")

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False, encoding="utf-8-sig")
    y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False, encoding="utf-8-sig")

    print(f"\n전처리 완료. 저장 위치: {OUTPUT_DIR}")
