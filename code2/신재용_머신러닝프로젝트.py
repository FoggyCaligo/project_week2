'''
#필요 라이브러리 설치
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl streamlit joblib notebook setuptools ydata-profiling

실행 순서

1. `python preprocess.py`
2. 신재용_머신러닝프로젝트.ipynb에서 **Run All**
3. `streamlit run 신재용_머신러닝프로젝트.py`
'''



from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import (
    BASE_DIR,
    MODEL_OUTPUT_DIR,
    RAW_BIKE_DIR,
    RAW_WEATHER_DIR,
    REFORMED_DIR,
    REPORT_DIR,
    TARGET_COL,
    build_prediction_input_row,
    ensure_preprocessed_files,
    load_default_prediction_values,
    run_preprocessing,
)

MODEL_JOBLIB_PATH = BASE_DIR / "best_model.joblib"
MODEL_METADATA_PKL_PATH = BASE_DIR / "model_metadata.pkl"


def load_model_artifacts():
    if not MODEL_JOBLIB_PATH.exists() or not MODEL_METADATA_PKL_PATH.exists():
        raise FileNotFoundError(
            "저장된 모델 아티팩트가 없습니다. 먼저 `신재용_머신러닝프로젝트.ipynb`를 실행해 주세요."
        )

    model = joblib.load(MODEL_JOBLIB_PATH)
    with open(MODEL_METADATA_PKL_PATH, "rb") as f:
        metadata = pickle.load(f)
    return model, metadata


def load_support_tables():
    comparison_path = MODEL_OUTPUT_DIR / "model_comparison.csv"
    importance_path = MODEL_OUTPUT_DIR / "best_model_feature_importance.csv"
    metrics_path = MODEL_OUTPUT_DIR / "best_model_metrics.csv"

    comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    importance_df = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    return comparison_df, importance_df, metrics_df


def run_heatmap(show_plot: bool = False):
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

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
    return heatmap_path


def run_profiling_report():
    try:
        from ydata_profiling import ProfileReport
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ydata-profiling 또는 setuptools가 설치되지 않았습니다. "
            "`python -m pip install setuptools ydata-profiling` 후 다시 실행하세요."
        ) from e

    ensure_preprocessed_files()
    df = pd.read_csv(REFORMED_DIR / "merged_features.csv")
    profile = ProfileReport(
        df,
        title="Ddarungi Demand Prediction Profiling Report (Improved Date + Weather Model)",
        explorative=True,
    )
    output_file = REPORT_DIR / "ddarungi_profiling_report.html"
    profile.to_file(output_file)
    return output_file


def run_app():
    st.set_page_config(page_title="따릉이 수요 예측 UI", layout="wide")
    st.title("따릉이 대여건수 예측 UI")
    st.caption("저장된 .joblib 모델과 .pkl 메타데이터를 불러와 예측합니다.")
    st.info(
        "실행 순서\n"
        "1. python preprocess.py\n"
        "2. 신재용_머신러닝프로젝트.ipynb 실행\n"
        "3. streamlit run 신재용_머신러닝프로젝트.py"
    )

    with st.expander("경로 및 아티팩트 상태", expanded=False):
        st.write(f"BASE_DIR: {BASE_DIR}")
        st.write(f"따릉이 원본 폴더: {RAW_BIKE_DIR}")
        st.write(f"날씨 원본 폴더: {RAW_WEATHER_DIR}")
        st.write(f"전처리 결과 폴더: {REFORMED_DIR}")
        st.write(f"모델 출력 폴더: {MODEL_OUTPUT_DIR}")
        st.write(f"리포트 폴더: {REPORT_DIR}")
        st.write(f"joblib 모델 존재: {MODEL_JOBLIB_PATH.exists()}")
        st.write(f"pkl 메타데이터 존재: {MODEL_METADATA_PKL_PATH.exists()}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("1) 전처리 실행"):
            with st.spinner("전처리 중입니다..."):
                summary = run_preprocessing()
            st.success("전처리가 완료되었습니다.")
            st.json(summary)

    with c2:
        if st.button("2) 히트맵 + HTML 리포트 생성"):
            with st.spinner("리포트 생성 중입니다..."):
                heatmap_path = run_heatmap(show_plot=False)
                profiling_path = run_profiling_report()
            st.success("리포트 생성이 완료되었습니다.")
            if Path(heatmap_path).exists():
                st.image(str(heatmap_path), caption="상관관계 히트맵", use_container_width=True)
            st.write(f"프로파일링 HTML 저장 위치: {profiling_path}")

    st.warning("모델 학습은 이 앱 안에서 하지 않습니다. `신재용_머신러닝프로젝트.ipynb`를 실행해 주세요.")

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
        with right:
            daily_precip_mm = st.number_input("일강수량(mm)", min_value=0.0, value=float(defaults["daily_precip_mm"]), step=0.1)
            avg_wind_speed_m_s = st.number_input("평균 풍속(m/s)", min_value=0.0, value=float(defaults["avg_wind_speed_m_s"]), step=0.1)
            avg_rel_humidity_pct = st.number_input("평균 상대습도(%)", min_value=0.0, max_value=100.0, value=float(defaults["avg_rel_humidity_pct"]), step=0.1)
        submitted = st.form_submit_button("저장된 모델로 예측하기")

    if submitted:
        try:
            if min_temp_c > max_temp_c:
                st.error("최저기온은 최고기온보다 클 수 없습니다.")
                st.stop()

            with st.spinner("joblib/pkl 아티팩트를 불러오는 중입니다..."):
                model, metadata = load_model_artifacts()
                input_df = build_prediction_input_row(
                    feature_columns=metadata["feature_columns"],
                    date_origin=metadata["date_origin"],
                    target_date=target_date,
                    avg_temp_c=avg_temp_c,
                    min_temp_c=min_temp_c,
                    max_temp_c=max_temp_c,
                    precip_duration_hr=precip_duration_hr,
                    daily_precip_mm=daily_precip_mm,
                    avg_wind_speed_m_s=avg_wind_speed_m_s,
                    avg_rel_humidity_pct=avg_rel_humidity_pct,
                )
                predicted_value = float(model.predict(input_df)[0])
                if "log" in metadata["model_name"]:
                    predicted_value = float(np.expm1(predicted_value))
                predicted_value = max(0.0, predicted_value)
                comparison_df, importance_df, metrics_df = load_support_tables()

            val_r2 = metadata["metrics"]["validation"]["r2"]
            test_r2 = metadata["metrics"]["test"]["r2"]

            st.success("예측이 완료되었습니다.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("예측 대여건수", f"{predicted_value:,.0f}")
            m2.metric("선택 모델", metadata["model_name"])
            m3.metric("Validation R²", f"{val_r2:.4f}")
            m4.metric("Test R²", f"{test_r2:.4f}")

            with st.expander("모델 입력값(파생변수 포함)", expanded=True):
                preview = input_df.copy()
                preview.insert(0, "target_date", str(pd.to_datetime(target_date).date()))
                st.dataframe(preview, use_container_width=True)

            with st.expander("모델 비교표", expanded=False):
                st.dataframe(comparison_df, use_container_width=True)

            with st.expander("상위 중요 변수", expanded=False):
                st.dataframe(importance_df.head(20), use_container_width=True)

            with st.expander("평가 지표", expanded=False):
                st.dataframe(metrics_df, use_container_width=True)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    run_app()
