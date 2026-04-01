import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

#경로 설정
SCRIPT_DIR = Path('./신재용_머신러닝프로젝트.ipynb').resolve().parent

def find_base_dir() -> Path:
  candidates = [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]
  for candidate in candidates:
    if (candidate / "dataset").exists():
      return candidate
  return SCRIPT_DIR

BASE_DIR = find_base_dir()
REFORMED_DIR = BASE_DIR / "dataset" / "reformed_data"
MODEL_OUTPUT_DIR = BASE_DIR / "dataset/model_output"

MODEL_JOBLIB_PATH = MODEL_OUTPUT_DIR / "best_model.joblib"
MODEL_METADATA_PKL_PATH = MODEL_OUTPUT_DIR / "model_metadata.pkl"

#모델 로드
if not MODEL_JOBLIB_PATH.exists() or not MODEL_METADATA_PKL_PATH.exists():
 raise FileNotFoundError(
  "저장된 모델 아티팩트가 없습니다. 먼저 `신재용_머신러닝프로젝트.ipynb`를 실행해 주세요."
 )
model = joblib.load(MODEL_JOBLIB_PATH)
with open(MODEL_METADATA_PKL_PATH, "rb") as f:
 metadata = pickle.load(f)

#서포트용 csv 파일을 df로 로드
comparison_path = MODEL_OUTPUT_DIR / "model_comparison.csv"
importance_path = MODEL_OUTPUT_DIR / "best_model_feature_importance.csv"
metrics_path = MODEL_OUTPUT_DIR / "best_model_metrics.csv"

comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
importance_df = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()





#UI
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
today = pd.Timestamp.today().date()

st.set_page_config(page_title = "따릉이 수요 예측 UI", layout="wide")
st.title("따릉이 대여건수 예측 UI")

st.sidebar.header('사용자 입력 파라미터')
target_date = st.sidebar.date_input("예측 날짜", value=today)
avg_temp_c = st.sidebar.number_input("평균기온(°C)", value=15.0, step=0.1)
min_temp_c = st.sidebar.number_input("최저기온(°C)", value=10.0, step=0.1)
max_temp_c = st.sidebar.number_input("최고기온(°C)", value=20.0, step=0.1)
precip_duration_hr = st.sidebar.number_input("강수 계속시간(hr)", min_value=0.0, value=0.0, step=0.1)

daily_precip_mm = st.sidebar.number_input("일강수량(mm)", min_value=0.0, value=0.0, step=0.1)
avg_wind_speed_m_s = st.sidebar.number_input("평균 풍속(m/s)", min_value=0.0, value=2.5, step=0.1)
avg_rel_humidity_pct = st.sidebar.number_input("평균 상대습도(%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
submitted = st.sidebar.button("따릉이 대여건수 예측하기")

if submitted:
 try:
  if min_temp_c > max_temp_c:
   st.error("최저기온은 최고기온보다 클 수 없습니다.")
   st.stop()

  with st.spinner("joblib/pkl 아티팩트를 불러오는 중입니다..."):
   
   ts = pd.to_datetime(target_date)
   temp_range_c = float(max_temp_c - min_temp_c)
   is_rainy = int(daily_precip_mm > 0)
   dayofweek = int(ts.dayofweek)
   month = int(ts.month)
   dayofyear = int(ts.dayofyear)
   row = {
    "trend_idx": int((ts - pd.to_datetime(today)).days),
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
   input_df = pd.DataFrame([{col: row.get(col, 0) for col in metadata["feature_columns"]}], columns=metadata["feature_columns"])
   
   predicted_value = float(model.predict(input_df)[0])
   if "log" in metadata["model_name"]:
    predicted_value = float(np.expm1(predicted_value))
   predicted_value = max(0.0, predicted_value)

  val_r2 = metadata["metrics"]["validation"]["r2"]
  test_r2 = metadata["metrics"]["test"]["r2"]

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("예측 대여건수", f"{predicted_value:,.0f}")
  m2.metric("선택 모델", metadata["model_name"])
  m3.metric("Validation R²", f"{val_r2:.4f}")
  m4.metric("Test R²", f"{test_r2:.4f}")
  st.success("예측이 완료되었습니다.")
  

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

