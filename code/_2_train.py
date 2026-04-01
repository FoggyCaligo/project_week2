# 사전설치 : pip install xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# 1. preprocess_1.py를 돌려서 전처리한 데이터 불러오기
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "reformed_data"
OUTPUT_DIR = BASE_DIR / "dataset" / "model_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_val.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_val = pd.read_csv(DATA_DIR / "y_val.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

train_df = pd.read_csv(DATA_DIR / "train_df.csv")
val_df = pd.read_csv(DATA_DIR / "val_df.csv")
test_df = pd.read_csv(DATA_DIR / "test_df.csv")

train_df["date"] = pd.to_datetime(train_df["date"])
val_df["date"] = pd.to_datetime(val_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


# 5. 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 하이퍼파라미터 튜닝: GridSearchCV
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)   # 제곱오차(reg:squarederror)로 MSE를 뜻함

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1],   # 각 트리 학습에 사용되는 데이터 샘플 비율, 과적합 방지
    'colsample_bytree': [0.8, 1],   # 각 트리에서 사용되는 피처(변수)의 비율, 과적합 방지
    'alpha': [0, 1],   # L1 정규화 (0: L1정규화 적용안함, 0.1~1: 약한 정규화, 1~10: 중간정규화, 10~100: 강한 정규화, 100이상: 매우강한 정규화)
    'lambda': [1, 10]  # L2 정규화 (0: L2정규화 적용안함, 0.1~1: 약한 정규화, 1~10: 중간정규화, 10~100: 강한 정규화, 100이상: 매우강한 정규화)
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error', # GridSearchCV는 점수가 높을수록 좋은 모델로 간주함으로 RMSE에 negative(-)가 붙어서 덜 음수인 값이 더 나은 모델로 평가되도록 함
    cv=3,
    verbose=1,
    n_jobs=-1  # (n_jobs=1: 병렬처리 없이 단일 CPU코어로 순차적 학습, n_jobs=n: n개의 CPU코어로 사용, n_jobs=-1: 모든 CPU코어 사용)
)

grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)

# 7. 최적 모델로 예측
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# 역변환 (log1p → 원래 값)
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

# 8. 평가
rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
r2 = r2_score(y_test_exp, y_pred_exp)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 9. 피처 중요도 시각화
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]  # argsort: 오름차순으로 정렬했을 때의 인덱스를 반환, [::-1}: 중요도 내림차순 정렬
sorted_features = X.columns[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(10, 6))
sns.barplot(
    x=sorted_importances,
    y=sorted_features
)
plt.title("Feature Importances (XGBoost) - Sorted")
plt.show()

# 10. 예측 vs 실제 시각화
plt.figure(figsize=(8, 5))
plt.plot(y_test_exp.values[:100], label="Actual")  # 테스트 데이터의 첫 100개 샘플만 시각화
plt.plot(y_pred_exp[:100], label="Predicted") # 예측 데이터의 첫 100개 샘플만 시각화
plt.legend()
plt.title("Prediction vs Actual (first 100)")
plt.xlabel("Sample")
plt.ylabel("Bike Count")
plt.show()
