import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "reformed_data"
OUTPUT_DIR = BASE_DIR / "dataset" / "model_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
pd.options.display.float_format = "{:.2f}".format

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

print("데이터 로드 완료")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print("모델 학습 완료")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print("예측 완료")

r2_train = model.score(X_train, y_train)
r2_val = model.score(X_val, y_val)
r2_test = model.score(X_test, y_test)

print("\n===== R-squared 결과 =====")
print(f"Train R-squared: {r2_train:.4f}")
print(f"Validation R-squared: {r2_val:.4f}")
print(f"Test R-squared: {r2_test:.4f}")

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

print("\n===== 회귀계수 상위 20개 =====")
print(coef_df[["feature", "coefficient"]].head(20))
print(f"\n절편(intercept): {model.intercept_:.4f}")

coef_df.to_csv(OUTPUT_DIR / "linear_regression_coefficients.csv", index=False, encoding="utf-8-sig")
print("회귀계수 저장 완료")

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

train_result.to_csv(OUTPUT_DIR / "train_prediction_result.csv", index=False, encoding="utf-8-sig")
val_result.to_csv(OUTPUT_DIR / "val_prediction_result.csv", index=False, encoding="utf-8-sig")
test_result.to_csv(OUTPUT_DIR / "test_prediction_result.csv", index=False, encoding="utf-8-sig")
print("예측 결과 CSV 저장 완료")

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
plt.savefig(OUTPUT_DIR / "val_scatter.png", dpi=150)
plt.show()
print("Validation 산점도 저장 완료")

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
plt.savefig(OUTPUT_DIR / "test_scatter.png", dpi=150)
plt.show()
print("Test 산점도 저장 완료")

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
plt.savefig(OUTPUT_DIR / "val_timeseries.png", dpi=150)
plt.show()
print("Validation 시계열 그래프 저장 완료")

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
plt.savefig(OUTPUT_DIR / "test_timeseries.png", dpi=150)
plt.show()
print("Test 시계열 그래프 저장 완료")

print(f"\n전체 완료. 결과 저장 위치: {OUTPUT_DIR}")
