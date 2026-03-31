import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# =========================
# 1. 전처리 결과 불러오기
# =========================
X_train = pd.read_csv('output/X_train.csv')
X_val = pd.read_csv('output/X_val.csv')
X_test = pd.read_csv('output/X_test.csv')

y_train = pd.read_csv('output/y_train.csv').iloc[:, 0]
y_val = pd.read_csv('output/y_val.csv').iloc[:, 0]
y_test = pd.read_csv('output/y_test.csv').iloc[:, 0]

# test 시계열 그래프용 날짜
test_df = pd.read_csv('output/test_df.csv')
test_df['date'] = pd.to_datetime(test_df['date'])

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# =========================
# 2. 모델 학습
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 3. 예측
# =========================
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# =========================
# 4. 모델 성능 평가 (R-squared)
# =========================
r2_train = model.score(X_train, y_train)
r2_val = model.score(X_val, y_val)
r2_test = model.score(X_test, y_test)

print(f"\nTrain R-squared: {r2_train:.4f}")
print(f"Validation R-squared: {r2_val:.4f}")
print(f"Test R-squared: {r2_test:.4f}")

# =========================
# 5. 회귀식 확인
# =========================
coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_
})

coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

print("\n회귀계수(절댓값 기준 상위 15개):")
print(coef_df[['feature', 'coefficient']].head(15))

print(f"\n절편(intercept): {model.intercept_:.4f}")

# =========================
# 6. 실제값 vs 예측값 산점도 (Test)
# =========================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel('실제 대여건수')
plt.ylabel('예측 대여건수')
plt.title('실제 대여건수 vs 예측 대여건수 산점도 (Test)')
plt.grid(True)
plt.show()

# =========================
# 7. 날짜별 실제값 vs 예측값 선 그래프 (Test)
# =========================
plt.figure(figsize=(12, 6))
plt.plot(test_df['date'], y_test.values, label='실제 대여건수')
plt.plot(test_df['date'], y_test_pred, label='예측 대여건수')
plt.xlabel('날짜')
plt.ylabel('대여건수')
plt.title('날짜별 실제 대여건수 vs 예측 대여건수 (Test)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 8. 예측 결과 저장
# =========================
result_df = pd.DataFrame({
    'date': test_df['date'],
    'actual': y_test.values,
    'predicted': y_test_pred
})

result_df.to_csv('output/linear_regression_test_result.csv', index=False, encoding='utf-8-sig')
print("\n예측 결과 저장 완료: output/linear_regression_test_result.csv")