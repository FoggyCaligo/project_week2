import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# 샘플 데이터 생성
df = pd.read_csv('dataset/Advertising.csv')

# 독립변수 생성
X1 = np.array(df.iloc[:,0])
X2 = np.array(df.iloc[:,1])
X3 = np.array(df.iloc[:,2])

Y = np.array(df.iloc[:,3])

#permutation 

# 데이터프레임 생성
data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'Y': Y
})

# 학습용과 테스트용 데이터 분리
X = data[['X1', 'X2', 'X3']]
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 새로운 데이터로 예측
new_data = np.array([[200, 50, 30]])
prediction = model.predict(new_data)
print(f"\n새로운 데이터 예측값(판매량): {prediction[0]:.4f}")

# 모델 성능 평가 (R-squared)
r_squared = model.score(X_test, y_test)
print(f"R-squared: {r_squared:.4f}")





# 실제값과 예측값 비교 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5) # alpha(투명도) : 0(투명), 0.5(반투명), 1(완전 불투명)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel('실제 판매량')
plt.ylabel('예측 판매량')
plt.title('실제 판매량 vs 예측 판매량 산점도')
plt.grid(True)
plt.show()