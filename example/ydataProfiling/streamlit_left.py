import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로드
@st.cache_data # 함수의 입력 매개변수와 반환 값을 캐시에 저장하여 호출 시 재사용
def load_data():
    df = pd.read_csv('dataset/HR_comma_sep.csv')
    df.rename(columns={'Departments ': 'Departments'}, inplace=True)
    df = pd.get_dummies(df, columns=['Departments', 'salary'], drop_first=True)
    return df

# 데이터 준비
df = load_data()

# 주요 Feature 선택
selected_features = ['satisfaction_level', 'number_project', 'time_spend_company']
X = df[selected_features]
y = df['left']

# 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Streamlit UI 구성
st.title("퇴사 여부 예측 시스템")
st.write("직원의 정보를 입력하여 퇴사 가능성을 예측합니다.")

# 사용자 입력
satisfaction_level = st.slider("만족도 (satisfaction_level)", 0.0, 1.0, 0.5, step=0.01)
number_project = st.number_input("프로젝트 수 (number_project)", min_value=1, max_value=10, value=3, step=1)
time_spend_company = st.number_input("근무 연수 (time_spend_company)", min_value=1, max_value=20, value=3, step=1)

# 입력 데이터를 모델에 맞게 변환
user_data = np.array([[satisfaction_level, number_project, time_spend_company]])
user_data_scaled = scaler.transform(user_data)

# 예측 버튼
if st.button("퇴사 여부 예측"):
    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0]
    
    if prediction == 1:
        st.error(f"퇴사 가능성이 높습니다! (확률: {prediction_proba[1] * 100:.2f}%)")
    else:
        st.success(f"퇴사 가능성이 낮습니다! (확률: {prediction_proba[0] * 100:.2f}%)")
