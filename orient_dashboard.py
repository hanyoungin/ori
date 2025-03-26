import subprocess
import sys

# 필요한 패키지 목록
required_packages = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'matplotlib']

# 패키지 설치 함수
def install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# 패키지 설치 실행
install_packages(required_packages)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Streamlit 애플리케이션 코드 시작
st.set_page_config(layout="wide")
st.title("오리엔트정공 감성 분석 기반 주가 예측 대시보드")

# 감성 키워드 기반 점수화 함수
positive_keywords = ['무죄', '급등', '반등', '수혜', '가시화', '강세']
negative_keywords = ['탄핵', '주의보', '불안정']

def classify_sentiment(text):
    pos = sum(1 for word in positive_keywords if word in text)
    neg = sum(1 for word in negative_keywords if word in text)
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    else:
        return 0

# 입력 뉴스
st.subheader("1. 뉴스 입력")
news_input = st.text_area("오리엔트정공 관련 뉴스 제목 또는 내용 입력", "이재명 무죄 확정, 테마주 반등 기대")

# 감성 점수 계산
score = classify_sentiment(news_input)
st.write(f"→ 감성 점수: {score}")

# 학습용 데이터셋 (샘플)
X_train = pd.DataFrame({'sentiment_score': [-2, -1, 0, 1, 2, 3]})
y_train = pd.Series([7249, 7363, 7477, 7592, 7706, 7821])
model = LinearRegression().fit(X_train, y_train)

# 예측
predicted_price = model.predict([[score]])[0]
st.subheader("2. 예측 결과")
st.metric(label="예측 종가", value=f"{predicted_price:,.0f} 원")

# 시각화
st.subheader("3. 감성 점수별 주가 예측 시각화")
fig, ax = plt.subplots()
ax.plot(X_train['sentiment_score'], y_train, marker='o', label='예측 선')
ax.scatter(score, predicted_price, color='red', label='현재 입력')
ax.set_xlabel('감성 점수')
ax.set_ylabel('예측 주가')
ax.grid(True)
ax.legend()
st.pyplot(fig)
