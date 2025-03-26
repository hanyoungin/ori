import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(layout="wide")
st.title("오리엔트정공 감성 분석 기반 주가 예측 대시보드")

# 감성 키워드 설정
positive_keywords = ['무죄', '급등', '반등', '수혜', '가시화', '강세']
negative_keywords = ['탄핵', '주의보', '불안정']

# 감성 분석 함수 정의
def classify_sentiment(text):
    pos = sum(word in text for word in positive_keywords)
    neg = sum(word in text for word in negative_keywords)
    return 1 if pos > neg else -1 if neg > pos else 0

# 뉴스 입력 섹션
st.subheader("1. 뉴스 입력")
news_input = st.text_area("오리엔트정공 관련 뉴스 제목 또는 내용 입력", "이재명 무죄 확정, 테마주 반등 기대")

# 감성 점수 계산 및 표시
score = classify_sentiment(news_input)
st.write(f"→ 감성 점수: {score}")

# 학습 데이터 생성
X_train = pd.DataFrame({'sentiment_score': [-2, -1, 0, 1, 2, 3]})
y_train = pd.Series([7249, 7363, 7477, 7592, 7706, 7821])

# 선형 회귀 모델 학습 및 예측
model = LinearRegression().fit(X_train, y_train)
predicted_price = model.predict([[score]])[0]

# 예측 결과 표시
st.subheader("2. 예측 결과")
st.metric(label="예측 종가", value=f"{predicted_price:,.0f} 원")

# 시각화
st.subheader("3. 감성 점수별 주가 예측 시각화")
fig, ax = plt.subplots()
ax.plot(X_train['sentiment_score'], y_train, marker='o', label='학습 데이터')
ax.scatter(score, predicted_price, color='red', label='예측 결과')
ax.set_xlabel('감성 점수')
ax.set_ylabel('예측 주가')
ax.grid(True)
ax.legend()
st.pyplot(fig)

