import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("VOC 분석 AI Agent")
st.write("고객 문의를 입력하면 VOC 분석 보고서를 생성합니다.")

voc_text = st.text_area(
    "고객 문의 입력",
    height=250,
    value="""배송이 늦어요.
제품이 고장났어요.
환불 요청합니다.
배송이 너무 느립니다.
설치 방법이 어렵습니다."""
)

if st.button("VOC 분석하기"):
    prompt = f"""
당신은 제조/판매 회사의 VOC 분석 전문가입니다.

아래 고객 문의 데이터를 분석하세요.

[고객 문의 데이터]
{voc_text}

아래 형식으로 보고서를 작성하세요.

# VOC 분석 보고서

## 1. 카테고리별 분류
## 2. 비율 분석
## 3. 가장 시급한 문제
## 4. 원인 추정
## 5. 개선 제안
## 6. 경영진 보고용 한 줄 요약
"""

    with st.spinner("분석 중입니다..."):
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

    st.markdown(response.output_text)
