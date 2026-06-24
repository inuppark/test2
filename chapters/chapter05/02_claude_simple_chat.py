import os
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

# .env 파일에서 환경 변수 로드 (ANTHROPIC_API_KEY 등)
load_dotenv()

# 페이지 기본 설정
st.set_page_config(
    page_title="Chapter 5 - Claude 간단 챗봇",
    page_icon="💬",
    layout="wide"
)

# .env에서 API Key 읽기
api_key = os.getenv("ANTHROPIC_API_KEY")

# 화면 제목 및 설명
st.title("Chapter 5 - Claude 간단 챗봇")
st.write("Streamlit 화면에서 Claude API를 호출하는 기본 실습입니다.")

# API Key가 없으면 오류 메시지를 표시하고 실행 중단
if not api_key:
    st.error("ANTHROPIC_API_KEY가 .env 파일에 없습니다. .env 파일을 확인해주세요.")
    st.stop()

# Anthropic 클라이언트 생성 (API Key를 사용해 Claude와 통신 준비)
client = Anthropic(api_key=api_key)

# Chapter 4.3 원칙을 반영한 시스템 프롬프트
# Role(역할), Goal(목표), Rules(규칙)을 명확히 작성
system_prompt = """
# Role
너는 앳플리 AX 실습을 돕는 친절한 AI 튜터다.

# Goal
사용자의 질문에 쉽고 실무적으로 답변한다.

# Rules
- 초보자도 이해할 수 있게 설명한다.
- 불필요하게 길게 답하지 않는다.
- 앳플리 AX, VOC Agent, Streamlit, Claude API 학습 맥락을 고려한다.
- 모르는 내용은 확정하지 말고 확인이 필요하다고 말한다.
"""

# 여러 줄 입력창 - 기본값으로 예시 질문을 넣어 둠
user_question = st.text_area(
    "Claude에게 질문하기",
    value="Streamlit이 무엇인지 초보자도 이해할 수 있게 설명해줘.",
    height=140
)

# 버튼 클릭 시 Claude API 호출
if st.button("Claude에게 질문하기", type="primary"):
    if not user_question.strip():
        # 입력값이 비어 있으면 경고 메시지 표시
        st.warning("질문을 먼저 입력해주세요.")
    else:
        # API 호출 중 로딩 스피너 표시
        with st.spinner("Claude가 답변을 작성하고 있습니다..."):
            try:
                # Claude API 호출
                response = client.messages.create(
                    model="claude-sonnet-4-5",   # 사용할 Claude 모델
                    max_tokens=800,              # 최대 출력 토큰 수
                    temperature=0.3,             # 낮을수록 일관된 답변 (0~1)
                    system=system_prompt,        # 역할/규칙 프롬프트
                    messages=[
                        {
                            "role": "user",
                            "content": user_question  # 사용자 질문
                        }
                    ]
                )

                # 응답 텍스트 추출 후 화면에 표시
                st.subheader("Claude 답변")
                st.write(response.content[0].text)

            except Exception as error:
                # 오류 발생 시 오류 내용을 화면에 표시
                st.error(f"Claude 호출 중 오류가 발생했습니다: {error}")
