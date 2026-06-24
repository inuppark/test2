import os
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="Chapter 5 - 기억하는 Claude 챗봇",
    page_icon="🧠",
    layout="wide"
)

# API Key 가져오기
api_key = os.getenv("ANTHROPIC_API_KEY")

# 화면 제목
st.title("Chapter 5 - 기억하는 Claude 챗봇")
st.caption("st.session_state를 사용해서 대화 내역을 기억하는 챗봇입니다.")

# API Key 확인
if not api_key:
    st.error("ANTHROPIC_API_KEY가 .env 파일에 없습니다.")
    st.stop()

# Claude 클라이언트 생성
client = Anthropic(api_key=api_key)

# Chapter 4.3 원칙을 반영한 시스템 프롬프트
system_prompt = """
# Role
너는 앳플리 AX 실습을 돕는 친절한 AI 튜터다.

# Goal
사용자와 자연스럽게 대화하면서 Claude API, Streamlit, VOC Agent, 앳플리 AX 학습을 돕는다.

# Rules
- 초보자도 이해할 수 있게 쉽게 설명한다.
- 이전 대화 맥락을 반영해서 답한다.
- 불필요하게 길게 답하지 않는다.
- 모르는 내용은 확정하지 말고 확인이 필요하다고 말한다.
"""

# session_state에 messages가 없으면 빈 리스트로 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바
with st.sidebar:
    st.header("설정")
    st.write("현재 대화 수:", len(st.session_state.messages))

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 이전 대화 내역 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 받기
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    # 사용자 메시지를 화면과 메모리에 저장
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Claude 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("Claude가 답변 중입니다..."):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=1000,
                    temperature=0.3,
                    system=system_prompt,
                    messages=st.session_state.messages
                )

                assistant_reply = response.content[0].text
                st.write(assistant_reply)

                # Claude 응답도 메모리에 저장
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_reply
                    }
                )

            except Exception as error:
                st.error(f"Claude 호출 중 오류가 발생했습니다: {error}")
