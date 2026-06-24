import os
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="앳플리 AX 학습 챗봇",
    page_icon="🤖",
    layout="wide"
)

# API Key 가져오기
api_key = os.getenv("ANTHROPIC_API_KEY")

# API Key 확인
if not api_key:
    st.error("ANTHROPIC_API_KEY가 .env 파일에 없습니다.")
    st.stop()

# Claude 클라이언트 생성
client = Anthropic(api_key=api_key)

# Chapter 4.3 원칙을 반영한 시스템 프롬프트
system_prompt = """
# Role
너는 앳플리 AX 프로젝트를 함께 학습하고 설계하는 AI 튜터이자 실무 파트너다.

# Goal
사용자가 Claude API, Streamlit, VOC Agent, RAG, LangGraph, AI-native 회사 전환을 이해하고 직접 구현할 수 있도록 돕는다.

# Context
사용자는 "조코딩의 바이브코딩 1인창업" 책을 따라 실습하고 있다.
최종 목표는 앳플리를 AI-native 회사로 전환하는 것이다.
현재는 Claude API와 Streamlit을 활용해 VOC Agent와 AX 실습 환경을 만들고 있다.

# Rules
- 초보자도 이해할 수 있게 쉽게 설명한다.
- 가능한 한 앳플리 AX 프로젝트와 연결해서 설명한다.
- 사용자가 바로 실행할 수 있는 코드나 명령어를 우선 제공한다.
- 모르는 사실은 확정하지 않고 확인이 필요하다고 말한다.
- 고객 문의, VOC, CS, 사내 위키, RAG, Agent 설계와 연결 가능한 방향으로 답한다.
- 답변은 너무 길지 않게 핵심부터 설명한다.

# Process
1. 사용자의 질문 의도를 파악한다.
2. 현재 학습 단계와 앳플리 AX 목표를 연결한다.
3. 필요한 경우 쉬운 비유를 사용한다.
4. 실행 가능한 다음 단계를 제안한다.
"""

# 화면 제목
st.title("앳플리 AX 학습 챗봇")
st.caption("Claude API, Streamlit, VOC Agent, RAG, AX 프로젝트를 함께 학습하는 챗봇입니다.")

# session_state 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바
with st.sidebar:
    st.header("챗봇 설정")
    st.write("현재 대화 수:", len(st.session_state.messages))

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.write("예시 질문")
    st.write("- Streamlit이 뭐야?")
    st.write("- VOC Agent는 어떻게 발전시켜?")
    st.write("- RAG를 앳플리에 어떻게 적용해?")
    st.write("- 지금까지 만든 코드 구조를 설명해줘.")

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력
user_input = st.chat_input("앳플리 AX에 대해 질문해보세요")

if user_input:
    # 사용자 메시지 저장
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # 사용자 메시지 화면 출력
    with st.chat_message("user"):
        st.write(user_input)

    # Claude 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("Claude가 답변 중입니다..."):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=1200,
                    temperature=0.3,
                    system=system_prompt,
                    messages=st.session_state.messages
                )

                assistant_reply = response.content[0].text
                st.write(assistant_reply)

                # Claude 응답 저장
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_reply
                    }
                )

            except Exception as error:
                st.error(f"Claude 호출 중 오류가 발생했습니다: {error}")
