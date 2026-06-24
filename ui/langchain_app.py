import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.langchain_agent import ask_langchain_agent, save_langchain_log

st.title("LangChain Agent")
st.write("LangChain으로 만든 Tool 사용형 운영 Agent입니다.")

st.info("V3: LangChain Agent가 RAG Tool을 호출해 reports와 data/wiki를 검색할 수 있습니다.")

sample_questions = [
    "현재 프로젝트 상태 알려줘",
    "최근 생성된 보고서 목록 보여줘",
    "제품 하자 문의가 들어오면 어떤 절차로 처리해야 해?",
    "최근 보고서와 사내 기준을 함께 봐서 가장 우선 처리할 업무 알려줘",
    "최근 VOC 기준으로 가장 중요한 이슈와 처리 기준을 같이 알려줘",
    "RAG Agent는 어떻게 써?",
]

question = st.text_area(
    "질문 입력",
    value=sample_questions[2],
    height=120
)

st.write("예시 질문:")
for q in sample_questions:
    st.code(q)

if st.button("LangChain Agent 실행"):
    try:
        with st.spinner("LangChain Agent가 필요한 Tool을 선택하고 답변 중입니다..."):
            answer = ask_langchain_agent(question)
            log_path = save_langchain_log(question, answer)

        st.subheader("답변")
        st.markdown(answer)

        st.success(f"실행 로그 저장 완료: {log_path}")

        if os.path.exists("logs/langchain_agent_log.csv"):
            st.subheader("최근 LangChain Agent 로그")
            log_df = pd.read_csv("logs/langchain_agent_log.csv")
            st.dataframe(log_df.tail(10), use_container_width=True)

    except Exception as e:
        st.error("LangChain Agent 실행 중 오류가 발생했습니다.")
        st.write(str(e))
else:
    st.write("질문을 입력한 뒤 실행 버튼을 눌러주세요.")
