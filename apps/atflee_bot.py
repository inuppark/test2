import os
import sys
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

# Streamlit Cloud는 apps/ 하위를 실행 디렉토리로 삼아 프로젝트 루트를 못 찾는 경우가 있다.
# 이 파일 위치(apps/)의 한 단계 위가 프로젝트 루트이므로 sys.path에 명시적으로 추가한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# utils.rag_utils에서 RAG 공통 함수를 가져온다.
from utils.rag_utils import (
    search_wiki,
    build_rag_context,
    get_source_file_names,
    format_search_results_for_display,
    evaluate_rag_answer,
    TOP_K,
)

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="앳플리 봇",
    page_icon="🧭",
    layout="wide"
)

def get_api_key():
    """
    로컬에서는 .env의 ANTHROPIC_API_KEY를 사용하고,
    Streamlit Cloud에서는 st.secrets의 ANTHROPIC_API_KEY를 사용한다.
    """
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    return os.getenv("ANTHROPIC_API_KEY")


api_key = get_api_key()

if not api_key:
    st.error("ANTHROPIC_API_KEY가 설정되어 있지 않습니다. 로컬에서는 .env, 배포 환경에서는 Streamlit Secrets에 등록해주세요.")
    st.stop()

client = Anthropic(api_key=api_key)
model_name = "claude-sonnet-4-5"


# ==============================
# System Prompt (RAG 버전)
# ==============================

system_prompt = """
# Role
너는 앳플리 위키 기반 RAG 답변 봇이다.

# Goal
사용자 질문에 대해 검색된 앳플리 위키 문서를 근거로 정확하고 친절하게 답변한다.

# Context
너에게 제공되는 <rag_context>는 data/wiki 문서 중 사용자 질문과 관련도가 높은 문서만 검색해서 가져온 것이다.
너는 이 문서 내용을 우선 근거로 사용한다.

# Rules
- <rag_context>에 있는 정보만 확정적으로 말한다.
- <rag_context>에 없는 내용은 추측하지 않는다.
- <rag_context> 안에 [관련 스니펫]이 포함되어 있으면 해당 부분을 우선 참고한다.
- 단, 스니펫만 보고 단정하지 말고 [문서 전체] 내용과 함께 판단한다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
- 정책, 보증, 교환/환불, AS 조건은 확실하지 않으면 "정확한 확인이 필요합니다"라고 말한다.
- 개인정보, 주문번호, 연락처 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
- 고객이 바로 해볼 수 있는 다음 행동을 안내한다.
- 답변은 초보자도 이해할 수 있게 쉽게 작성한다.
- 제품 사용법은 가능한 경우 사용자 매뉴얼과 문제 해결 FAQ 확인을 함께 안내한다.

# Process
1. 사용자 질문의 의도를 파악한다.
2. <rag_context>에서 관련 근거를 찾는다.
3. 확실한 정보와 확인이 필요한 정보를 구분한다.
4. 사용자가 바로 할 수 있는 행동을 안내한다.
5. 마지막에 참고한 문서명을 표시한다.

# Output Format
아래 형식으로 답변한다.

1. 간단한 답변
2. 근거가 되는 앳플리 위키 정보
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 문서
"""


def build_messages(user_question, rag_context, source_files):
    """
    Claude에게 보낼 messages를 구성한다.
    새 질문에는 RAG로 검색된 문서만 rag_context로 전달한다.
    이전 대화 내역도 함께 전달해 맥락을 유지한다.
    """
    prompt = f"""
<rag_context>
{rag_context}
</rag_context>

<source_files>
{", ".join(source_files)}
</source_files>

<user_question>
{user_question}
</user_question>
"""

    messages = []

    for message in st.session_state.docent_messages:
        messages.append(message)

    messages.append({"role": "user", "content": prompt})

    return messages


def ask_docent(user_question, rag_context, source_files):
    """
    RAG로 검색된 문서를 Context로 Claude에게 전달하고 답변을 받는다.
    """
    response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        messages=build_messages(user_question, rag_context, source_files)
    )

    return response.content[0].text


# =========================
# Streamlit UI
# =========================

st.title("앳플리 봇")
st.caption("질문과 관련 있는 data/wiki 문서를 검색해 답변합니다.")

with st.sidebar:
    st.header("앳플리 봇")
    st.write("제품/앱/정책 관련 질문에 대해 RAG 기반으로 답변합니다.")
    st.info("현재 앳플리 봇은 data/wiki 문서와 공식몰 공개 정보를 바탕으로 답변합니다.")

    st.markdown("### 예시 질문")
    st.write("- 체중계가 앱이랑 연결이 안 돼요.")
    st.write("- 배송은 얼마나 걸리나요?")
    st.write("- 환불은 어떻게 처리되나요?")
    st.write("- 앱 문의는 어디로 하면 되나요?")
    st.write("- 가족도 같이 사용할 수 있나요?")

    st.markdown("---")

    if st.button("대화 초기화"):
        st.session_state.docent_messages = []
        st.rerun()

# 대화 저장소 초기화
if "docent_messages" not in st.session_state:
    st.session_state.docent_messages = []

# 이전 대화 출력
for message in st.session_state.docent_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력
user_question = st.chat_input("앳플리 제품/앱/정책에 대해 질문해보세요")

if user_question:
    st.session_state.docent_messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("앳플리 봇이 답변 중입니다..."):
            try:
                # 1단계: 질문과 관련 있는 문서를 검색한다.
                search_results = search_wiki(user_question, top_k=TOP_K)

                if not search_results:
                    st.warning("data/wiki 폴더에 문서가 없어 답변을 생성할 수 없습니다.")
                else:
                    # 2단계: 검색된 문서만 RAG Context로 합친다.
                    rag_context = build_rag_context(search_results)
                    source_files = get_source_file_names(search_results)

                    # 검색된 참고 문서와 관련 스니펫을 expander로 표시한다.
                    with st.expander("검색된 참고 문서"):
                        for result in search_results:
                            st.markdown(f"**{result['file_name']}** / 점수: {result['score']}")
                            st.caption(result.get("snippet", ""))

                    # 3단계: 검색된 문서를 Claude에게 전달해 답변을 받는다.
                    answer = ask_docent(user_question, rag_context, source_files)
                    st.write(answer)

                    # 4단계: 답변 품질을 체크한다.
                    evaluation = evaluate_rag_answer(answer, source_files)
                    with st.expander("답변 품질 체크"):
                        st.write(f"점수: {evaluation['score']}점 / 상태: {evaluation['status']}")
                        for check in evaluation["checks"]:
                            if check["passed"]:
                                st.success(f"✔ {check['name']}: {check['message']}")
                            else:
                                st.warning(f"△ {check['name']}: {check['message']}")

                    st.session_state.docent_messages.append(
                        {"role": "assistant", "content": answer}
                    )

            except Exception as error:
                st.error(f"앳플리 봇 답변 생성 중 오류가 발생했습니다: {error}")
