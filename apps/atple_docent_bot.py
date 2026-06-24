import os
import re
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

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
# RAG 설정값
# ==============================

WIKI_DIR = "data/wiki"
TOP_K = 3


# ==============================
# RAG 함수들
# ==============================

def load_wiki_documents():
    """
    data/wiki 폴더의 .md 파일을 모두 읽어
    [{"file_name": 파일명, "content": 내용}, ...] 형태로 반환한다.
    """
    documents = []

    if not os.path.exists(WIKI_DIR):
        return documents

    for file_name in os.listdir(WIKI_DIR):
        if file_name.endswith(".md"):
            file_path = os.path.join(WIKI_DIR, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(
                    {
                        "file_name": file_name,
                        "content": file.read()
                    }
                )

    return documents


def tokenize(text):
    """
    텍스트를 소문자로 바꾸고 단어 단위로 나눈다.
    한글, 영어, 숫자 이외의 문자는 공백으로 처리한다.
    2글자 미만 토큰은 제거한다.
    """
    text = text.lower()

    cleaned_chars = []

    for char in text:
        if char.isalnum() or char.isspace():
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")

    cleaned_text = "".join(cleaned_chars)

    tokens = cleaned_text.split()

    tokens = [token for token in tokens if len(token) >= 2]

    return tokens


def score_document(question, document_content):
    """
    질문 토큰과 문서 토큰의 교집합 크기를 점수로 반환한다.
    set을 사용해 중복 카운트를 방지한다.
    """
    question_tokens = set(tokenize(question))
    document_tokens = set(tokenize(document_content))

    if not question_tokens or not document_tokens:
        return 0

    score = len(question_tokens.intersection(document_tokens))

    return score


def search_wiki(question, top_k=TOP_K):
    """
    질문과 관련도가 높은 문서를 top_k개 반환한다.
    점수가 같으면 파일명 오름차순으로 정렬해 결과가 일정하게 나온다.
    """
    documents = load_wiki_documents()

    scored_documents = []

    for document in documents:
        score = score_document(question, document["content"])

        scored_documents.append(
            {
                "file_name": document["file_name"],
                "content": document["content"],
                "score": score
            }
        )

    scored_documents.sort(
        key=lambda item: (-item["score"], item["file_name"])
    )

    return scored_documents[:top_k]


def build_rag_context(search_results):
    """
    검색된 문서들을 하나의 문자열로 합친다.
    각 문서 앞에 파일명 출처를 붙인다.
    """
    context_parts = []

    for result in search_results:
        context_parts.append(
            f"[문서명: {result['file_name']}]\n{result['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_source_file_names(search_results):
    """검색된 문서의 파일명 목록을 반환한다."""
    return [result["file_name"] for result in search_results]


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

    # 이전 대화 내역을 함께 전달한다.
    for message in st.session_state.docent_messages:
        messages.append(message)

    # 새 사용자 질문은 RAG Context와 함께 전달한다.
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

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
    # 사용자 질문을 화면에 표시하고 저장한다.
    st.session_state.docent_messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.write(user_question)

    # Claude 답변 생성
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

                    # 검색된 참고 문서를 expander로 표시한다.
                    with st.expander("검색된 참고 문서"):
                        for result in search_results:
                            st.write(f"- {result['file_name']} / 점수: {result['score']}")

                    # 3단계: 검색된 문서를 Claude에게 전달해 답변을 받는다.
                    answer = ask_docent(user_question, rag_context, source_files)
                    st.write(answer)

                    st.session_state.docent_messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

            except Exception as error:
                st.error(f"앳플리 봇 답변 생성 중 오류가 발생했습니다: {error}")
