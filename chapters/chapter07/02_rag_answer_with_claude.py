import os
from dotenv import load_dotenv
from anthropic import Anthropic

# utils.rag_utils에서 RAG 공통 함수를 가져온다.
# 7-4부터는 중복 구현 대신 공통 모듈을 사용한다.
from utils.rag_utils import (
    search_wiki,
    build_rag_context,
    get_source_file_names,
    TOP_K,
)

# .env 파일에서 환경변수를 불러온다.
load_dotenv()

# ==============================
# 설정값
# ==============================

model_name = "claude-sonnet-4-5"

# ==============================
# API Key 확인
# ==============================

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("[오류] ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
    print(".env 파일에 ANTHROPIC_API_KEY=sk-... 형식으로 등록해주세요.")
    exit(1)

client = Anthropic(api_key=api_key)

# ==============================
# System Prompt (Chapter 4.3 원칙 반영)
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


# ==============================
# Claude RAG 답변 함수
# ==============================

def ask_claude_with_rag(question, rag_context, source_files):
    """
    검색된 문서를 Context로 Claude에게 전달하고 답변을 받는다.
    XML 태그로 rag_context와 user_question을 구분한다.
    source_files는 참고 문서명 목록이다.
    """
    user_prompt = f"""
<rag_context>
{rag_context}
</rag_context>

<source_files>
{", ".join(source_files)}
</source_files>

<user_question>
{question}
</user_question>
"""

    response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text


# ==============================
# 검색 결과 출력 함수
# ==============================

def print_search_results(results):
    """검색된 문서 목록과 점수를 출력한다."""
    print("\n[검색된 문서]")
    print("-" * 40)

    for rank, doc in enumerate(results, start=1):
        print(f"  {rank}위. {doc['file_name']}  (점수: {doc['score']})")

    print("-" * 40)


# ==============================
# 실습용 메인 실행
# ==============================

# 테스트할 질문을 이 변수에 넣고 실행한다.
question = "앳플리 체중계가 앱이랑 연결이 안 될 때 어떻게 해야 해?"

print("=" * 60)
print(f"질문: {question}")
print("=" * 60)

# 1단계: utils.rag_utils의 search_wiki로 관련 문서를 검색한다.
search_results = search_wiki(question, top_k=TOP_K)

if not search_results:
    print("[안내] 검색된 문서가 없어 답변을 생성할 수 없습니다.")
    exit(0)

# 검색된 문서 목록 출력
print_search_results(search_results)

# 2단계: 검색된 문서만 하나의 Context로 합친다.
rag_context = build_rag_context(search_results)
source_files = get_source_file_names(search_results)

# 3단계: Claude에게 검색된 문서를 넘기고 답변을 받는다.
print("\n[Claude 답변 생성 중...]\n")

answer = ask_claude_with_rag(question, rag_context, source_files)

print(answer)
print("\n" + "=" * 60)
