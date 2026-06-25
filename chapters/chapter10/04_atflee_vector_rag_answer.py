"""
Chapter 10-4: 앳플리 TF-IDF 벡터 RAG 답변 생성 + 키워드 검색 비교

목적:
  TF-IDF 벡터 검색 결과를 Claude에게 Context로 전달해 답변을 생성한다.
  동시에 Chapter 7의 키워드 검색 RAG 결과와 나란히 비교한다.

10-5 변경 사항:
  벡터 검색 로직을 utils/vector_rag_utils.py로 분리했다.
  이 파일은 Claude 답변 생성과 비교 출력만 담당한다.

흐름:
  질문 → TF-IDF 벡터 검색 TOP 3 → vector_rag_context 조립 → Claude 답변
                ↕ 비교
  질문 → 키워드 검색 TOP 3 (utils.rag_utils.search_wiki)
"""

import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

# 프로젝트 루트 경로를 sys.path에 추가한다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 벡터 검색 공통 모듈 (10-5에서 분리)
from utils.vector_rag_utils import (
    load_vector_index,
    search_similar_chunks,
    build_vector_rag_context,
    get_source_chunks,
    summarize_query_vector,
)

# Chapter 7의 키워드 검색 유틸리티 (비교용)
try:
    from utils.rag_utils import search_wiki
except ImportError:
    search_wiki = None

# =========================
# 상수
# =========================

TOP_K = 3

# =========================
# API 클라이언트 초기화
# =========================

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

client     = Anthropic(api_key=api_key)
model_name = "claude-sonnet-4-5"

# =========================
# 시스템 프롬프트
# =========================

SYSTEM_PROMPT = """
# Role
너는 앳플리 벡터 RAG 답변 봇이다.

# Goal
사용자 질문에 대해 TF-IDF 벡터 검색으로 찾은 앳플리 위키 청크를 근거로 쉽고 안전하게 답변한다.

# Rules
* <vector_rag_context>에 있는 정보만 확정적으로 말한다.
* Context에 없는 내용은 추측하지 않는다.
* 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
* 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
* 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
* 답변 마지막에는 참고한 source_file과 chunk_id를 표시한다.
* 검색 결과의 유사도가 낮으면 "정확한 확인이 필요합니다"라고 안내한다.

# Output Format
아래 형식으로 답변한다.

1. 간단한 답변
2. 근거가 되는 앳플리 위키 청크
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 청크
"""


# =========================
# 함수 1: Claude 답변 생성
# =========================

def ask_claude_with_vector_context(question, vector_results):
    """
    벡터 검색 결과를 Context로 붙여 Claude에게 답변을 요청한다.
    build_vector_rag_context는 utils.vector_rag_utils에서 가져온다.
    """
    vector_rag_context = build_vector_rag_context(vector_results)

    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"<vector_rag_context>\n{vector_rag_context}\n</vector_rag_context>\n\n"
                    f"<user_question>\n{question}\n</user_question>"
                ),
            }
        ],
    )

    return response.content[0].text


# =========================
# 함수 2: 키워드 vs 벡터 검색 비교 출력
# =========================

def compare_keyword_and_vector_search(question, vector_results):
    """
    같은 질문에 대한 키워드 검색 결과와 벡터 검색 결과를 나란히 출력한다.

    키워드 검색: utils.rag_utils.search_wiki (Chapter 7)
    벡터 검색:   utils.vector_rag_utils (Chapter 10)
    """
    print("\n[키워드 검색 vs 벡터 검색 비교]")

    # ── 키워드 검색 ───────────────────────────────────────────────
    if search_wiki is not None:
        keyword_results = search_wiki(question, top_k=3)
        print("\n  [키워드 검색 TOP 3]")
        if keyword_results:
            for rank, item in enumerate(keyword_results, start=1):
                file_name = item.get("file_name", "-")
                score     = item.get("score", 0)
                print(f"    {rank}위  {file_name}  (score: {score})")
        else:
            print("    결과 없음")
    else:
        print("  키워드 검색 비교를 건너뜁니다. (utils.rag_utils import 실패)")

    # ── 벡터 검색 ─────────────────────────────────────────────────
    print("\n  [벡터 검색 TOP 3]")
    for rank, result in enumerate(vector_results, start=1):
        print(
            f"    {rank}위  {result['source_file']}  "
            f"chunk_id: {result['chunk_id']}  "
            f"(similarity: {result['similarity']:.4f})"
        )


# =========================
# 함수 3: 벡터 검색 결과 출력
# =========================

def print_vector_results(vector_results, query_vector):
    """질문 벡터 토큰과 벡터 검색 결과를 출력한다."""
    # summarize_query_vector는 utils.vector_rag_utils에서 가져온다.
    print("\n[질문 벡터 토큰 (상위 10개)]")
    top_tokens = summarize_query_vector(query_vector, top_n=10)
    if top_tokens:
        for token, value in top_tokens:
            print(f"  - {token}: {value:.4f}")
    else:
        print("  질문 벡터가 비어 있습니다. (인덱스 vocabulary 외 단어만 포함)")

    print(f"\n[벡터 검색 TOP {TOP_K}]")
    for rank, result in enumerate(vector_results, start=1):
        print(
            f"  {rank}위  {result['source_file']}  "
            f"(similarity: {result['similarity']:.4f})  "
            f"chunk_id: {result['chunk_id']}"
        )


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    print("[앳플리 TF-IDF 벡터 RAG 답변 생성 시작]")
    print(f"  모델:        {model_name}")
    print(f"  search_wiki: {'사용 가능' if search_wiki else '사용 불가 (import 실패)'}")

    # 1단계: 벡터 인덱스를 읽는다 (utils.vector_rag_utils.load_vector_index).
    index_payload = load_vector_index()

    if not index_payload:
        print("먼저 10-2를 실행해 벡터 인덱스를 생성하세요.")
    else:
        print(f"\n  청크 수:         {index_payload.get('chunk_count')}")
        print(f"  Vocabulary 크기: {index_payload.get('vocabulary_size')}")

        test_questions = [
            "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
            "배송은 보통 얼마나 걸려?",
            "환불은 언제 처리돼?",
            "T9은 어떤 제품이야?",
            "AS 접수됐는지 확인해줘.",
        ]

        for question in test_questions:
            print("\n" + "=" * 100)
            print(f"[질문] {question}")

            # 2단계: TF-IDF 벡터 검색 (utils.vector_rag_utils.search_similar_chunks)
            # 반환 형태: {"query_vector": {...}, "results": [...], "error": None}
            search_output = search_similar_chunks(
                question, index_payload=index_payload, top_k=TOP_K
            )

            vector_results = search_output["results"]
            query_vector   = search_output["query_vector"]

            # 3단계: 검색 결과 출력
            print_vector_results(vector_results, query_vector)

            # 4단계: 키워드 검색과 비교
            compare_keyword_and_vector_search(question, vector_results)

            # 5단계: Claude 답변 생성
            print("\n[Claude 벡터 RAG 답변]")
            answer = ask_claude_with_vector_context(question, vector_results)
            print(answer)

            print("\n" + "-" * 100)

        # 6단계: TF-IDF 방식 한계 안내
        print("\n[해석]")
        print(
            "- TF-IDF 벡터 검색은 단어 가중치와 코사인 유사도를 사용하지만, "
            "한글 형태소 분석이 없어 일부 질문에서는 한계가 있습니다."
        )
        print(
            "- 실제 서비스 수준의 의미 기반 검색은 Voyage AI, OpenAI Embeddings, "
            "Claude Embeddings 등 임베딩 API 또는 한국어 형태소 분석을 함께 사용하는 것이 좋습니다."
        )
        print("- 이번 실습의 목적은 벡터 검색의 구조를 이해하는 것입니다.")
