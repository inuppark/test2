"""
Chapter 10 선택 실습 10-13: 앳플리 하이브리드 RAG 답변 생성

키워드 검색(Chapter 7)과 Upstage Embedding 검색(Chapter 10 선택 실습)을 결합해
최종 TOP 3 근거 청크를 만들고 Claude가 답변한다.

점수 계산:
  keyword_weight = 0.4  (Reciprocal Rank 기반)
  upstage_weight = 0.6  (Reciprocal Rank 기반)
  overlap_bonus  = 0.2  (양쪽에 모두 등장한 source_file에 가산)

사전 준비:
  1. .env에 UPSTAGE_API_KEY, ANTHROPIC_API_KEY 설정
  2. python chapters/chapter10/07_atflee_upstage_embedding_practice.py 실행
     → data/rag/atflee_upstage_embedding_index.json 생성
"""

import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hybrid_rag_utils import (
    search_hybrid_rag,
    build_hybrid_rag_context,
)

CLAUDE_MODEL_NAME = "claude-sonnet-4-5"


# ==============================
# 함수 1: API Key 로드
# ==============================
def load_api_keys():
    """
    .env 파일에서 UPSTAGE_API_KEY, ANTHROPIC_API_KEY를 읽는다.
    키 값은 절대 출력하지 않는다.
    """
    load_dotenv()

    upstage_api_key   = os.getenv("UPSTAGE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not upstage_api_key:
        raise ValueError("UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

    return upstage_api_key, anthropic_api_key


# ==============================
# 함수 2: Claude 하이브리드 RAG 답변
# ==============================
def ask_claude_with_hybrid_context(question, hybrid_results, anthropic_api_key):
    """
    하이브리드 검색 결과를 Context로 Claude에게 답변을 요청한다.
    API Key는 출력하지 않는다.
    """
    client = Anthropic(api_key=anthropic_api_key)
    hybrid_rag_context = build_hybrid_rag_context(hybrid_results)

    system_prompt = """
# Role
너는 앳플리 하이브리드 RAG 답변 봇이다.

# Goal
사용자 질문에 대해 키워드 검색과 Upstage Embedding 검색을 함께 활용해 찾은 앳플리 위키 근거로 안전하게 답변한다.

# Rules
- <hybrid_rag_context>에 있는 정보만 확정적으로 말한다.
- Context에 없는 내용은 추측하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
- 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
- 답변 마지막에는 참고한 source_file과 chunk_id를 표시한다.
- 검색 결과가 부족하면 "정확한 확인이 필요합니다"라고 말한다.

# Output Format
아래 형식으로 답변한다.

1. 간단한 답변
2. 근거가 되는 앳플리 위키 정보
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 청크
"""

    response = client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"<hybrid_rag_context>\n{hybrid_rag_context}\n</hybrid_rag_context>\n\n"
                    f"<user_question>\n{question}\n</user_question>"
                )
            }
        ]
    )

    return response.content[0].text


# ==============================
# 함수 3: 검색 결과 콘솔 출력
# ==============================
def print_hybrid_results(question, payload):
    """
    키워드 / Upstage / 하이브리드 최종 결과를 콘솔에 출력한다.
    """
    print("\n" + "=" * 100)
    print("[질문]")
    print(question)

    print("\n[키워드 검색 결과]")
    for item in payload.get("keyword_results", []):
        print(f"  - rank {item['rank']}: {item['source_file']} / score={item['raw_score']}")

    print("\n[Upstage 검색 결과]")
    for item in payload.get("upstage_results", []):
        print(f"  - rank {item['rank']}: {item['source_file']} / similarity={item['raw_score']:.4f}")

    print("\n[하이브리드 최종 TOP 3]")
    for index, item in enumerate(payload.get("hybrid_results", []), start=1):
        print(
            f"  {index}위: {item['source_file']} / "
            f"chunk_id={item.get('chunk_id')} / "
            f"hybrid_score={item['hybrid_score']:.4f} / "
            f"sources={item['sources']}"
        )


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    print("[Chapter 10 선택 실습 10-13] 앳플리 하이브리드 RAG 답변 생성")
    print("-" * 60)

    upstage_api_key, anthropic_api_key = load_api_keys()
    print("API Key 로드 완료 (보안상 값은 출력하지 않습니다.)")
    print(f"Claude 모델: {CLAUDE_MODEL_NAME}")

    test_questions = [
        "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
        "고객센터 전화번호 알려줘.",
        "제품이 불량 같고 교환하고 싶어요."
    ]

    for question in test_questions:
        payload = search_hybrid_rag(
            question=question,
            upstage_api_key=upstage_api_key,
            top_k=3
        )

        print_hybrid_results(question, payload)

        if payload.get("error"):
            print(f"\n[경고] Upstage 오류로 키워드 결과만 사용합니다: {payload['error']}")

        print("\n[Claude 하이브리드 RAG 답변]")
        answer = ask_claude_with_hybrid_context(
            question=question,
            hybrid_results=payload.get("hybrid_results", []),
            anthropic_api_key=anthropic_api_key
        )
        try:
            print(answer)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            print(answer.encode(encoding, errors="replace").decode(encoding))

        print("\n" + "-" * 100)

    print("\n[실습 완료]")
    print("키워드 검색과 Upstage Embedding 검색을 결합한 하이브리드 RAG 답변 생성이 완료되었습니다.")
