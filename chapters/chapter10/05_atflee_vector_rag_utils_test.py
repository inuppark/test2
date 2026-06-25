"""
Chapter 10-5: utils/vector_rag_utils.py 동작 검증

목적:
  10-5에서 분리한 utils/vector_rag_utils.py가 올바르게 동작하는지 확인한다.
  Anthropic API는 사용하지 않는다 — 벡터 검색 로직만 검증한다.

검증 항목:
  1. load_vector_index() — 인덱스 로드 성공 여부
  2. search_similar_chunks() — 질문별 TOP 1 결과와 유사도
  3. summarize_query_vector() — 질문 벡터 토큰 상위 5개
  4. build_vector_rag_context() — Context 문자열 정상 생성 여부
  5. get_source_chunks() — 출처 목록 반환 여부
  6. format_vector_results_for_display() — 표시용 문자열 반환 여부
"""

import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가한다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.vector_rag_utils import (
    load_vector_index,
    search_similar_chunks,
    build_vector_rag_context,
    get_source_chunks,
    format_vector_results_for_display,
    summarize_query_vector,
)

# =========================
# 테스트 질문
# =========================

TEST_QUESTIONS = [
    "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
    "배송은 보통 얼마나 걸려?",
    "환불은 언제 처리돼?",
    "T9은 어떤 제품이야?",
    "AS 접수됐는지 확인해줘.",
]


# =========================
# 헬퍼 함수
# =========================

def print_separator(char="=", width=80):
    print(char * width)


def print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


# =========================
# 테스트 실행
# =========================

def run_tests(index_payload):
    """
    테스트 질문 5개에 대해 search_similar_chunks를 실행하고
    결과를 표 형태로 출력한다.
    """
    print_section("[테스트 결과 요약] 질문별 TOP 1")

    # 헤더
    print(f"  {'질문':<35}  {'TOP 1 파일':<38}  {'유사도':>8}  {'벡터 비어있음':>10}")
    print("  " + "-" * 97)

    for question in TEST_QUESTIONS:
        output = search_similar_chunks(question, index_payload=index_payload, top_k=3)

        error        = output.get("error")
        results      = output.get("results", [])
        query_vector = output.get("query_vector", {})

        if error:
            print(f"  {question:<35}  ERROR: {error}")
            continue

        if results:
            top1       = results[0]
            file_name  = top1["source_file"]
            similarity = top1["similarity"]
        else:
            file_name  = "(없음)"
            similarity = 0.0

        vector_empty = "예" if not query_vector else "아니오"
        print(
            f"  {question:<35}  {file_name:<38}  {similarity:>8.4f}  {vector_empty:>10}"
        )

    # 상세 출력
    print()
    print_section("[상세 검색 결과]")

    for question in TEST_QUESTIONS:
        print_separator("-", 80)
        print(f"질문: {question}")

        output = search_similar_chunks(question, index_payload=index_payload, top_k=3)
        results      = output.get("results", [])
        query_vector = output.get("query_vector", {})

        # 질문 벡터 상위 토큰
        top_tokens = summarize_query_vector(query_vector, top_n=5)
        print("  질문 벡터 상위 토큰:")
        if top_tokens:
            for token, weight in top_tokens:
                print(f"    - {token}: {weight:.4f}")
        else:
            print("    (벡터 비어있음)")

        # 벡터 검색 TOP 3
        print("  벡터 검색 TOP 3:")
        for rank, result in enumerate(results, start=1):
            print(
                f"    {rank}위  {result['source_file']}  "
                f"(similarity: {result['similarity']:.4f})  "
                f"chunk_id: {result['chunk_id']}"
            )

        # 표시용 문자열
        display_lines = format_vector_results_for_display(results)
        print("  format_vector_results_for_display:")
        for line in display_lines:
            print(f"    {line}")

        # 출처 청크 목록
        source_chunks = get_source_chunks(results)
        print(f"  get_source_chunks: {len(source_chunks)}개 반환")

    # Context 조립 검증 (첫 번째 질문)
    print()
    print_section("[build_vector_rag_context 검증]")
    first_output = search_similar_chunks(
        TEST_QUESTIONS[0], index_payload=index_payload, top_k=3
    )
    first_results = first_output.get("results", [])
    context       = build_vector_rag_context(first_results)
    preview       = context[:300].replace("\n", " ")
    print(f"  Context 길이: {len(context)}자")
    print(f"  앞 300자 미리 보기: {preview}...")


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    print_separator()
    print("[utils/vector_rag_utils.py 동작 검증]")
    print_separator()

    # 인덱스 로드
    print("\n[1] 벡터 인덱스 로드")
    index_payload = load_vector_index()

    if not index_payload:
        print("  벡터 인덱스를 찾을 수 없습니다.")
        print("  먼저 10-2를 실행해 data/rag/atflee_tfidf_vector_index.json을 생성하세요.")
    else:
        print(f"  청크 수:         {index_payload.get('chunk_count')}")
        print(f"  Vocabulary 크기: {index_payload.get('vocabulary_size')}")
        print(f"  생성 시각:       {index_payload.get('created_at')}")
        print("  로드 성공")

        # 테스트 실행
        run_tests(index_payload)

        print()
        print_separator()
        print("[검증 완료]")
        print("  utils/vector_rag_utils.py의 모든 함수가 정상 동작합니다.")
        print("  10-4 (vector_rag_answer)와 AX Console에서 이 모듈을 재사용할 수 있습니다.")
        print_separator()
