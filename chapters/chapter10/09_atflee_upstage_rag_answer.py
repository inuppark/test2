"""
Chapter 10 선택 실습 10-9: Upstage Embedding RAG 답변 생성

07_atflee_upstage_embedding_practice.py 에서 만든 Upstage 임베딩 인덱스를 사용해
질문과 유사한 청크 TOP 3을 찾고, 그 내용을 Context로 Claude에 전달해 RAG 답변을 생성한다.

흐름:
  질문 → Upstage query 임베딩 → 코사인 유사도 TOP 3 청크 → Claude 답변 생성

사전 준비:
  1. .env에 UPSTAGE_API_KEY, ANTHROPIC_API_KEY 설정
  2. python chapters/chapter10/07_atflee_upstage_embedding_practice.py 실행
     → data/rag/atflee_upstage_embedding_index.json 생성
"""

import os
import sys
import json
import math

from dotenv import load_dotenv
import requests
from anthropic import Anthropic

# 프로젝트 루트를 sys.path에 추가한다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Upstage 임베딩 인덱스 파일 경로
UPSTAGE_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_upstage_embedding_index.json")

# Upstage Embedding API 설정
UPSTAGE_EMBEDDING_URL = "https://api.upstage.ai/v1/solar/embeddings"
UPSTAGE_MODEL_QUERY   = "solar-embedding-1-large-query"    # 질문(query) 임베딩용
UPSTAGE_MODEL_PASSAGE = "solar-embedding-1-large-passage"  # 청크(passage) 임베딩용

TOP_K            = 3
CLAUDE_MODEL_NAME = "claude-sonnet-4-5"


# ==============================
# 시스템 프롬프트
# ==============================
SYSTEM_PROMPT = """
# Role
너는 앳플리 Upstage Embedding RAG 답변 봇이다.

# Goal
사용자 질문에 대해 Upstage 임베딩 벡터 검색으로 찾은 앳플리 위키 청크를 근거로
쉽고 안전하게 답변한다.

# Rules
* <upstage_rag_context>에 있는 정보만 확정적으로 말한다.
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


# ==============================
# 함수 1: 환경 변수 로드
# ==============================
def load_env():
    """
    .env 파일에서 ANTHROPIC_API_KEY 와 UPSTAGE_API_KEY 를 읽는다.
    키 값은 절대 출력하지 않는다.
    """
    load_dotenv()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    upstage_key   = os.getenv("UPSTAGE_API_KEY")

    if not anthropic_key:
        raise ValueError(
            "ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다."
        )

    if not upstage_key:
        raise ValueError(
            "UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.\n"
            "Upstage Console에서 API Key를 발급받아 .env에 추가하세요.\n"
            "예: UPSTAGE_API_KEY=up_xxxxxxxxxxxxxxxxxxxx"
        )

    return anthropic_key, upstage_key


# ==============================
# 함수 2: Upstage 임베딩 인덱스 로드
# ==============================
def load_upstage_index():
    """
    data/rag/atflee_upstage_embedding_index.json을 읽어 반환한다.
    파일이 없으면 None을 반환하고 생성 방법을 안내한다.
    """
    if not os.path.exists(UPSTAGE_INDEX_PATH):
        print(f"[안내] Upstage 임베딩 인덱스 파일이 없습니다: {UPSTAGE_INDEX_PATH}")
        print("먼저 아래 명령어를 실행하세요.")
        print("  python chapters/chapter10/07_atflee_upstage_embedding_practice.py")
        return None

    with open(UPSTAGE_INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ==============================
# 함수 3: Upstage Embedding API 호출
# ==============================
def call_upstage_embedding(text, upstage_key, model=None):
    """
    Upstage Embedding API를 호출해 text의 임베딩 벡터(리스트)를 반환한다.
    model 미지정 시 query 모델을 사용한다.
    API Key는 헤더에만 사용하고 절대 출력하지 않는다.
    """
    if model is None:
        model = UPSTAGE_MODEL_QUERY

    headers = {
        "Authorization": f"Bearer {upstage_key}",
        "Content-Type":  "application/json"
    }

    payload = {
        "model": model,
        "input": text
    }

    response = requests.post(
        UPSTAGE_EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstage Embedding API 호출 실패: "
            f"status={response.status_code}, body={response.text[:300]}"
        )

    data = response.json()

    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError):
        raise RuntimeError(f"예상하지 못한 응답 형식입니다: {str(data)[:300]}")


# ==============================
# 함수 4: 코사인 유사도 계산
# ==============================
def cosine_similarity(vector_a, vector_b):
    """두 dense 벡터의 코사인 유사도를 계산한다. (−1 ~ 1)"""
    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ==============================
# 함수 5: Upstage 임베딩 검색
# ==============================
def search_upstage(question, upstage_index, upstage_key, top_k=TOP_K):
    """
    질문을 Upstage query 임베딩으로 변환한 뒤
    인덱스의 각 청크와 코사인 유사도를 계산해 TOP K를 반환한다.
    """
    query_embedding = call_upstage_embedding(question, upstage_key, model=UPSTAGE_MODEL_QUERY)

    scored = []
    for chunk in upstage_index.get("chunks", []):
        sim = cosine_similarity(query_embedding, chunk.get("embedding", []))
        scored.append(
            {
                "similarity":  sim,
                "chunk_id":    chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "text":        chunk["text"],
                "char_count":  chunk["char_count"]
            }
        )

    scored.sort(key=lambda x: -x["similarity"])
    return scored[:top_k]


# ==============================
# 함수 6: RAG Context 조립
# ==============================
def build_upstage_rag_context(search_results):
    """
    Upstage 검색 결과 청크들을 Claude에게 전달할 Context 문자열로 조립한다.
    """
    lines = []

    for rank, result in enumerate(search_results, start=1):
        lines.append(f"[청크 {rank}]")
        lines.append(f"source_file: {result['source_file']}")
        lines.append(f"chunk_id:    {result['chunk_id']}")
        lines.append(f"similarity:  {result['similarity']:.4f}")
        lines.append("")
        lines.append(result["text"])
        lines.append("")

    return "\n".join(lines)


# ==============================
# 함수 7: Claude RAG 답변 생성
# ==============================
def ask_claude_with_upstage_context(question, search_results, client, model_name):
    """
    Upstage 검색 결과를 Context로 붙여 Claude에게 답변을 요청한다.
    """
    rag_context = build_upstage_rag_context(search_results)

    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"<upstage_rag_context>\n{rag_context}\n</upstage_rag_context>\n\n"
                    f"<user_question>\n{question}\n</user_question>"
                )
            }
        ]
    )

    return response.content[0].text


# ==============================
# 함수 8: 검색 결과 콘솔 출력
# ==============================
def print_search_results(search_results):
    """Upstage 검색 TOP K 결과를 콘솔에 출력한다."""
    print(f"\n[Upstage Embedding 검색 TOP {len(search_results)}]")

    for rank, result in enumerate(search_results, start=1):
        print(
            f"  {rank}위  {result['source_file']:<40} "
            f"(similarity: {result['similarity']:.4f})  "
            f"chunk_id: {result['chunk_id']}"
        )


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    print("[Chapter 10 선택 실습 10-9] 앳플리 Upstage Embedding RAG 답변 생성")
    print("-" * 60)

    # API Key 로드 — 값은 출력하지 않는다.
    anthropic_key, upstage_key = load_env()
    print("API Key 로드 완료 (값은 보안상 출력하지 않습니다.)")

    # Upstage 임베딩 인덱스 로드
    upstage_index = load_upstage_index()

    if not upstage_index:
        print("\n실습을 종료합니다.")
    else:
        print(
            f"Upstage 인덱스 로드 완료 / "
            f"청크 수: {upstage_index.get('chunk_count')}, "
            f"임베딩 차원: {upstage_index.get('embedding_dimension')}, "
            f"passage 모델: {upstage_index.get('embedding_model_passage')}"
        )

        # Claude 클라이언트 초기화
        client     = Anthropic(api_key=anthropic_key)
        model_name = CLAUDE_MODEL_NAME
        print(f"Claude 모델: {model_name}")

        # 테스트 질문 목록 — 07/08 파일과 동일하게 맞춘다.
        test_questions = [
            "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
            "블루투스 페어링이 계속 실패해요.",
            "배송은 보통 얼마나 걸려?",
            "환불은 언제 처리돼?",
            "T9은 어떤 제품이야?",
            "AS 접수됐는지 확인해줘.",
            "제품이 불량 같고 교환하고 싶어요."
        ]

        print(f"\n[테스트 질문 {len(test_questions)}개 RAG 답변 시작]")

        for question in test_questions:
            print("\n" + "=" * 100)
            print(f"[질문] {question}")

            # 1단계: Upstage 임베딩 검색 TOP 3
            search_results = search_upstage(question, upstage_index, upstage_key, top_k=TOP_K)
            print_search_results(search_results)

            # 2단계: Claude RAG 답변 생성
            print("\n[Claude Upstage RAG 답변]")
            answer = ask_claude_with_upstage_context(question, search_results, client, model_name)
            try:
                print(answer)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or "utf-8"
                print(answer.encode(encoding, errors="replace").decode(encoding))

            print("\n" + "-" * 100)

        print("\n[실습 완료]")
        print("Upstage Embedding 검색 → Claude RAG 답변 파이프라인이 정상 작동했습니다.")
        print(f"다음 단계: 08_atflee_three_way_rag_compare_report.py 에서 세 방식(키워드/TF-IDF/Upstage)을 비교할 수 있습니다.")
