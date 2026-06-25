"""
Chapter 10-4: 앳플리 TF-IDF 벡터 RAG 답변 생성 + 키워드 검색 비교

목적:
  10-3의 TF-IDF 벡터 검색 결과를 Claude에게 Context로 전달해 답변을 생성한다.
  동시에 Chapter 7의 키워드 검색 RAG 결과와 나란히 비교한다.

흐름:
  질문 → TF-IDF 벡터 검색 TOP 3 → vector_rag_context 조립 → Claude 답변
                ↕ 비교
  질문 → 키워드 검색 TOP 3 (utils.rag_utils.search_wiki)
"""

import os
import sys
import json
import math
from collections import Counter
from dotenv import load_dotenv
from anthropic import Anthropic

# 프로젝트 루트 경로를 sys.path에 추가한다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Chapter 7의 키워드 검색 유틸리티를 가져온다.
# import에 실패해도 프로그램이 종료되지 않도록 None으로 처리한다.
try:
    from utils.rag_utils import search_wiki
except ImportError:
    search_wiki = None

# =========================
# 경로 상수
# =========================

INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_tfidf_vector_index.json")
TOP_K      = 3

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
# 불용어(Stopwords)
# =========================

STOPWORDS = {
    "그리고", "또는", "있는", "없는", "합니다", "있습니다", "됩니다",
    "대한", "관련", "기준", "확인", "안내", "경우", "사용", "고객",
    "앳플리", "atflee", "the", "and", "for", "with", "this", "that",
    "뭐", "어떻게", "해야", "하나요", "해요", "알려줘", "얼마나",
    "언제", "보통", "되나요", "돼요", "안", "될", "때"
}

# =========================
# 도메인 동의어 사전 (10-3 대비 강화)
# =========================

DOMAIN_SYNONYMS = {
    "as":      ["as", "에이에스", "수리", "고장", "접수", "센터", "교환", "불량", "품질"],
    "에이에스": ["as", "에이에스", "수리", "고장", "접수", "센터", "교환", "불량", "품질"],
    "수리":    ["as", "에이에스", "수리", "고장", "접수", "센터"],
    "고장":    ["as", "에이에스", "수리", "고장", "불량", "교환", "품질"],
    "배송":    ["배송", "택배", "송장", "출고", "지연", "도착", "기간", "지역", "무료배송"],
    "택배":    ["배송", "택배", "송장", "출고", "지연", "도착"],
    "얼마나":  ["기간", "배송", "처리"],
    "걸려":    ["기간", "배송", "처리"],
    "환불":    ["환불", "반품", "취소", "카드", "승인취소", "결제취소", "영업일"],
    "반품":    ["환불", "반품", "교환", "취소", "회수"],
    "교환":    ["교환", "반품", "불량", "as", "고장"],
    "앱":      ["앱", "어플", "애플리케이션", "연동", "연결", "블루투스", "권한"],
    "어플":    ["앱", "어플", "애플리케이션", "연동", "연결", "블루투스", "권한"],
    "연결":    ["연결", "연동", "블루투스", "권한", "페어링", "앱"],
    "연동":    ["연결", "연동", "블루투스", "권한", "페어링", "앱"],
    "페어링":  ["연결", "연동", "블루투스", "권한", "페어링"],
    "체중계":  ["체중계", "스마트체중계", "스마트", "인바디", "체성분", "측정", "t9", "t8"],
    "t9":      ["t9", "체중계", "스마트체중계", "듀얼주파수", "앱", "연동"],
    # 한글 조사가 붙은 형태도 동의어로 등록한다 (normalize_token 보완용)
    "t9은":    ["t9", "체중계", "스마트체중계", "듀얼주파수"],
    "문의":    ["문의", "고객센터", "1:1", "전화", "대표번호", "이메일"],
    "접수":    ["접수", "문의", "고객센터", "1:1", "as"],
}

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
# 함수 1: 벡터 인덱스 읽기
# =========================

def load_vector_index():
    """
    10-2에서 만든 TF-IDF 벡터 인덱스를 읽는다.
    파일이 없으면 None을 반환한다.
    """
    if not os.path.exists(INDEX_PATH):
        print(f"벡터 인덱스 파일을 찾을 수 없습니다: {INDEX_PATH}")
        return None

    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================
# 함수 2: 토큰 정규화
# =========================

def normalize_token(token):
    """
    영어/숫자 조합 뒤에 한글 조사가 붙는 경우를 단순 처리한다.

    문제 상황:
      "T9은" → tokenize 후 "t9은" 이 되어 DOMAIN_SYNONYMS["t9"] 매핑 실패
    해결:
      끝부분이 한글 조사이고, 앞부분에 영문자/숫자가 있으면 조사를 제거한다.

    예: "t9은" → "t9", "wifi가" → "wifi", "앱을" (한글만) → 변환 안 함
    """
    token = token.lower().strip()

    for suffix in ["은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "으로"]:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            candidate = token[: -len(suffix)]
            # 앞부분에 숫자나 영문이 있는 경우만 조사 제거한다.
            if any(char.isdigit() for char in candidate) or any(
                "a" <= char <= "z" for char in candidate
            ):
                return candidate

    return token


# =========================
# 함수 3: 토큰화
# =========================

def tokenize(text):
    """
    텍스트를 간단히 토큰화한다.

    처리 순서:
    1. 소문자 변환
    2. 특수문자를 공백으로 치환
    3. 공백 기준으로 분리
    4. normalize_token 적용 (영숫자+조사 처리)
    5. 2글자 미만 / 불용어 제거
    """
    text = text.lower()

    cleaned_chars = []
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")

    cleaned_text = "".join(cleaned_chars)
    raw_tokens   = cleaned_text.split()

    tokens = []
    for token in raw_tokens:
        # 조사가 붙은 영숫자 조합을 정규화한다.
        token = normalize_token(token)

        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)

    return tokens


# =========================
# 함수 4: 동의어 확장
# =========================

def expand_tokens(tokens):
    """
    도메인 동의어 사전을 사용해 질문 토큰을 확장한다.
    청크 벡터에는 적용하지 않고, 질문 벡터에만 적용한다.
    """
    expanded = set(tokens)

    for token in tokens:
        synonyms = DOMAIN_SYNONYMS.get(token, [])
        for synonym in synonyms:
            expanded.add(synonym)

    return list(expanded)


# =========================
# 함수 5: 질문 벡터 생성
# =========================

def build_query_vector(question, idf):
    """
    질문을 TF-IDF sparse vector로 변환한다.
    인덱스에 저장된 idf 값을 그대로 사용해 vocabulary 공간을 일치시킨다.
    """
    tokens = tokenize(question)
    tokens = expand_tokens(tokens)

    if not tokens:
        return {}

    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    vector = {}
    for token, count in token_counts.items():
        if token not in idf:
            continue
        tf             = count / total_tokens
        vector[token]  = tf * idf[token]

    return vector


# =========================
# 함수 6: 코사인 유사도
# =========================

def cosine_similarity(vector_a, vector_b):
    """
    sparse vector 두 개의 cosine similarity를 계산한다.
    공통 키가 있는 부분만 내적에 기여한다.
    """
    if not vector_a or not vector_b:
        return 0.0

    common_tokens = set(vector_a.keys()).intersection(vector_b.keys())

    dot_product = sum(
        vector_a[token] * vector_b[token]
        for token in common_tokens
    )

    norm_a = math.sqrt(sum(v * v for v in vector_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vector_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# =========================
# 함수 7: 유사 청크 검색
# =========================

def search_similar_chunks(question, index_payload, top_k=TOP_K):
    """
    질문과 가장 유사한 청크 TOP K를 반환한다.
    query_vector도 함께 반환해 디버깅에 활용한다.
    """
    idf    = index_payload.get("idf", {})
    chunks = index_payload.get("chunks", [])

    query_vector = build_query_vector(question, idf)

    results = []
    for chunk in chunks:
        similarity = cosine_similarity(query_vector, chunk.get("vector", {}))
        results.append(
            {
                "similarity":  similarity,
                "chunk_id":    chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "text":        chunk["text"],
                "char_count":  chunk["char_count"],
            }
        )

    results.sort(
        key=lambda item: (-item["similarity"], item["source_file"], item["chunk_index"])
    )

    return results[:top_k], query_vector


# =========================
# 함수 8: 벡터 RAG Context 조립
# =========================

def build_vector_rag_context(results):
    """
    검색된 청크를 Claude에게 전달할 Context 문자열로 합친다.

    각 청크 앞에 출처(source_file, chunk_id, similarity)를 표시해
    Claude가 참고 청크를 답변에 인용할 수 있게 한다.
    """
    parts = []

    for result in results:
        header = (
            f"[출처: {result['source_file']} / "
            f"chunk_id: {result['chunk_id']} / "
            f"similarity: {result['similarity']:.4f}]"
        )
        parts.append(f"{header}\n{result['text']}")

    return "\n\n".join(parts)


# =========================
# 함수 9: Claude 답변 생성
# =========================

def ask_claude_with_vector_context(question, vector_results):
    """
    벡터 검색 결과를 Context로 붙여 Claude에게 답변을 요청한다.
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
# 함수 10: 키워드 vs 벡터 검색 비교 출력
# =========================

def compare_keyword_and_vector_search(question, vector_results):
    """
    같은 질문에 대한 키워드 검색 결과와 벡터 검색 결과를 나란히 출력한다.

    키워드 검색: utils.rag_utils.search_wiki (Chapter 7)
    벡터 검색:   TF-IDF cosine similarity (Chapter 10)
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
# 함수 11: 벡터 검색 결과 출력
# =========================

def print_vector_results(question, vector_results, query_vector):
    """질문 벡터 토큰과 벡터 검색 결과를 출력한다."""
    print("\n[질문 벡터 토큰 (상위 10개)]")
    if query_vector:
        top_tokens = sorted(query_vector.items(), key=lambda item: -item[1])[:10]
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
    print(f"  인덱스 파일: {INDEX_PATH}")
    print(f"  모델:        {model_name}")
    print(f"  search_wiki: {'사용 가능' if search_wiki else '사용 불가 (import 실패)'}")

    # 1단계: 벡터 인덱스를 읽는다.
    index_payload = load_vector_index()

    if not index_payload:
        print("먼저 10-2를 실행해 벡터 인덱스를 생성하세요.")
    else:
        print(f"\n  청크 수: {index_payload.get('chunk_count')}")
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

            # 2단계: TF-IDF 벡터 검색
            vector_results, query_vector = search_similar_chunks(
                question, index_payload, TOP_K
            )

            # 3단계: 검색 결과 출력
            print_vector_results(question, vector_results, query_vector)

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
