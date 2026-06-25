"""
Chapter 10-3: 앳플리 TF-IDF 벡터 검색 실습

목적:
  10-2에서 만든 data/rag/atflee_tfidf_vector_index.json을 읽고,
  사용자 질문을 TF-IDF 벡터로 변환한 뒤,
  코사인 유사도로 가장 관련 있는 앳플리 위키 청크 TOP 3을 검색한다.

흐름:
  1. 질문 → tokenize → 도메인 동의어 확장 → TF-IDF 벡터
  2. 질문 벡터 vs 각 청크 벡터 → 코사인 유사도 계산
  3. 유사도 높은 TOP 3 청크 출력

코사인 유사도 공식:
  cos(A, B) = (A · B) / (|A| × |B|)
  - 두 벡터의 방향이 같을수록 1.0에 가깝다 (매우 유사)
  - 공통 단어가 없으면 0.0 (전혀 관련 없음)
  - 방향이 반대이면 -1.0 (의미상 반대)
"""

import os
import json
import math
from collections import Counter

# 프로젝트 루트 경로를 계산한다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# =========================
# 경로 상수
# =========================

# 10-2에서 만든 TF-IDF 벡터 인덱스 파일
INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_tfidf_vector_index.json")

# 검색 결과 개수
TOP_K = 3

# =========================
# 불용어(Stopwords)
# =========================
# 10-2와 동일한 불용어 목록을 사용한다.
# 질문에만 추가로 등장하는 구어체 표현도 포함한다.

STOPWORDS = {
    "그리고", "또는", "있는", "없는", "합니다", "있습니다", "됩니다",
    "대한", "관련", "기준", "확인", "안내", "경우", "사용", "고객",
    "앳플리", "atflee", "the", "and", "for", "with", "this", "that",
    # 질문에만 나오는 구어체 표현
    "뭐", "어떻게", "해야", "하나요", "해요", "알려줘"
}

# =========================
# 도메인 동의어 사전
# =========================
# 한글 형태소 분석 없이도 관련 단어를 함께 검색하기 위한 동의어 확장 테이블.
# 예: 사용자가 "연결"이라고 하면 "연동", "블루투스", "페어링"도 함께 검색한다.

DOMAIN_SYNONYMS = {
    "as":     ["as", "에이에스", "수리", "고장", "접수", "센터", "교환", "불량"],
    "에이에스": ["as", "에이에스", "수리", "고장", "접수", "센터", "교환", "불량"],
    "수리":   ["as", "에이에스", "수리", "고장", "접수", "센터"],
    "고장":   ["as", "에이에스", "수리", "고장", "불량", "교환"],
    "배송":   ["배송", "택배", "송장", "출고", "지연", "도착"],
    "택배":   ["배송", "택배", "송장", "출고", "지연", "도착"],
    "환불":   ["환불", "반품", "취소", "카드", "승인취소", "결제취소"],
    "반품":   ["환불", "반품", "교환", "취소", "회수"],
    "교환":   ["교환", "반품", "불량", "as", "고장"],
    "앱":     ["앱", "어플", "애플리케이션", "연동", "연결", "블루투스"],
    "어플":   ["앱", "어플", "애플리케이션", "연동", "연결", "블루투스"],
    "연결":   ["연결", "연동", "블루투스", "권한", "페어링"],
    "연동":   ["연결", "연동", "블루투스", "권한", "페어링"],
    "체중계": ["체중계", "스마트체중계", "인바디", "체성분", "측정"],
    "t9":     ["t9", "체중계", "스마트체중계", "듀얼주파수"],
    "문의":   ["문의", "고객센터", "1:1", "전화", "대표번호", "이메일"],
}


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
        print("먼저 10-2를 실행해 벡터 인덱스를 생성하세요.")
        return None

    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================
# 함수 2: 토큰화
# =========================

def tokenize(text):
    """
    텍스트를 간단히 토큰화한다.
    10-2와 동일한 방식을 사용해야 인덱스와 호환된다.

    질문 벡터와 청크 벡터를 같은 vocabulary 공간에 놓으려면
    토큰화 방식이 반드시 같아야 한다.
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
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)

    return tokens


# =========================
# 함수 3: 동의어 확장
# =========================

def expand_tokens(tokens):
    """
    도메인 동의어를 사용해 질문 토큰을 확장한다.

    한글 형태소 분석 없이 검색 품질을 높이는 방법이다.
    예: 토큰 ["연결"] → 확장 ["연결", "연동", "블루투스", "권한", "페어링"]

    주의: 동의어 확장은 질문 벡터에만 적용한다.
    청크 벡터는 10-2에서 이미 원문 그대로 인덱싱되어 있다.
    """
    expanded = set(tokens)  # 원래 토큰 유지

    for token in tokens:
        synonyms = DOMAIN_SYNONYMS.get(token, [])
        for synonym in synonyms:
            expanded.add(synonym)

    return list(expanded)


# =========================
# 함수 4: 질문 벡터 생성
# =========================

def build_query_vector(question, idf):
    """
    질문을 TF-IDF sparse vector로 변환한다.

    핵심 포인트:
    - 인덱스에 저장된 idf 값을 그대로 사용한다.
    - idf에 없는 토큰(vocabulary 밖의 단어)은 무시한다.
    - 동의어 확장으로 검색 범위를 넓힌다.

    반환 예: {"블루투스": 0.12, "연결": 0.09, "권한": 0.05}
    """
    tokens = tokenize(question)
    tokens = expand_tokens(tokens)

    if not tokens:
        return {}

    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    vector = {}

    for token, count in token_counts.items():
        # 인덱스 vocabulary에 없는 단어는 유사도 계산에 기여하지 않는다.
        if token not in idf:
            continue

        tf = count / total_tokens
        vector[token] = tf * idf[token]

    return vector


# =========================
# 함수 5: 코사인 유사도 계산
# =========================

def cosine_similarity(vector_a, vector_b):
    """
    sparse vector 두 개의 cosine similarity를 계산한다.

    sparse vector는 0이 아닌 값만 dict에 저장되어 있으므로
    공통 키(토큰)가 있는 부분만 내적(dot product)에 기여한다.

    cos(A, B) = dot(A, B) / (|A| × |B|)

    결과:
    - 1.0: 두 벡터가 완전히 같은 방향 (매우 유사)
    - 0.5: 어느 정도 유사
    - 0.0: 공통 단어 없음 (전혀 관련 없음)
    """
    if not vector_a or not vector_b:
        return 0.0

    # 두 벡터에서 공통으로 등장하는 토큰만 내적에 기여한다.
    common_tokens = set(vector_a.keys()).intersection(vector_b.keys())

    dot_product = sum(
        vector_a[token] * vector_b[token]
        for token in common_tokens
    )

    # 각 벡터의 L2 norm (크기)을 구한다.
    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# =========================
# 함수 6: 유사 청크 검색
# =========================

def search_similar_chunks(question, index_payload, top_k=TOP_K):
    """
    질문과 가장 유사한 청크 TOP K를 검색한다.

    처리 순서:
    1. 인덱스에서 idf와 청크 목록을 가져온다.
    2. 질문을 TF-IDF 벡터로 변환한다.
    3. 모든 청크와 코사인 유사도를 계산한다.
    4. 유사도 내림차순으로 정렬해 TOP K를 반환한다.

    반환:
    - results: 유사도 + 메타데이터를 담은 TOP K 리스트
    - query_vector: 디버깅용 질문 벡터
    """
    idf    = index_payload.get("idf", {})
    chunks = index_payload.get("chunks", [])

    # 질문을 TF-IDF 벡터로 변환한다.
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
                "char_count":  chunk["char_count"]
            }
        )

    # 유사도 내림차순 정렬; 동점이면 source_file, chunk_index 기준으로 안정 정렬한다.
    results.sort(
        key=lambda item: (-item["similarity"], item["source_file"], item["chunk_index"])
    )

    return results[:top_k], query_vector


# =========================
# 함수 7: 결과 출력
# =========================

def print_search_results(question, results, query_vector):
    """검색 결과를 보기 좋게 콘솔에 출력한다."""
    print("\n" + "=" * 80)
    print("[질문]")
    print(question)

    # 질문 벡터에서 TF-IDF 가중치가 높은 상위 10개 토큰을 보여준다.
    # 이 토큰들이 검색에 주로 기여한 단어들이다.
    print("\n[질문 벡터 토큰 (상위 10개)]")
    if query_vector:
        top_tokens = sorted(query_vector.items(), key=lambda item: -item[1])[:10]
        for token, value in top_tokens:
            print(f"  - {token}: {value:.4f}")
    else:
        print("  질문 벡터가 비어 있습니다. (인덱스에 없는 단어만 포함된 질문)")

    print(f"\n[검색 결과 TOP {TOP_K}]")

    for rank, result in enumerate(results, start=1):
        print(f"\n  {rank}위")
        print(f"  유사도:     {result['similarity']:.4f}")
        print(f"  source_file: {result['source_file']}")
        print(f"  chunk_id:   {result['chunk_id']}")
        print("  preview:")
        # 500자까지 미리 보기를 출력한다.
        preview = result["text"][:500].replace("\n", " ")
        print(f"    {preview}")


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    print("[앳플리 TF-IDF 벡터 검색 시작]")
    print(f"  인덱스 파일: {INDEX_PATH}")

    # 1단계: 벡터 인덱스를 읽는다.
    index_payload = load_vector_index()

    if not index_payload:
        print("먼저 10-2를 실행해 벡터 인덱스를 생성하세요.")
    else:
        print("\n[벡터 인덱스 로드 완료]")
        print(f"  청크 수:         {index_payload.get('chunk_count')}")
        print(f"  Vocabulary 크기: {index_payload.get('vocabulary_size')}")

        # 2단계: 테스트 질문 5개를 순서대로 검색한다.
        test_questions = [
            "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",   # 기대: atflee_app_guide.md
            "배송은 보통 얼마나 걸려?",                            # 기대: atflee_delivery_refund_policy.md
            "환불은 언제 처리돼?",                                 # 기대: atflee_delivery_refund_policy.md
            "T9은 어떤 제품이야?",                                 # 기대: atflee_product_guide.md
            "AS 접수됐는지 확인해줘.",                             # 기대: atflee_contact_guide.md / customer_service_policy.md
        ]

        for question in test_questions:
            results, query_vector = search_similar_chunks(question, index_payload, TOP_K)
            print_search_results(question, results, query_vector)

        print("\n" + "=" * 80)
        print("[다음 단계]")
        print("  Chapter 10-4에서는 이 벡터 검색 결과를 Claude 답변 생성에 연결합니다.")
        print("  질문 → 벡터 검색 → 관련 청크 → Claude 프롬프트 → 최종 답변")
