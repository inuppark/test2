"""
utils/vector_rag_utils.py

앳플리 TF-IDF 벡터 검색 공통 모듈.
AX Console, 앳플리 봇, 콘솔 실습 파일 어디서든 재사용할 수 있도록
벡터 인덱스 로드부터 RAG Context 조립까지의 기능을 제공한다.

외부 의존성 없음 — Python 표준 라이브러리만 사용한다.
Anthropic API, Streamlit은 이 파일에서 사용하지 않는다.
"""

import os
import json
import math
from collections import Counter

# 이 파일의 위치: utils/vector_rag_utils.py
# PROJECT_ROOT: utils의 한 단계 위
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# =========================
# 기본 경로 상수
# =========================

DEFAULT_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_tfidf_vector_index.json")
DEFAULT_TOP_K      = 3

# =========================
# 불용어(Stopwords)
# =========================
# 너무 자주 등장하거나 의미가 없는 단어를 벡터 계산에서 제외한다.

STOPWORDS = {
    # 한글 일반 불용어
    "그리고", "또는", "있는", "없는", "합니다", "있습니다", "됩니다",
    "대한", "관련", "기준", "확인", "안내", "경우", "사용", "고객",
    "앳플리", "atflee",
    # 영어 일반 불용어
    "the", "and", "for", "with", "this", "that",
    # 구어체 / 질문 어미
    "뭐", "어떻게", "해야", "하나요", "해요", "알려줘", "얼마나",
    "언제", "보통", "되나요", "돼요", "안", "될", "때",
}

# =========================
# 도메인 동의어 사전
# =========================
# 한글 형태소 분석기 없이도 관련 단어를 함께 검색하도록 질문 토큰을 확장한다.
# 이 사전은 질문 벡터에만 적용되고, 청크 벡터에는 적용되지 않는다.

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
    "t9은":    ["t9", "체중계", "스마트체중계", "듀얼주파수"],
    "문의":    ["문의", "고객센터", "1:1", "전화", "대표번호", "이메일"],
    "접수":    ["접수", "문의", "고객센터", "1:1", "as"],
}


# =========================
# 함수 1: 토큰 정규화
# =========================

def normalize_token(token):
    """
    영어/숫자 조합 뒤에 한글 조사가 붙는 경우를 단순 처리한다.

    예: "t9은" → "t9",  "wifi가" → "wifi"
    순수 한글 어절("앱을")은 변환하지 않는다 — 형태소 분석기가 필요한 영역이다.
    """
    token = token.lower().strip()

    for suffix in ["은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "으로"]:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            candidate = token[: -len(suffix)]
            # 앞부분에 숫자 또는 영문이 있을 때만 조사를 제거한다.
            if any(char.isdigit() for char in candidate) or any(
                "a" <= char <= "z" for char in candidate
            ):
                return candidate

    return token


# =========================
# 함수 2: 벡터 인덱스 로드
# =========================

def load_vector_index(index_path=DEFAULT_INDEX_PATH):
    """
    TF-IDF 벡터 인덱스 JSON 파일을 읽는다.

    반환:
    - 성공: index payload dict
    - 실패: None (파일 없음)
    """
    if not os.path.exists(index_path):
        return None

    with open(index_path, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================
# 함수 3: 토큰화
# =========================

def tokenize(text):
    """
    텍스트를 간단히 토큰화한다.

    처리 순서:
    1. 소문자 변환
    2. 특수문자를 공백으로 치환 (한글/영문/숫자/공백만 남긴다)
    3. 공백 기준으로 분리
    4. normalize_token 적용 (영숫자+한글조사 분리)
    5. 2글자 미만 및 불용어 제거
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
    for raw_token in raw_tokens:
        token = normalize_token(raw_token)

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
    DOMAIN_SYNONYMS를 사용해 질문 토큰을 확장한다.
    청크 벡터에는 적용하지 않고, 질문 벡터에만 적용한다.

    예: ["연결"] → ["연결", "연동", "블루투스", "권한", "페어링", "앱"]
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

    인덱스에 저장된 idf 값을 그대로 사용해야
    청크 벡터와 같은 vocabulary 공간에서 유사도를 비교할 수 있다.

    반환 예: {"블루투스": 0.12, "연결": 0.09}
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
        tf            = count / total_tokens
        vector[token] = tf * idf[token]

    return vector


# =========================
# 함수 6: 코사인 유사도
# =========================

def cosine_similarity(vector_a, vector_b):
    """
    sparse vector 두 개의 cosine similarity를 계산한다.

    cos(A, B) = dot(A, B) / (|A| × |B|)
    공통 키가 없으면 0.0을 반환한다.
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

def search_similar_chunks(
    question,
    index_payload=None,
    top_k=DEFAULT_TOP_K,
    index_path=DEFAULT_INDEX_PATH,
):
    """
    질문과 가장 유사한 청크 TOP K를 검색한다.

    index_payload를 직접 넘기면 매번 파일을 읽지 않아 성능이 좋다.
    None이면 index_path에서 자동으로 로드한다.

    반환:
    {
        "query_vector": {"token": weight, ...},
        "results": [
            {
                "similarity": float,
                "chunk_id": str,
                "source_file": str,
                "chunk_index": int,
                "text": str,
                "char_count": int,
            },
            ...
        ],
        "error": None 또는 에러 메시지
    }
    """
    if index_payload is None:
        index_payload = load_vector_index(index_path)

    if not index_payload:
        return {
            "query_vector": {},
            "results":      [],
            "error":        "벡터 인덱스를 찾을 수 없습니다.",
        }

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

    return {
        "query_vector": query_vector,
        "results":      results[:top_k],
        "error":        None,
    }


# =========================
# 함수 8: RAG Context 조립
# =========================

def build_vector_rag_context(vector_results):
    """
    검색된 청크를 Claude에게 전달할 Context 문자열로 만든다.

    각 청크 앞에 출처(source_file, chunk_id, similarity)를 붙여
    Claude가 답변에 청크를 인용할 수 있게 한다.
    """
    context_parts = []

    for result in vector_results:
        header = (
            f"[출처: {result['source_file']} / "
            f"chunk_id: {result['chunk_id']} / "
            f"similarity: {result['similarity']:.4f}]"
        )
        context_parts.append(f"{header}\n{result['text']}")

    return "\n\n---\n\n".join(context_parts)


# =========================
# 함수 9: 참고 청크 요약 반환
# =========================

def get_source_chunks(vector_results):
    """
    참고 청크 정보를 간단한 list로 반환한다.
    Streamlit UI나 로그에서 출처 표시용으로 사용한다.

    반환 예:
    [
        {"source_file": "atflee_app_guide.md", "chunk_id": "...", "similarity": 0.12, "chunk_index": 1},
        ...
    ]
    """
    return [
        {
            "source_file": result["source_file"],
            "chunk_id":    result["chunk_id"],
            "similarity":  result["similarity"],
            "chunk_index": result["chunk_index"],
        }
        for result in vector_results
    ]


# =========================
# 함수 10: 표시용 문자열 반환
# =========================

def format_vector_results_for_display(vector_results):
    """
    Streamlit 또는 콘솔 표시용 문자열 list를 반환한다.

    반환 예:
    [
        "- atflee_app_guide.md / atflee_app_guide__chunk_001 / 유사도: 0.1234",
        ...
    ]
    """
    lines = []

    for result in vector_results:
        lines.append(
            f"- {result['source_file']} / {result['chunk_id']} / 유사도: {result['similarity']:.4f}"
        )

    return lines


# =========================
# 함수 11: 질문 벡터 토큰 요약
# =========================

def summarize_query_vector(query_vector, top_n=10):
    """
    질문 벡터에서 TF-IDF 가중치가 높은 상위 토큰을 반환한다.
    디버깅 및 화면 표시용으로 사용한다.

    반환: [(token, weight), ...] — 가중치 내림차순
    """
    return sorted(query_vector.items(), key=lambda item: -item[1])[:top_n]
