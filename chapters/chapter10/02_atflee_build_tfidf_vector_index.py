"""
Chapter 10-2: 앳플리 위키 TF-IDF 벡터 인덱스 구축

목적:
  10-1에서 만든 data/rag/atflee_wiki_chunks.json을 읽고,
  각 청크를 TF-IDF 벡터로 변환해 data/rag/atflee_tfidf_vector_index.json에 저장한다.

배경:
  임베딩 API를 사용하기 전에, Python 표준 라이브러리만으로 TF-IDF 기반 벡터 인덱스를 만들어
  "텍스트 → 숫자 벡터 → 유사도 검색" 구조를 직접 이해한다.

TF-IDF 개념:
  - TF (Term Frequency):  해당 단어가 문서 내에서 얼마나 자주 등장하는가
  - IDF (Inverse Document Frequency): 전체 문서 중 몇 개에만 등장하는 희귀한 단어인가
  - TF-IDF = TF × IDF: 특정 문서에서 특히 중요한 단어일수록 높은 점수를 가진다
  - Sparse Vector: 전체 단어 공간 중 해당 문서에 등장한 단어만 값을 가지는 벡터
"""

import os
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

# 프로젝트 루트 경로를 계산한다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# =========================
# 경로 상수
# =========================

# 10-1에서 만든 청크 파일
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_wiki_chunks.json")

# 벡터 인덱스 저장 경로
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "rag")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "atflee_tfidf_vector_index.json")

# =========================
# 불용어(Stopwords)
# =========================
# 너무 자주 등장해 의미가 없는 단어는 벡터에서 제외한다.
# TF-IDF 자체도 불용어를 낮게 평가하지만, 명시적으로 제거하면 벡터가 더 간결해진다.

STOPWORDS = {
    "그리고", "또는", "있는", "없는", "합니다", "있습니다", "됩니다",
    "대한", "관련", "기준", "확인", "안내", "경우", "사용", "고객",
    "앳플리", "atflee", "the", "and", "for", "with", "this", "that"
}


# =========================
# 함수 1: 청크 파일 읽기
# =========================

def load_chunks():
    """
    10-1에서 만든 청크 JSON 파일을 읽는다.
    파일이 없으면 빈 리스트를 반환한다.
    """
    if not os.path.exists(CHUNKS_PATH):
        print(f"청크 파일을 찾을 수 없습니다: {CHUNKS_PATH}")
        print("먼저 10-1을 실행해 청크 파일을 생성하세요.")
        return []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return payload.get("chunks", [])


# =========================
# 함수 2: 토큰화
# =========================

def tokenize(text):
    """
    텍스트를 간단히 토큰화한다.

    처리 순서:
    1. 소문자로 변환한다.
    2. 특수문자(한글/영문/숫자/공백 외)를 공백으로 바꾼다.
    3. 공백 기준으로 단어를 나눈다.
    4. 2글자 미만 토큰을 제거한다.
    5. 불용어를 제거한다.

    한글 형태소 분석기(konlpy 등)를 쓰지 않아 정확도는 제한적이지만,
    외부 라이브러리 없이 TF-IDF 개념을 이해하기에 충분하다.
    """
    text = text.lower()

    # 한글, 영문, 숫자, 공백이 아닌 문자를 공백으로 치환한다.
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
        # 너무 짧은 토큰은 의미가 없으므로 제외한다.
        if len(token) < 2:
            continue
        # 불용어를 제외한다.
        if token in STOPWORDS:
            continue
        tokens.append(token)

    return tokens


# =========================
# 함수 3: Vocabulary 구축
# =========================

def build_vocabulary(chunks):
    """
    모든 청크에서 등장한 단어 목록(vocabulary)을 만든다.

    vocabulary는 전체 단어의 정렬된 리스트다.
    token_to_index는 단어 → 인덱스 번호 매핑 dict다.
    (Dense 벡터를 만들 때 인덱스가 필요하지만, Sparse 벡터에서는 선택적이다.)
    """
    vocab_set = set()

    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        vocab_set.update(tokens)

    # 정렬해서 일관된 순서를 보장한다.
    vocabulary = sorted(vocab_set)

    token_to_index = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    return vocabulary, token_to_index


# =========================
# 함수 4: Document Frequency 계산
# =========================

def compute_document_frequency(chunks):
    """
    각 단어가 몇 개의 청크(문서)에 등장했는지 계산한다.

    Document Frequency(DF):
    - "블루투스"가 8개 청크 중 5개에 등장하면 DF("블루투스") = 5
    - DF가 높을수록 흔한 단어 → IDF가 낮아진다
    """
    document_frequency = defaultdict(int)

    for chunk in chunks:
        # 같은 청크 안에서 중복 카운트를 막기 위해 set을 사용한다.
        tokens = set(tokenize(chunk["text"]))

        for token in tokens:
            document_frequency[token] += 1

    return dict(document_frequency)


# =========================
# 함수 5: IDF 계산
# =========================

def compute_idf(document_frequency, total_docs):
    """
    IDF(Inverse Document Frequency)를 계산한다.

    공식: IDF(t) = log((1 + N) / (1 + DF(t))) + 1
    - N: 전체 청크(문서) 수
    - DF(t): 해당 단어가 등장한 청크 수
    - + 1 스무딩: DF가 0인 경우 분모가 0이 되는 것을 방지한다.
    - + 1 (공식 마지막): IDF가 0이 되는 것을 방지해 모든 단어가 양수 가중치를 가진다.

    결과:
    - 모든 청크에 등장하는 단어: IDF ≈ 1.0 (낮음)
    - 1개 청크에만 등장하는 단어: IDF ≈ 높음 → 특정 문서 식별에 중요
    """
    idf = {}

    for token, df in document_frequency.items():
        idf[token] = math.log((1 + total_docs) / (1 + df)) + 1

    return idf


# =========================
# 함수 6: Sparse TF-IDF 벡터 생성
# =========================

def build_sparse_tfidf_vector(text, idf):
    """
    청크 하나의 TF-IDF sparse vector를 만든다.

    Sparse Vector:
    - Dense 벡터는 vocabulary 크기만큼의 배열 (대부분 0)
    - Sparse 벡터는 값이 0이 아닌 단어만 dict로 저장
    - 예: {"블루투스": 0.34, "연결": 0.52, "앱": 0.28}

    TF 공식: count(단어) / 전체 토큰 수
    TF-IDF: TF × IDF
    """
    tokens = tokenize(text)

    if not tokens:
        return {}

    # 각 단어의 등장 횟수를 센다.
    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    vector = {}

    for token, count in token_counts.items():
        # IDF에 없는 단어(토큰화 후 어휘에 없는 경우)는 건너뛴다.
        if token not in idf:
            continue

        # TF: 이 청크에서의 상대적 등장 빈도
        tf = count / total_tokens

        # TF-IDF: 높을수록 이 청크에서 특별히 중요한 단어
        vector[token] = tf * idf[token]

    return vector


# =========================
# 함수 7: 전체 벡터 인덱스 구축
# =========================

def build_vector_index(chunks):
    """
    전체 청크에 대해 TF-IDF 벡터 인덱스를 만든다.

    처리 순서:
    1. Vocabulary 구축
    2. Document Frequency 계산
    3. IDF 계산
    4. 각 청크를 Sparse TF-IDF 벡터로 변환
    5. 인덱스 payload 구성
    """
    print("[1/4] Vocabulary 구축 중...")
    vocabulary, token_to_index = build_vocabulary(chunks)

    print("[2/4] Document Frequency 계산 중...")
    document_frequency = compute_document_frequency(chunks)

    print("[3/4] IDF 계산 중...")
    idf = compute_idf(document_frequency, len(chunks))

    print("[4/4] 청크별 TF-IDF 벡터 변환 중...")
    indexed_chunks = []

    for chunk in chunks:
        vector = build_sparse_tfidf_vector(chunk["text"], idf)

        indexed_chunks.append(
            {
                "chunk_id":    chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "text":        chunk["text"],
                "char_count":  chunk["char_count"],
                "vector":      vector
            }
        )

    index_payload = {
        "created_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project":            "atflee",
        "index_type":         "tfidf_sparse_vector",
        "description":        "TF-IDF vector index for Atflee wiki chunks",
        "chunk_count":        len(chunks),
        "vocabulary_size":    len(vocabulary),
        "vocabulary":         vocabulary,
        "document_frequency": document_frequency,
        "idf":                idf,
        "chunks":             indexed_chunks
    }

    return index_payload


# =========================
# 함수 8: 인덱스 저장
# =========================

def save_index(index_payload):
    """벡터 인덱스를 data/rag/atflee_tfidf_vector_index.json에 저장한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(index_payload, file, ensure_ascii=False, indent=2)

    return OUTPUT_PATH


# =========================
# 함수 9: 요약 출력
# =========================

def print_summary(index_payload, saved_path):
    """인덱스 구축 결과를 콘솔에 출력한다."""
    print("\n[TF-IDF 벡터 인덱스 생성 요약]")
    print(f"  청크 수:        {index_payload['chunk_count']}")
    print(f"  Vocabulary 크기: {index_payload['vocabulary_size']}")
    print(f"  저장 경로:       {saved_path}")

    chunks = index_payload.get("chunks", [])

    if chunks:
        first = chunks[0]
        # TF-IDF 값이 높은 순으로 정렬해 중요 단어를 먼저 보여준다.
        vector_items = sorted(
            first["vector"].items(),
            key=lambda item: item[1],
            reverse=True
        )[:10]

        print("\n[첫 번째 청크 벡터 (상위 10개 단어)]")
        print(f"  chunk_id:    {first['chunk_id']}")
        print(f"  source_file: {first['source_file']}")
        print("  vector sample (높은 TF-IDF 순):")
        for token, value in vector_items:
            print(f"    - {token}: {value:.4f}")


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    print("[앳플리 TF-IDF 벡터 인덱스 구축 시작]")
    print(f"  입력 파일: {CHUNKS_PATH}")
    print(f"  출력 파일: {OUTPUT_PATH}")

    # 1단계: 10-1에서 만든 청크를 읽는다.
    chunks = load_chunks()

    if not chunks:
        print("\n청크가 없습니다. 먼저 10-1을 실행해 data/rag/atflee_wiki_chunks.json을 생성하세요.")
    else:
        print(f"\n  로드된 청크 수: {len(chunks)}")

        # 2단계: TF-IDF 벡터 인덱스를 만든다.
        index_payload = build_vector_index(chunks)

        # 3단계: 인덱스를 JSON으로 저장한다.
        saved_path = save_index(index_payload)

        # 4단계: 요약을 출력한다.
        print_summary(index_payload, saved_path)

        print("\n[저장 완료]")
        print(f"  {saved_path}")

        print("\n[다음 단계]")
        print("  Chapter 10-3에서는 질문도 TF-IDF 벡터로 변환하고,")
        print("  코사인 유사도로 관련 청크를 검색합니다.")
