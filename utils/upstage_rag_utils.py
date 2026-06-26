"""
utils/upstage_rag_utils.py

Upstage Embedding 기반 RAG 공통 유틸리티.
AX Console의 "Upstage RAG" 탭과 Chapter 10 선택 실습에서 공유한다.

제공 함수:
  load_upstage_index      — 임베딩 인덱스 JSON 로드
  call_upstage_query_embedding — Upstage Query Embedding API 호출
  cosine_similarity       — 코사인 유사도 계산
  search_upstage_chunks   — TOP K 청크 검색
  build_upstage_rag_context — Claude 전달용 context 문자열 생성
  get_upstage_index_status  — Streamlit 화면 표시용 인덱스 상태 반환
"""

import os
import json
import math
import requests

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

UPSTAGE_INDEX_PATH    = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_upstage_embedding_index.json")
UPSTAGE_EMBEDDING_URL = "https://api.upstage.ai/v1/solar/embeddings"
UPSTAGE_QUERY_MODEL   = "solar-embedding-1-large-query"
DEFAULT_TOP_K         = 3


# ==============================
# 함수 1: 인덱스 로드
# ==============================
def load_upstage_index(index_path=UPSTAGE_INDEX_PATH):
    """
    Upstage 임베딩 인덱스를 읽는다.
    파일이 없으면 None을 반환한다.
    """
    if not os.path.exists(index_path):
        return None

    with open(index_path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


# ==============================
# 함수 2: Upstage Query Embedding API 호출
# ==============================
def call_upstage_query_embedding(question, api_key):
    """
    Upstage Query Embedding API를 호출한다.
    API Key는 출력하지 않는다.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json"
    }

    payload = {
        "model": UPSTAGE_QUERY_MODEL,
        "input": question
    }

    response = requests.post(
        UPSTAGE_EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstage Query Embedding API 호출 실패: "
            f"status={response.status_code}, body={response.text[:500]}"
        )

    data = response.json()

    try:
        return data["data"][0]["embedding"]
    except KeyError:
        raise RuntimeError(f"예상하지 못한 Upstage 응답 형식입니다: {data}")


# ==============================
# 함수 3: 코사인 유사도
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
# 함수 4: TOP K 청크 검색
# ==============================
def search_upstage_chunks(question, api_key, index_payload=None, top_k=DEFAULT_TOP_K):
    """
    질문을 Upstage query embedding으로 변환하고,
    저장된 passage embedding 청크들과 비교해 TOP K를 반환한다.

    반환 형태:
      {"results": [...], "error": None}
      {"results": [], "error": "오류 메시지"}
    """
    if index_payload is None:
        index_payload = load_upstage_index()

    if not index_payload:
        return {
            "results": [],
            "error":   "Upstage 임베딩 인덱스 파일을 찾을 수 없습니다."
        }

    query_embedding = call_upstage_query_embedding(question, api_key)

    results = []

    for chunk in index_payload.get("chunks", []):
        similarity = cosine_similarity(query_embedding, chunk.get("embedding", []))

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

    results.sort(key=lambda item: -item["similarity"])

    return {
        "results": results[:top_k],
        "error":   None
    }


# ==============================
# 함수 5: Claude 전달용 context 생성
# ==============================
def build_upstage_rag_context(results):
    """
    Upstage 검색 결과를 Claude에 전달할 context 문자열로 조립한다.
    """
    context_parts = []

    for result in results:
        context_parts.append(
            f"[출처: {result['source_file']} / "
            f"chunk_id: {result['chunk_id']} / "
            f"similarity: {result['similarity']:.4f}]\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


# ==============================
# 함수 6: 인덱스 상태 정보 (Streamlit 표시용)
# ==============================
def get_upstage_index_status(index_path=UPSTAGE_INDEX_PATH):
    """
    Streamlit 화면 표시용 인덱스 상태 정보를 반환한다.
    """
    if not os.path.exists(index_path):
        return {
            "exists":             False,
            "path":               index_path,
            "chunk_count":        0,
            "embedding_dimension": 0,
            "created_at":         None
        }

    try:
        with open(index_path, "r", encoding="utf-8-sig") as file:
            payload = json.load(file)

        return {
            "exists":             True,
            "path":               index_path,
            "chunk_count":        payload.get("chunk_count", 0),
            "embedding_dimension": payload.get("embedding_dimension", 0),
            "created_at":         payload.get("created_at")
        }

    except Exception:
        return {
            "exists":             False,
            "path":               index_path,
            "chunk_count":        0,
            "embedding_dimension": 0,
            "created_at":         None
        }
