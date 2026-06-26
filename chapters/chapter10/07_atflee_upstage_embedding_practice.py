"""
Chapter 10 선택 실습: Upstage Embedding API로 앳플리 벡터 검색 고도화

TF-IDF 방식 대신 실제 외부 임베딩 API를 사용해 청크를 벡터로 변환하고
사용자 질문도 동일한 임베딩으로 변환해 코사인 유사도로 TOP 3를 검색한다.
이번 파일은 검색 품질 확인용 콘솔 실습이며 Claude 답변 생성은 포함하지 않는다.

사전 준비:
  1. Upstage Console(https://console.upstage.ai)에서 API Key 발급
  2. .env 파일에 UPSTAGE_API_KEY=<발급받은 키> 추가
  3. pip install requests python-dotenv
"""

import os
import sys
import json
import math
import time
from datetime import datetime

from dotenv import load_dotenv
import requests

# 프로젝트 루트를 sys.path에 추가한다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 입력/출력 파일 경로
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_wiki_chunks.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_upstage_embedding_index.json")

# Upstage Embedding API 설정
# 2026년 기준 모델명이 쿼리용/문서용으로 분리됐다.
# 최신 모델 목록: https://console.upstage.ai/docs/models
UPSTAGE_EMBEDDING_URL    = "https://api.upstage.ai/v1/solar/embeddings"
UPSTAGE_MODEL_QUERY      = "solar-embedding-1-large-query"    # 질문(query) 임베딩용
UPSTAGE_MODEL_PASSAGE    = "solar-embedding-1-large-passage"  # 문서/청크(passage) 임베딩용


# ==============================
# 함수 1: API Key 로드
# ==============================
def load_api_key():
    """
    .env 파일에서 UPSTAGE_API_KEY를 읽는다.
    키 값은 절대 출력하지 않는다.
    """
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")

    if not api_key:
        raise ValueError(
            "UPSTAGE_API_KEY가 .env 파일에 설정되어 있지 않습니다.\n"
            "Upstage Console에서 API Key를 발급받아 .env에 추가하세요.\n"
            "예: UPSTAGE_API_KEY=up_xxxxxxxxxxxxxxxxxxxx"
        )

    return api_key


# ==============================
# 함수 2: 청크 파일 로드
# ==============================
def load_chunks():
    """
    data/rag/atflee_wiki_chunks.json을 읽어 청크 목록을 반환한다.
    파일이 없으면 10-1 스크립트를 먼저 실행하라고 안내한다.
    """
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"청크 파일을 찾을 수 없습니다: {CHUNKS_PATH}\n"
            "먼저 python chapters/chapter10/01_atflee_chunk_wiki_documents.py 를 실행하세요."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return payload.get("chunks", [])


# ==============================
# 함수 3: Upstage Embedding API 호출
# ==============================
def call_upstage_embedding(text, api_key, model=None):
    """
    Upstage Embedding API를 호출해 text의 임베딩 벡터(리스트)를 반환한다.
    model을 지정하지 않으면 query 모델을 기본으로 사용한다.
    청크(문서) 임베딩은 UPSTAGE_MODEL_PASSAGE,
    질문 임베딩은 UPSTAGE_MODEL_QUERY를 전달한다.
    """
    if model is None:
        model = UPSTAGE_MODEL_QUERY

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
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
            f"status={response.status_code}, body={response.text[:500]}"
        )

    data = response.json()

    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError):
        raise RuntimeError(f"예상하지 못한 응답 형식입니다: {data}")


# ==============================
# 함수 4: 코사인 유사도 계산
# ==============================
def cosine_similarity(vector_a, vector_b):
    """
    두 벡터의 코사인 유사도를 계산한다. (−1 ~ 1 범위, 1에 가까울수록 유사)
    """
    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ==============================
# 함수 5: 임베딩 인덱스 생성
# ==============================
def build_upstage_embedding_index(chunks, api_key, sleep_seconds=0.2):
    """
    각 청크를 Upstage Embedding API로 벡터화한다.
    API Rate Limit을 고려해 청크 사이에 짧은 sleep을 넣는다.
    """
    indexed_chunks = []
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        print(f"  [{index}/{total}] 임베딩 생성 중: {chunk['chunk_id']}")

        # 청크(문서)는 passage 모델로 임베딩한다.
        embedding = call_upstage_embedding(chunk["text"], api_key, model=UPSTAGE_MODEL_PASSAGE)

        indexed_chunks.append(
            {
                "chunk_id":    chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "text":        chunk["text"],
                "char_count":  chunk["char_count"],
                "embedding":   embedding
            }
        )

        # API 연속 호출 사이에 짧게 대기한다.
        time.sleep(sleep_seconds)

    payload = {
        "created_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project":           "atflee",
        "index_type":        "upstage_embedding",
        "embedding_model_passage": UPSTAGE_MODEL_PASSAGE,
        "embedding_model_query":   UPSTAGE_MODEL_QUERY,
        "embedding_url":           UPSTAGE_EMBEDDING_URL,
        "chunk_count":       len(indexed_chunks),
        "embedding_dimension": len(indexed_chunks[0]["embedding"]) if indexed_chunks else 0,
        "chunks":            indexed_chunks
    }

    return payload


# ==============================
# 함수 6: 임베딩 인덱스 저장
# ==============================
def save_embedding_index(index_payload):
    """
    임베딩 인덱스를 data/rag/atflee_upstage_embedding_index.json에 저장한다.
    임베딩 벡터는 각 차원이 float이므로 파일이 크다. (수 MB 예상)
    """
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 임베딩 숫자는 indent 없이 저장해 파일 크기를 줄인다.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(index_payload, file, ensure_ascii=False)

    return OUTPUT_PATH


# ==============================
# 함수 7: 임베딩 기반 검색
# ==============================
def search_upstage_embedding(question, index_payload, api_key, top_k=3):
    """
    질문을 Upstage 임베딩으로 변환한 뒤
    인덱스의 각 청크 임베딩과 코사인 유사도를 계산해 TOP K를 반환한다.
    """
    print(f"  질문 임베딩 생성 중: {question[:40]}...")
    # 질문은 query 모델로 임베딩한다.
    query_embedding = call_upstage_embedding(question, api_key, model=UPSTAGE_MODEL_QUERY)

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

    return results[:top_k]


# ==============================
# 함수 8: 검색 결과 콘솔 출력
# ==============================
def print_search_results(question, results):
    """
    질문과 TOP 3 검색 결과를 콘솔에 출력한다.
    """
    print("\n" + "=" * 80)
    print("[질문]")
    print(question)
    print("\n[Upstage Embedding 검색 TOP 3]")

    for rank, result in enumerate(results, start=1):
        print(f"\n  {rank}위")
        print(f"  유사도:      {result['similarity']:.4f}")
        print(f"  source_file: {result['source_file']}")
        print(f"  chunk_id:    {result['chunk_id']}")
        print("  preview:")
        # BOM(﻿) 등 Windows 터미널에서 출력 불가한 문자를 제거하고 400자로 제한한다.
        preview = result["text"].replace("﻿", "")[:400].replace("\n", "\n  ")
        try:
            print("  " + preview)
        except UnicodeEncodeError:
            print("  " + preview.encode("utf-8", errors="replace").decode("utf-8"))


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    print("[Chapter 10 선택 실습] 앳플리 Upstage Embedding 벡터 검색")
    print("-" * 60)

    # API Key 로드 — 값은 출력하지 않는다.
    api_key = load_api_key()
    print("API Key 로드 완료 (보안상 값은 출력하지 않습니다.)")

    # 청크 파일 로드
    chunks = load_chunks()

    if not chunks:
        print("청크가 없습니다. 먼저 10-1 청크 생성을 실행하세요.")
    else:
        print(f"청크 수: {len(chunks)}")

        # 기존 인덱스가 있으면 재사용, 없으면 새로 생성한다.
        if os.path.exists(OUTPUT_PATH):
            print(f"\n[기존 Upstage 임베딩 인덱스 사용]")
            print(f"경로: {OUTPUT_PATH}")
            with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
                index_payload = json.load(file)
            print(f"청크 수: {index_payload.get('chunk_count')}")
            print(f"임베딩 차원: {index_payload.get('embedding_dimension')}")
            print(f"생성 시각: {index_payload.get('created_at')}")
        else:
            print(f"\n[새 Upstage 임베딩 인덱스 생성 시작]")
            print(f"passage 모델: {UPSTAGE_MODEL_PASSAGE}")
            print(f"query 모델:   {UPSTAGE_MODEL_QUERY}")
            index_payload = build_upstage_embedding_index(chunks, api_key)
            saved_path = save_embedding_index(index_payload)

            print(f"\n[저장 완료]")
            print(f"경로: {saved_path}")
            print(f"임베딩 차원: {index_payload.get('embedding_dimension')}")

        # 테스트 질문 목록 — 10-8의 비교 대상과 동일하게 맞춘다.
        test_questions = [
            "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
            "블루투스 페어링이 계속 실패해요.",
            "배송은 보통 얼마나 걸려?",
            "환불은 언제 처리돼?",
            "T9은 어떤 제품이야?",
            "AS 접수됐는지 확인해줘.",
            "제품이 불량 같고 교환하고 싶어요."
        ]

        print(f"\n[테스트 질문 {len(test_questions)}개 검색 시작]")

        for question in test_questions:
            results = search_upstage_embedding(question, index_payload, api_key, top_k=3)
            print_search_results(question, results)

        print("\n" + "=" * 80)
        print("[다음 단계]")
        print("Upstage 임베딩 검색 결과와 TF-IDF/키워드 검색 결과를 비교하는 리포트를 만들 수 있습니다.")
        print("또는 TOP 3 청크를 Claude에 전달해 실제 RAG 답변을 생성하는 실습으로 확장할 수 있습니다.")
