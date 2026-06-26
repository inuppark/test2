"""
utils/hybrid_rag_utils.py

키워드 검색(Chapter 7)과 Upstage Embedding 검색(Chapter 10 선택 실습)을 결합한
하이브리드 RAG 공통 유틸리티.

제공 함수:
  normalize_keyword_results  — search_wiki 결과를 표준 형태로 변환
  normalize_upstage_results  — Upstage 검색 결과를 표준 형태로 변환
  merge_hybrid_results       — 두 결과 병합 + hybrid_score 계산 + TOP K 반환
  search_hybrid_rag          — 키워드 + Upstage 검색 통합 실행
  build_hybrid_rag_context   — Claude 전달용 context 문자열 생성

점수 계산:
  keyword_weight = 0.4  (Reciprocal Rank 기반 정규화 점수)
  upstage_weight = 0.6  (Reciprocal Rank 기반 정규화 점수)
  overlap_bonus  = 0.2  (양쪽 검색 모두에서 등장한 source_file에 가산)
"""

import os
import sys

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.rag_utils import search_wiki
from utils.upstage_rag_utils import (
    load_upstage_index,
    search_upstage_chunks,
)

DEFAULT_TOP_K = 3


# ==============================
# 함수 1: 키워드 검색 결과 정규화
# ==============================
def normalize_keyword_results(keyword_results):
    """
    search_wiki 결과를 하이브리드 검색용 표준 형태로 변환한다.
    keyword 결과에는 chunk_id가 없을 수 있으므로 chunk_id는 None으로 둔다.
    """
    normalized = []

    for rank, result in enumerate(keyword_results, start=1):
        score = result.get("score", 0) or 0

        normalized.append(
            {
                "source":           "keyword",
                "rank":             rank,
                "source_file":      result.get("file_name"),
                "chunk_id":         result.get("chunk_id"),
                "text":             result.get("snippet", ""),
                "raw_score":        score,
                "normalized_score": 1 / rank,
            }
        )

    return normalized


# ==============================
# 함수 2: Upstage 검색 결과 정규화
# ==============================
def normalize_upstage_results(upstage_results):
    """
    Upstage 검색 결과를 하이브리드 검색용 표준 형태로 변환한다.
    """
    normalized = []

    for rank, result in enumerate(upstage_results, start=1):
        similarity = result.get("similarity", 0) or 0

        normalized.append(
            {
                "source":           "upstage",
                "rank":             rank,
                "source_file":      result.get("source_file"),
                "chunk_id":         result.get("chunk_id"),
                "text":             result.get("text", ""),
                "raw_score":        similarity,
                "normalized_score": 1 / rank,
            }
        )

    return normalized


# ==============================
# 함수 3: 두 결과 병합
# ==============================
def merge_hybrid_results(keyword_items, upstage_items, top_k=DEFAULT_TOP_K):
    """
    키워드 검색 결과와 Upstage 검색 결과를 합쳐 최종 TOP K를 만든다.

    점수 계산:
    - keyword_weight = 0.4, upstage_weight = 0.6 (Reciprocal Rank 기반)
    - 같은 source_file이 양쪽에 모두 등장하면 overlap_bonus = 0.2 가산
    - chunk_id가 있으면 Upstage 청크 본문을 우선 사용 (더 정밀한 단위)
    - keyword 결과는 chunk_id가 없을 수 있으므로 source_file 기준으로 병합
    """
    keyword_weight = 0.4
    upstage_weight = 0.6
    overlap_bonus  = 0.2

    merged = {}

    for item in keyword_items:
        key = item["source_file"]

        if key not in merged:
            merged[key] = {
                "source_file":   item["source_file"],
                "chunk_id":      item.get("chunk_id"),
                "text":          item.get("text", ""),
                "keyword_score": 0,
                "upstage_score": 0,
                "sources":       set(),
                "keyword_rank":  None,
                "upstage_rank":  None,
            }

        merged[key]["keyword_score"] = max(
            merged[key]["keyword_score"],
            item["normalized_score"]
        )
        merged[key]["sources"].add("keyword")
        merged[key]["keyword_rank"] = item["rank"]

    for item in upstage_items:
        key = item["source_file"]

        if key not in merged:
            merged[key] = {
                "source_file":   item["source_file"],
                "chunk_id":      item.get("chunk_id"),
                "text":          item.get("text", ""),
                "keyword_score": 0,
                "upstage_score": 0,
                "sources":       set(),
                "keyword_rank":  None,
                "upstage_rank":  None,
            }

        merged[key]["upstage_score"] = max(
            merged[key]["upstage_score"],
            item["normalized_score"]
        )
        merged[key]["sources"].add("upstage")
        merged[key]["upstage_rank"] = item["rank"]

        # Upstage 청크 본문이 더 정밀하므로 우선 사용
        if item.get("text"):
            merged[key]["text"]     = item["text"]
            merged[key]["chunk_id"] = item.get("chunk_id")

    final_results = []

    for item in merged.values():
        hybrid_score = (
            item["keyword_score"] * keyword_weight
            + item["upstage_score"] * upstage_weight
        )

        if "keyword" in item["sources"] and "upstage" in item["sources"]:
            hybrid_score += overlap_bonus

        final_results.append(
            {
                "source_file":   item["source_file"],
                "chunk_id":      item.get("chunk_id"),
                "text":          item.get("text", ""),
                "keyword_score": item["keyword_score"],
                "upstage_score": item["upstage_score"],
                "hybrid_score":  hybrid_score,
                "sources":       sorted(list(item["sources"])),
                "keyword_rank":  item["keyword_rank"],
                "upstage_rank":  item["upstage_rank"],
            }
        )

    final_results.sort(key=lambda item: -item["hybrid_score"])

    return final_results[:top_k]


# ==============================
# 함수 4: 하이브리드 RAG 통합 검색
# ==============================
def search_hybrid_rag(question, upstage_api_key, top_k=DEFAULT_TOP_K):
    """
    질문에 대해 키워드 검색 + Upstage 검색을 함께 수행하고,
    최종 하이브리드 TOP K 결과를 반환한다.

    반환 형태:
      {
        "keyword_results": [...],
        "upstage_results": [...],
        "hybrid_results":  [...],
        "error": None 또는 오류 메시지
      }
    """
    keyword_raw    = search_wiki(question, top_k=top_k)
    keyword_items  = normalize_keyword_results(keyword_raw)

    upstage_payload = search_upstage_chunks(
        question=question,
        api_key=upstage_api_key,
        top_k=top_k
    )

    if upstage_payload.get("error"):
        return {
            "keyword_results": keyword_items,
            "upstage_results": [],
            "hybrid_results":  keyword_items[:top_k],
            "error":           upstage_payload.get("error")
        }

    upstage_items = normalize_upstage_results(upstage_payload.get("results", []))

    hybrid_results = merge_hybrid_results(
        keyword_items=keyword_items,
        upstage_items=upstage_items,
        top_k=top_k
    )

    return {
        "keyword_results": keyword_items,
        "upstage_results": upstage_items,
        "hybrid_results":  hybrid_results,
        "error":           None
    }


# ==============================
# 함수 5: Claude 전달용 context 생성
# ==============================
def build_hybrid_rag_context(hybrid_results):
    """
    Claude에게 전달할 하이브리드 RAG context를 만든다.
    """
    context_parts = []

    for result in hybrid_results:
        context_parts.append(
            f"[출처: {result['source_file']} / "
            f"chunk_id: {result.get('chunk_id')} / "
            f"hybrid_score: {result['hybrid_score']:.4f} / "
            f"sources: {', '.join(result['sources'])}]\n"
            f"{result.get('text', '')}"
        )

    return "\n\n---\n\n".join(context_parts)
