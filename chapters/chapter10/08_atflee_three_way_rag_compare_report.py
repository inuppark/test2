"""
Chapter 10 선택 실습 10-10: 앳플리 3-way RAG 품질 비교 리포트

키워드 검색 / TF-IDF 벡터 검색 / Upstage Embedding 검색
세 방식을 같은 테스트 질문으로 비교하고 hit rate를 계산한다.

사전 준비:
  1. .env에 UPSTAGE_API_KEY 설정
  2. python chapters/chapter10/07_atflee_upstage_embedding_practice.py 실행
     → data/rag/atflee_upstage_embedding_index.json 생성
"""

import os
import sys
import json
import math
from datetime import datetime

from dotenv import load_dotenv
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.rag_utils import search_wiki
from utils.vector_rag_utils import search_similar_chunks

REPORT_DIR         = os.path.join(PROJECT_ROOT, "reports", "chapter10")
UPSTAGE_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "rag", "atflee_upstage_embedding_index.json")

UPSTAGE_EMBEDDING_URL  = "https://api.upstage.ai/v1/solar/embeddings"
UPSTAGE_MODEL_QUERY    = "solar-embedding-1-large-query"    # 질문(query) 임베딩용
UPSTAGE_MODEL_PASSAGE  = "solar-embedding-1-large-passage"  # 문서/청크(passage) 임베딩용

# ==============================
# 테스트 케이스
# ==============================
test_cases = [
    {
        "question": "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
        "expected_files": ["atflee_app_guide.md", "atflee_contact_guide.md"]
    },
    {
        "question": "블루투스 페어링이 계속 실패해요.",
        "expected_files": ["atflee_app_guide.md", "atflee_contact_guide.md"]
    },
    {
        "question": "배송은 보통 얼마나 걸려?",
        "expected_files": ["logistics_policy.md", "atflee_delivery_refund_policy.md"]
    },
    {
        "question": "환불은 언제 처리돼?",
        "expected_files": ["atflee_delivery_refund_policy.md"]
    },
    {
        "question": "T9은 어떤 제품이야?",
        "expected_files": ["atflee_product_guide.md"]
    },
    {
        "question": "AS 접수됐는지 확인해줘.",
        "expected_files": ["atflee_contact_guide.md", "customer_service_policy.md", "product_quality_policy.md"]
    },
    {
        "question": "고객센터 전화번호 알려줘.",
        "expected_files": ["atflee_contact_guide.md"]
    },
    {
        "question": "제품이 불량 같고 교환하고 싶어요.",
        "expected_files": ["product_quality_policy.md", "atflee_delivery_refund_policy.md"]
    }
]


# ==============================
# 함수 1: Upstage API Key 로드
# ==============================
def load_upstage_api_key():
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
# 함수 2: Upstage 임베딩 인덱스 로드
# ==============================
def load_upstage_index():
    """
    data/rag/atflee_upstage_embedding_index.json을 읽는다.
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
def call_upstage_embedding(text, api_key, model=None):
    """
    Upstage Embedding API로 text의 임베딩 벡터를 생성한다.
    model 미지정 시 query 모델 사용. 청크는 passage 모델을 전달한다.
    API Key는 헤더에만 사용하고 절대 출력하지 않는다.
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
            f"status={response.status_code}, body={response.text[:300]}"
        )

    data = response.json()

    try:
        return data["data"][0]["embedding"]
    except (KeyError, IndexError):
        raise RuntimeError(f"예상하지 못한 응답 형식입니다: {str(data)[:300]}")


# ==============================
# 함수 4: 코사인 유사도
# ==============================
def cosine_similarity(vector_a, vector_b):
    """두 dense 벡터의 코사인 유사도를 계산한다."""
    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ==============================
# 함수 5: 키워드 검색 결과 정규화
# ==============================
def get_keyword_results(question, top_k=3):
    """Chapter 7 키워드 검색 결과를 표준 형태로 변환한다."""
    results = search_wiki(question, top_k=top_k)

    return [
        {
            "source_file": r.get("file_name"),
            "score":       r.get("score"),
            "snippet":     r.get("snippet", "")
        }
        for r in results
    ]


# ==============================
# 함수 6: TF-IDF 벡터 검색 결과 정규화
# ==============================
def get_tfidf_results(question, top_k=3):
    """Chapter 10 TF-IDF 벡터 검색 결과를 표준 형태로 변환한다."""
    payload = search_similar_chunks(question, top_k=top_k)

    if payload.get("error"):
        return []

    return [
        {
            "source_file":  r.get("source_file"),
            "chunk_id":     r.get("chunk_id"),
            "similarity":   r.get("similarity"),
            "text_preview": r.get("text", "")[:300]
        }
        for r in payload.get("results", [])
    ]


# ==============================
# 함수 7: Upstage Embedding 검색 결과
# ==============================
def get_upstage_results(question, upstage_index, api_key, top_k=3):
    """
    질문을 Upstage Embedding으로 변환한 뒤
    인덱스 청크와 코사인 유사도를 계산해 TOP K를 반환한다.
    """
    query_embedding = call_upstage_embedding(question, api_key, model=UPSTAGE_MODEL_QUERY)

    scored = []
    for chunk in upstage_index.get("chunks", []):
        sim = cosine_similarity(query_embedding, chunk.get("embedding", []))
        scored.append(
            {
                "source_file":  chunk["source_file"],
                "chunk_id":     chunk["chunk_id"],
                "similarity":   sim,
                "text_preview": chunk.get("text", "")[:300]
            }
        )

    scored.sort(key=lambda x: -x["similarity"])
    return scored[:top_k]


# ==============================
# 함수 8: hit 평가
# ==============================
def evaluate_hits(results, expected_files):
    """
    TOP 1이 expected_files에 있으면 top1_hit = True
    TOP 3 안에 expected_files 중 하나라도 있으면 top3_hit = True
    """
    if not results:
        return {"top1_hit": False, "top3_hit": False, "top1_file": None, "top3_files": []}

    top1_file  = results[0].get("source_file")
    top3_files = [r.get("source_file") for r in results[:3]]

    return {
        "top1_hit":   top1_file in expected_files,
        "top3_hit":   any(f in expected_files for f in top3_files),
        "top1_file":  top1_file,
        "top3_files": top3_files
    }


# ==============================
# 함수 9: 질문 하나 3-way 비교
# ==============================
def compare_one_case(test_case, upstage_index, api_key):
    """질문 하나에 대해 세 방식의 검색 결과와 평가를 반환한다."""
    question       = test_case["question"]
    expected_files = test_case["expected_files"]

    keyword_results = get_keyword_results(question)
    tfidf_results   = get_tfidf_results(question)
    upstage_results = get_upstage_results(question, upstage_index, api_key)

    return {
        "question":       question,
        "expected_files": expected_files,
        "keyword": {
            "results":    keyword_results,
            "evaluation": evaluate_hits(keyword_results, expected_files)
        },
        "tfidf": {
            "results":    tfidf_results,
            "evaluation": evaluate_hits(tfidf_results, expected_files)
        },
        "upstage": {
            "results":    upstage_results,
            "evaluation": evaluate_hits(upstage_results, expected_files)
        }
    }


# ==============================
# 함수 10: 전체 hit rate 계산
# ==============================
def calculate_summary(case_results):
    """세 방식의 TOP1 / TOP3 hit rate를 계산한다."""
    total = len(case_results)
    if total == 0:
        return {}

    def count(method, key):
        return sum(1 for c in case_results if c[method]["evaluation"][key])

    kw_t1 = count("keyword", "top1_hit")
    kw_t3 = count("keyword", "top3_hit")
    tf_t1 = count("tfidf",   "top1_hit")
    tf_t3 = count("tfidf",   "top3_hit")
    up_t1 = count("upstage", "top1_hit")
    up_t3 = count("upstage", "top3_hit")

    return {
        "total_cases":            total,
        "keyword_top1_hits":      kw_t1,
        "keyword_top3_hits":      kw_t3,
        "tfidf_top1_hits":        tf_t1,
        "tfidf_top3_hits":        tf_t3,
        "upstage_top1_hits":      up_t1,
        "upstage_top3_hits":      up_t3,
        "keyword_top1_hit_rate":  round(kw_t1 / total, 3),
        "keyword_top3_hit_rate":  round(kw_t3 / total, 3),
        "tfidf_top1_hit_rate":    round(tf_t1 / total, 3),
        "tfidf_top3_hit_rate":    round(tf_t3 / total, 3),
        "upstage_top1_hit_rate":  round(up_t1 / total, 3),
        "upstage_top3_hit_rate":  round(up_t3 / total, 3),
    }


# ==============================
# 함수 11: 질문별 콘솔 출력
# ==============================
def print_case_results(case_results):
    """질문별 세 방식의 TOP 1 파일과 hit 여부를 출력한다."""
    print("\n[키워드 / TF-IDF / Upstage Embedding 3-way 비교 결과]")

    for idx, case in enumerate(case_results, start=1):
        print("\n" + "=" * 80)
        print(f"{idx}. 질문: {case['question']}")
        print(f"   기대 파일: {case['expected_files']}")

        for method, label in [("keyword", "키워드  "), ("tfidf", "TF-IDF  "), ("upstage", "Upstage ")]:
            ev = case[method]["evaluation"]
            hit1 = "O" if ev["top1_hit"] else "X"
            hit3 = "O" if ev["top3_hit"] else "X"
            print(f"   [{label}] TOP1: {ev['top1_file']:<40} TOP1={hit1}  TOP3={hit3}")


# ==============================
# 함수 12: 전체 요약 콘솔 출력
# ==============================
def print_summary(summary):
    """세 방식의 hit rate 표와 간단한 해석을 출력한다."""
    total = summary["total_cases"]

    print("\n" + "=" * 80)
    print("[전체 요약]")
    print(f"{'방식':<12}{'TOP1 hit':>10}{'TOP1 rate':>12}{'TOP3 hit':>10}{'TOP3 rate':>12}")
    print("-" * 58)

    for method, label in [
        ("keyword", "키워드"),
        ("tfidf",   "TF-IDF"),
        ("upstage", "Upstage"),
    ]:
        t1 = summary[f"{method}_top1_hits"]
        t3 = summary[f"{method}_top3_hits"]
        r1 = summary[f"{method}_top1_hit_rate"]
        r3 = summary[f"{method}_top3_hit_rate"]
        print(f"  {label:<10}{t1:>4}/{total:<4}   {r1:>6}    {t3:>4}/{total:<4}   {r3:>6}")

    print("\n[해석]")
    rates = {
        "키워드":   summary["keyword_top1_hit_rate"],
        "TF-IDF":   summary["tfidf_top1_hit_rate"],
        "Upstage":  summary["upstage_top1_hit_rate"],
    }
    best = max(rates, key=rates.get)
    print(f"- TOP 1 hit rate 기준 가장 높은 방식: {best} ({rates[best]})")
    print("- 키워드 검색은 명확한 단어가 있는 질문에 강합니다.")
    print("- TF-IDF는 단어 가중치 기반으로 키워드와 유사한 수준입니다.")
    print("- Upstage Embedding은 의미 기반 검색으로, 표현이 다른 질문에서 강점을 보일 수 있습니다.")
    print("- 복합 의도 질문('제품 불량+교환')은 세 방식 모두 보강이 필요합니다.")


# ==============================
# 함수 13: JSON 리포트 저장
# ==============================
def save_report(case_results, summary):
    """비교 결과를 reports/chapter10 폴더에 JSON으로 저장한다."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    created_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "created_at":  created_at,
        "chapter":     "chapter10",
        "report_name": "atflee_keyword_tfidf_upstage_rag_compare",
        "summary":     summary,
        "cases":       case_results,
        "notes": [
            "키워드 검색은 단어 일치 기반이다.",
            "TF-IDF는 단어 가중치 기반 벡터 검색이다.",
            "Upstage Embedding은 외부 임베딩 모델 기반 semantic search에 가깝다.",
            "현재 평가는 expected_files 기준의 간단한 hit rate다."
        ]
    }

    file_name = f"atflee_three_way_rag_compare_{file_timestamp}.json"
    file_path = os.path.join(REPORT_DIR, file_name)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return file_path


# ==============================
# 함수 14: reports gitignore 확인
# ==============================
def check_reports_gitignore():
    """reports/ 폴더가 .gitignore에 포함되어 있는지 확인한다."""
    gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")

    if not os.path.exists(gitignore_path):
        return False

    with open(gitignore_path, "r", encoding="utf-8") as file:
        content = file.read()

    return "reports/" in content or "reports" in content


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    print("[Chapter 10 선택 실습 10-10] 앳플리 3-way RAG 품질 비교")
    print(f"테스트 질문 수: {len(test_cases)}")
    print("-" * 60)

    # API Key 로드 — 값은 출력하지 않는다.
    api_key = load_upstage_api_key()
    print("Upstage API Key 로드 완료 (값은 보안상 출력하지 않습니다.)")

    # Upstage 임베딩 인덱스 로드
    upstage_index = load_upstage_index()

    if not upstage_index:
        print("\nUpstage 임베딩 인덱스가 없습니다.")
        print("먼저 아래 명령어를 실행하세요.")
        print("  python chapters/chapter10/07_atflee_upstage_embedding_practice.py")
    else:
        print(f"Upstage 인덱스 로드 완료 / 청크 수: {upstage_index.get('chunk_count')}, "
              f"임베딩 차원: {upstage_index.get('embedding_dimension')}")

        # 질문별 3-way 비교
        case_results = []
        for test_case in test_cases:
            print(f"  처리 중: {test_case['question'][:35]}...")
            result = compare_one_case(test_case, upstage_index, api_key)
            case_results.append(result)

        # 전체 요약 계산
        summary = calculate_summary(case_results)

        # 콘솔 출력
        print_case_results(case_results)
        print_summary(summary)

        # JSON 리포트 저장
        saved_path = save_report(case_results, summary)

        print("\n[리포트 저장 완료]")
        print(saved_path)

        # 보안 확인
        if check_reports_gitignore():
            print("\n[보안 확인] reports 폴더가 .gitignore에 포함되어 있습니다.")
        else:
            print("\n[주의] reports 폴더가 .gitignore에 포함되어 있는지 확인이 필요합니다.")
