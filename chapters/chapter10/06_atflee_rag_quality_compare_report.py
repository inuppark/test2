"""
Chapter 10-8: 키워드 RAG vs 벡터 RAG 검색 품질 비교 리포트

Chapter 7의 키워드 검색(search_wiki)과
Chapter 10의 TF-IDF 벡터 검색(search_similar_chunks)을 같은 질문으로 비교하여
어떤 방식이 어떤 유형의 질문에 더 잘 맞는지 평가한다.

결과는 콘솔 출력 + reports/chapter10/JSON 파일로 저장된다.
"""

import os
import sys
import json
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가해 utils 모듈을 임포트할 수 있게 한다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.rag_utils import search_wiki
from utils.vector_rag_utils import search_similar_chunks

# 리포트 저장 폴더
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "chapter10")

# ==============================
# 테스트 질문 목록
# expected_files: 이 질문에 대해 검색 결과 상위에 나와야 하는 파일명
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
# 함수 1: 키워드 검색 결과 정규화
# ==============================
def get_keyword_results(question, top_k=3):
    """
    Chapter 7 키워드 검색 결과를 표준 형태로 변환한다.
    search_wiki는 {"file_name", "score", "snippet", ...} 형태를 반환한다.
    """
    results = search_wiki(question, top_k=top_k)

    normalized = []
    for result in results:
        normalized.append(
            {
                "source_file": result.get("file_name"),
                "score": result.get("score"),
                "snippet": result.get("snippet", "")
            }
        )

    return normalized


# ==============================
# 함수 2: 벡터 검색 결과 정규화
# ==============================
def get_vector_results(question, top_k=3):
    """
    Chapter 10 TF-IDF 벡터 검색 결과를 표준 형태로 변환한다.
    search_similar_chunks는 {"results": [...], "error": ...} 형태를 반환한다.
    """
    payload = search_similar_chunks(question, top_k=top_k)

    # 인덱스 파일이 없거나 오류가 있을 때는 빈 목록 반환
    if payload.get("error"):
        return []

    results = payload.get("results", [])

    normalized = []
    for result in results:
        normalized.append(
            {
                "source_file": result.get("source_file"),
                "chunk_id": result.get("chunk_id"),
                "similarity": result.get("similarity"),
                # 미리보기는 300자로 제한해 리포트 가독성을 높인다.
                "text_preview": result.get("text", "")[:300]
            }
        )

    return normalized


# ==============================
# 함수 3: 검색 결과 hit 평가
# ==============================
def evaluate_hits(results, expected_files):
    """
    검색 결과 TOP 1과 TOP 3가 expected_files를 맞췄는지 평가한다.

    top1_hit: TOP 1 파일이 expected_files 안에 있으면 True
    top3_hit: TOP 3 중 하나라도 expected_files 안에 있으면 True
    """
    if not results:
        return {
            "top1_hit": False,
            "top3_hit": False,
            "top1_file": None,
            "top3_files": []
        }

    top1_file = results[0].get("source_file")
    top3_files = [result.get("source_file") for result in results[:3]]

    top1_hit = top1_file in expected_files
    top3_hit = any(file_name in expected_files for file_name in top3_files)

    return {
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "top1_file": top1_file,
        "top3_files": top3_files
    }


# ==============================
# 함수 4: 질문 하나 비교
# ==============================
def compare_one_case(test_case):
    """
    질문 하나에 대해 키워드 검색과 벡터 검색을 모두 실행하고 결과를 합친다.
    """
    question = test_case["question"]
    expected_files = test_case["expected_files"]

    keyword_results = get_keyword_results(question)
    vector_results = get_vector_results(question)

    keyword_eval = evaluate_hits(keyword_results, expected_files)
    vector_eval = evaluate_hits(vector_results, expected_files)

    return {
        "question": question,
        "expected_files": expected_files,
        "keyword": {
            "results": keyword_results,
            "evaluation": keyword_eval
        },
        "vector": {
            "results": vector_results,
            "evaluation": vector_eval
        }
    }


# ==============================
# 함수 5: 전체 hit rate 계산
# ==============================
def calculate_summary(case_results):
    """
    전체 질문에 대한 hit rate를 계산한다.
    hit rate = 맞춘 질문 수 / 전체 질문 수
    """
    total = len(case_results)

    if total == 0:
        return {}

    keyword_top1_hits = sum(1 for case in case_results if case["keyword"]["evaluation"]["top1_hit"])
    keyword_top3_hits = sum(1 for case in case_results if case["keyword"]["evaluation"]["top3_hit"])
    vector_top1_hits  = sum(1 for case in case_results if case["vector"]["evaluation"]["top1_hit"])
    vector_top3_hits  = sum(1 for case in case_results if case["vector"]["evaluation"]["top3_hit"])

    return {
        "total_cases":            total,
        "keyword_top1_hits":      keyword_top1_hits,
        "keyword_top3_hits":      keyword_top3_hits,
        "vector_top1_hits":       vector_top1_hits,
        "vector_top3_hits":       vector_top3_hits,
        "keyword_top1_hit_rate":  round(keyword_top1_hits / total, 3),
        "keyword_top3_hit_rate":  round(keyword_top3_hits / total, 3),
        "vector_top1_hit_rate":   round(vector_top1_hits  / total, 3),
        "vector_top3_hit_rate":   round(vector_top3_hits  / total, 3),
    }


# ==============================
# 함수 6: 질문별 콘솔 출력
# ==============================
def print_case_results(case_results):
    """
    질문별 비교 결과를 콘솔에 보기 좋게 출력한다.
    """
    print("\n[키워드 RAG vs 벡터 RAG 비교 결과]")

    for index, case in enumerate(case_results, start=1):
        print("\n" + "=" * 80)
        print(f"{index}. 질문: {case['question']}")
        print(f"   기대 파일: {case['expected_files']}")

        keyword_eval = case["keyword"]["evaluation"]
        vector_eval  = case["vector"]["evaluation"]

        print("\n   [키워드 검색]")
        print(f"   TOP 1: {keyword_eval['top1_file']}")
        print(f"   TOP 1 hit: {keyword_eval['top1_hit']}")
        print(f"   TOP 3 hit: {keyword_eval['top3_hit']}")
        print(f"   TOP 3 목록: {keyword_eval['top3_files']}")

        print("\n   [벡터 검색]")
        print(f"   TOP 1: {vector_eval['top1_file']}")
        print(f"   TOP 1 hit: {vector_eval['top1_hit']}")
        print(f"   TOP 3 hit: {vector_eval['top3_hit']}")
        print(f"   TOP 3 목록: {vector_eval['top3_files']}")


# ==============================
# 함수 7: 전체 요약 콘솔 출력
# ==============================
def print_summary(summary):
    """
    전체 hit rate 요약과 해석을 콘솔에 출력한다.
    """
    print("\n" + "=" * 80)
    print("[전체 요약]")
    print(f"전체 질문 수: {summary['total_cases']}")
    print(f"키워드 TOP 1 hit rate: {summary['keyword_top1_hit_rate']}  ({summary['keyword_top1_hits']}/{summary['total_cases']})")
    print(f"키워드 TOP 3 hit rate: {summary['keyword_top3_hit_rate']}  ({summary['keyword_top3_hits']}/{summary['total_cases']})")
    print(f"벡터   TOP 1 hit rate: {summary['vector_top1_hit_rate']}  ({summary['vector_top1_hits']}/{summary['total_cases']})")
    print(f"벡터   TOP 3 hit rate: {summary['vector_top3_hit_rate']}  ({summary['vector_top3_hits']}/{summary['total_cases']})")

    print("\n[해석]")
    print("- 키워드 검색은 명확한 단어가 포함된 질문에서 강합니다.")
    print("- TF-IDF 벡터 검색은 특정 단어 밀도와 유사도 기반으로 작동하지만, 한국어 형태소 분석이 없어 한계가 있습니다.")
    print("- 실제 서비스 고도화 단계에서는 임베딩 API 또는 한국어 형태소 분석기를 함께 검토하는 것이 좋습니다.")


# ==============================
# 함수 8: JSON 리포트 저장
# ==============================
def save_report(case_results, summary):
    """
    비교 결과를 reports/chapter10 폴더에 JSON으로 저장한다.
    파일명에 날짜시간을 붙여 덮어쓰기를 방지한다.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    created_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "created_at":  created_at,
        "chapter":     "chapter10",
        "report_name": "atflee_keyword_vs_vector_rag_quality_compare",
        "summary":     summary,
        "cases":       case_results,
        "notes": [
            "현재 벡터 검색은 학습용 TF-IDF 방식입니다.",
            "실제 semantic embedding과는 다릅니다.",
            "검색 품질 평가는 expected_files 기준의 간단한 hit rate로 계산했습니다."
        ]
    }

    file_name = f"atflee_rag_quality_compare_{file_timestamp}.json"
    file_path = os.path.join(REPORT_DIR, file_name)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return file_path


# ==============================
# 함수 9: reports/ gitignore 확인
# ==============================
def check_reports_gitignore():
    """
    .gitignore에 reports/ 가 포함되어 있는지 확인한다.
    리포트 파일이 실수로 GitHub에 올라가는 것을 방지하기 위한 안전 확인이다.
    """
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
    print("[Chapter 10-8] 앳플리 키워드 RAG vs 벡터 RAG 검색 품질 비교")
    print(f"테스트 질문 수: {len(test_cases)}")

    # 질문별 비교 실행
    case_results = []
    for test_case in test_cases:
        print(f"  처리 중: {test_case['question'][:30]}...")
        result = compare_one_case(test_case)
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
