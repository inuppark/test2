from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import json
import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CATEGORIES = [
    "배송",
    "제품품질",
    "앱/소프트웨어",
    "제품기능",
    "고객서비스",
    "교환/환불",
    "사용안내",
    "가격/프로모션",
    "기타"
]

SEVERITIES = [
    "낮음",
    "보통",
    "높음",
    "긴급"
]

DEPARTMENTS = [
    "물류팀",
    "품질관리팀",
    "고객서비스팀",
    "제품개발팀",
    "앱개발팀",
    "마케팅팀",
    "운영팀",
    "기타"
]

def extract_json_array(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(f"JSON 배열을 찾을 수 없습니다:\n{text}")

    return text[start:end + 1]

def classify_voc_items(voc_list):
    voc_text = "\n".join([f"{i+1}. {voc}" for i, voc in enumerate(voc_list)])

    prompt = f"""
당신은 제조/판매 회사의 VOC 분석 전문가입니다.

아래 VOC 목록을 각각 분석하세요.

각 VOC마다 반드시 아래 항목을 작성하세요.
1. 대표 카테고리 1개
2. 심각도 1개
3. 담당 부서 후보 1개
4. 후속 액션 1개

사용 가능한 카테고리는 반드시 아래 목록 중 하나만 선택하세요.
{CATEGORIES}

사용 가능한 심각도는 반드시 아래 목록 중 하나만 선택하세요.
{SEVERITIES}

사용 가능한 담당 부서는 반드시 아래 목록 중 하나만 선택하세요.
{DEPARTMENTS}

심각도 판단 기준:
- 낮음: 단순 문의, 정보 요청, 큰 불만 없음
- 보통: 일반 불만, 사용 불편, 단순 교환/환불 요청
- 높음: 제품 품질, 배송 지연, 앱 오류, 반복 가능성이 있는 문제
- 긴급: 안전 문제, 법적 리스크, 대량 클레임 가능성, 브랜드 신뢰 훼손 가능성

중요 규칙:
1. VOC 1건당 카테고리는 반드시 1개만 선택합니다.
2. VOC 1건당 심각도는 반드시 1개만 선택합니다.
3. VOC 1건당 담당 부서는 반드시 1개만 선택합니다.
4. 전체 VOC 개수와 출력 JSON 개수는 반드시 같아야 합니다.
5. JSON 배열만 출력하세요.
6. 설명 문장, 마크다운, 코드블록은 출력하지 마세요.

출력 형식:
[
  {{
    "id": 1,
    "voc": "고객문의",
    "category": "배송",
    "severity": "높음",
    "department": "물류팀",
    "action": "배송 상태 확인 후 고객에게 지연 사유와 예상 도착일 안내"
  }}
]

[VOC 목록]
{voc_text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = response.output_text.strip()
    json_text = extract_json_array(text)

    try:
        result = json.loads(json_text)
    except json.JSONDecodeError:
        raise ValueError(f"GPT가 JSON 형식으로 답하지 않았습니다:\n{text}")

    if len(result) != len(voc_list):
        raise ValueError(
            f"분류 결과 개수가 VOC 개수와 다릅니다. VOC={len(voc_list)}, 결과={len(result)}"
        )

    cleaned = []

    for index, item in enumerate(result):
        category = item.get("category", "기타")
        severity = item.get("severity", "보통")
        department = item.get("department", "기타")
        action = item.get("action", "담당자 확인 후 고객에게 안내")

        if category not in CATEGORIES:
            category = "기타"

        if severity not in SEVERITIES:
            severity = "보통"

        if department not in DEPARTMENTS:
            department = "기타"

        cleaned.append({
            "id": index + 1,
            "voc": voc_list[index],
            "category": category,
            "severity": severity,
            "department": department,
            "action": action
        })

    return cleaned

def build_summary_table(classified_items):
    df = pd.DataFrame(classified_items)

    total = len(df)

    category_summary = (
        df["category"]
        .value_counts()
        .reindex(CATEGORIES, fill_value=0)
        .reset_index()
    )
    category_summary.columns = ["category", "count"]
    category_summary["ratio"] = (category_summary["count"] / total * 100).round(1)

    severity_summary = (
        df["severity"]
        .value_counts()
        .reindex(SEVERITIES, fill_value=0)
        .reset_index()
    )
    severity_summary.columns = ["severity", "count"]
    severity_summary["ratio"] = (severity_summary["count"] / total * 100).round(1)

    department_summary = (
        df["department"]
        .value_counts()
        .reindex(DEPARTMENTS, fill_value=0)
        .reset_index()
    )
    department_summary.columns = ["department", "count"]
    department_summary["ratio"] = (department_summary["count"] / total * 100).round(1)

    return df, category_summary, severity_summary, department_summary, total

def create_voc_report(classified_df, category_summary, severity_summary, department_summary, total):
    classified_text = classified_df.to_string(index=False)
    category_summary_text = category_summary.to_string(index=False)
    severity_summary_text = severity_summary.to_string(index=False)
    department_summary_text = department_summary.to_string(index=False)

    prompt = f"""
당신은 제조/판매 회사의 VOC 분석 전문가입니다.

아래는 Python으로 정확히 계산된 VOC 분석 결과입니다.
숫자와 비율은 절대 변경하지 말고, 그대로 사용하세요.

[전체 VOC 건수]
{total}

[카테고리별 집계]
{category_summary_text}

[심각도별 집계]
{severity_summary_text}

[담당부서별 집계]
{department_summary_text}

[개별 VOC 분석 결과]
{classified_text}

아래 형식으로 경영진 보고서를 작성하세요.

# VOC 분석 보고서

## 1. 전체 요약
- 전체 VOC 건수:
- 핵심 이슈:
- 가장 시급한 개선 영역:

## 2. 카테고리별 분류
카테고리별 문의 수와 비율을 표 형태로 정리하세요.

## 3. 심각도 분석
심각도별 문의 수와 비율을 정리하고, 높음/긴급 VOC가 있다면 주요 내용을 요약하세요.

## 4. 담당 부서별 조치 필요 항목
부서별로 어떤 조치가 필요한지 정리하세요.

## 5. 가장 시급한 문제
- 문제:
- 이유:
- 고객 영향:

## 6. 개선 제안
- 단기 개선:
- 중기 개선:
- 담당 부서 후보:

## 7. 경영진 보고용 한 줄 요약
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

def analyze_voc_text(voc_data):
    voc_list = [
        line.strip()
        for line in voc_data.splitlines()
        if line.strip()
    ]

    if len(voc_list) == 0:
        raise ValueError("분석할 VOC 데이터가 없습니다.")

    classified_items = classify_voc_items(voc_list)

    (
        classified_df,
        category_summary,
        severity_summary,
        department_summary,
        total
    ) = build_summary_table(classified_items)

    report = create_voc_report(
        classified_df,
        category_summary,
        severity_summary,
        department_summary,
        total
    )

    return report, classified_df, category_summary, severity_summary, department_summary

def save_report(report):
    os.makedirs("reports", exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = f"reports/voc_report_{today}.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    return report_path

def run_voc_analysis():
    with open("data/voc_sample.txt", "r", encoding="utf-8") as file:
        voc_data = file.read()

    (
        report,
        classified_df,
        category_summary,
        severity_summary,
        department_summary
    ) = analyze_voc_text(voc_data)

    report_path = save_report(report)

    print(report)
    print(f"\n보고서 저장 완료: {report_path}")

if __name__ == "__main__":
    run_voc_analysis()
