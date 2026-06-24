from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import fitz
import tempfile
import re

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_pdf_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Remove common page number patterns
        if re.fullmatch(r"\d+", line):
            continue

        if re.fullmatch(r"-\s*\d+\s*-", line):
            continue

        # Remove repeated whitespace
        line = re.sub(r"\s+", " ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def extract_text_from_pdf(pdf_file):
    pdf_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)

    total_pages = len(doc)
    text_list = []
    page_status = []

    for page_index in range(total_pages):
        page_number = page_index + 1

        try:
            page = doc[page_index]

            raw_text = page.get_text("text")
            cleaned_text = clean_pdf_text(raw_text)

            if cleaned_text and cleaned_text.strip():
                text_list.append(f"\n\n--- Page {page_number} ---\n{cleaned_text}")

                page_status.append({
                    "page": page_number,
                    "status": "success",
                    "raw_text_length": len(raw_text),
                    "cleaned_text_length": len(cleaned_text),
                    "message": "PyMuPDF 텍스트 추출 성공"
                })
            else:
                page_status.append({
                    "page": page_number,
                    "status": "failed",
                    "raw_text_length": len(raw_text),
                    "cleaned_text_length": 0,
                    "message": "추출 가능한 텍스트 없음"
                })

        except Exception as e:
            page_status.append({
                "page": page_number,
                "status": "failed",
                "raw_text_length": 0,
                "cleaned_text_length": 0,
                "message": str(e)
            })

    doc.close()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    full_text = "\n".join(text_list).strip()

    success_pages = len([item for item in page_status if item["status"] == "success"])
    failed_pages = total_pages - success_pages

    if not full_text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF일 가능성이 있습니다.")

    return {
        "text": full_text,
        "total_pages": total_pages,
        "success_pages": success_pages,
        "failed_pages": failed_pages,
        "page_status": page_status,
        "text_length": len(full_text)
    }

def summarize_pdf_text(file_name, pdf_text, total_pages, success_pages, failed_pages, max_chars=12000):
    original_length = len(pdf_text)
    used_text = pdf_text

    is_truncated = False

    if len(used_text) > max_chars:
        used_text = used_text[:max_chars]
        is_truncated = True

    prompt = f"""
당신은 제조/판매 회사의 AI 문서 분석가입니다.

아래 PDF 문서 내용을 분석하고, 실무자가 바로 볼 수 있는 요약 보고서를 작성하세요.

주의:
- 문서에 없는 내용은 추측하지 마세요.
- 불확실한 내용은 "문서에서 확인되지 않음"이라고 표시하세요.
- 긴 문서라 일부만 분석한 경우 그 한계를 명확히 표시하세요.
- 실무 액션이 있으면 별도로 정리하세요.
- 문서가 연구논문이면 연구 목적, 방법, 주요 결과, 시사점을 구분하세요.
- 회사 내부 문서라면 업무 적용 포인트와 담당 부서 후보를 중심으로 정리하세요.

[파일명]
{file_name}

[문서 정보]
PDF 처리 엔진: PyMuPDF
전체 페이지 수: {total_pages}
텍스트 추출 성공 페이지 수: {success_pages}
텍스트 추출 실패 페이지 수: {failed_pages}
원본 문자 수: {original_length}
분석 사용 문자 수: {len(used_text)}
일부만 분석했는가: {is_truncated}

[PDF 텍스트]
{used_text}

아래 형식으로 작성하세요.

# PDF 문서 요약 보고서

## 1. 문서 개요
- 파일명:
- 문서 주제:
- 문서 목적:
- 주요 대상:
- 분석 한계:

## 2. 핵심 요약
핵심 내용을 5개 이내 bullet로 요약하세요.

## 3. 주요 내용 상세
문서의 중요한 내용을 항목별로 정리하세요.

## 4. 실무 적용 포인트
이 문서를 실제 회사 업무에 적용한다면 무엇을 해야 하는지 정리하세요.

## 5. 확인 필요 사항
문서만으로 부족하거나 담당자 확인이 필요한 내용을 정리하세요.

## 6. 경영진 보고용 한 줄 요약
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        "report": response.output_text,
        "original_length": original_length,
        "used_length": len(used_text),
        "is_truncated": is_truncated
    }

def create_integrated_report(document_results):
    summary_source = ""

    for index, item in enumerate(document_results, start=1):
        summary_source += f"""

==============================
문서 {index}
파일명: {item["file_name"]}
전체 페이지 수: {item["total_pages"]}
추출 성공 페이지 수: {item["success_pages"]}
추출 실패 페이지 수: {item["failed_pages"]}
분석 일부 제한 여부: {item["is_truncated"]}

문서별 요약:
{item["report"][:5000]}
"""

    prompt = f"""
당신은 제조/판매 회사의 AI 문서 분석가입니다.

아래 여러 PDF 문서의 요약 결과를 바탕으로 통합 요약 보고서를 작성하세요.

주의:
- 각 문서 요약에 없는 내용은 새로 만들지 마세요.
- 문서 간 공통점과 차이점을 구분하세요.
- 회사 실무에 적용할 수 있는 액션을 정리하세요.
- 중요한데 문서만으로 부족한 내용은 확인 필요 사항으로 분리하세요.

[문서별 요약 자료]
{summary_source}

아래 형식으로 작성하세요.

# 통합 PDF 문서 요약 보고서

## 1. 전체 문서 개요
- 분석 문서 수:
- 전체 주제:
- 주요 대상:
- 종합 판단:

## 2. 문서별 핵심 요약
각 문서별 핵심 내용을 요약하세요.

## 3. 공통 핵심 내용
여러 문서에서 반복되거나 연결되는 핵심 내용을 정리하세요.

## 4. 문서 간 차이점
문서별 관점, 목적, 내용 차이를 정리하세요.

## 5. 실무 적용 우선순위
- 1순위:
- 2순위:
- 3순위:

## 6. 확인 필요 사항
추가 확인이 필요한 내용과 담당자 확인이 필요한 내용을 정리하세요.

## 7. 경영진 보고용 한 줄 요약
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

def save_text_report(report, prefix):
    os.makedirs("reports", exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = f"reports/{prefix}_{now}.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    return report_path
