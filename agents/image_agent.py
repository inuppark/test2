from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import base64

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_image(image_bytes, file_name, analysis_type, context_text="", mime_type="image/jpeg"):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
당신은 제조/판매 회사의 이미지 분석 AI Agent입니다.

업로드된 이미지와 사용자가 입력한 상황 설명을 함께 분석하세요.

[파일명]
{file_name}

[분석 목적]
{analysis_type}

[사용자 입력 상황 설명 또는 고객 VOC]
{context_text if context_text.strip() else "입력 없음"}

분석 원칙:
1. 이미지에서 보이는 것과 보이지 않는 것을 구분하세요.
2. 확정할 수 없는 내용은 "이미지만으로 확정 불가"라고 표시하세요.
3. 고객 VOC가 있으면 이미지 내용과 고객 주장 사이의 일치 여부를 판단하세요.
4. 제조/판매 회사의 CS, 품질관리, 마케팅, 물류 관점에서 실무적으로 작성하세요.
5. 과도한 단정은 피하고, 확인 필요 사항을 분리하세요.

아래 형식으로 보고서를 작성하세요.

# 이미지 분석 보고서

## 1. 이미지 개요
- 파일명:
- 분석 목적:
- 이미지 유형:
- 주요 피사체:
- 전체 상황:

## 2. 이미지에서 확인되는 사실
이미지에서 직접 확인 가능한 내용을 bullet로 정리하세요.

## 3. 고객/VOC 내용과의 비교
- 고객 주장 또는 상황 설명:
- 이미지와 일치하는 부분:
- 이미지로 확인되지 않는 부분:
- 추가 확인이 필요한 부분:

## 4. 제품 상태 및 하자 의심
- 정상으로 보이는 부분:
- 하자 또는 손상 의심 부분:
- 심각도: 낮음/보통/높음/긴급
- 판단 근거:

## 5. 실무 적용 포인트
- CS 대응 관점:
- 품질관리 관점:
- 마케팅/상세페이지 관점:
- 물류/포장 관점:

## 6. 추천 후속 액션
- 고객에게 확인할 질문:
- 내부 담당 부서 후보:
- 추가로 필요한 사진 또는 자료:
- 즉시 조치가 필요한 사항:

## 7. 경영진 보고용 한 줄 요약
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    return response.output_text

def save_image_report(report):
    os.makedirs("reports", exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = f"reports/image_analysis_report_{now}.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    return report_path
