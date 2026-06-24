import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

model_name = "claude-sonnet-4-5"

customer_messages = [
    "체중계가 앱이랑 계속 연결이 안 됩니다. AS 문의도 남겼는데 답변이 늦어서 너무 답답합니다.",
    "배송이 예정일보다 4일 늦어졌는데 아무 안내도 없었어요. 선물용이었는데 이미 늦었습니다.",
    "제품을 받았는데 화면이 깨져 있고 박스도 찌그러져 있었습니다. 바로 교환받고 싶습니다."
]

system_prompt = """
너는 앳플리 VOC Agent v1이다.

역할:
- 고객 문의를 읽고 이슈 유형, 심각도, 고객 감정, 원인 후보, 담당 부서, 응대 방향, 다음 액션을 분석한다.
- 실무자가 바로 처리할 수 있도록 구조화된 JSON으로만 답한다.
- 고객이 말하지 않은 사실을 확정하지 않는다.
- 원인은 반드시 가능성으로 표현한다.
- 심각도는 낮음, 중간, 높음 중 하나로 판단한다.
- 사람이 확인해야 하는 이슈는 needs_human_review를 true로 표시한다.

판단 순서:
1. 고객 감정 파악
2. 이슈 유형 분류
3. 심각도 판단
4. 원인 후보 도출
5. 담당 부서 판단
6. 고객 응대 방향 제안
7. 내부 후속 액션 제안

출력 규칙:
- 반드시 JSON만 출력한다.
- JSON 앞뒤에 설명, 마크다운, 코드블록을 붙이지 않는다.
- 아래 스키마를 지킨다.

출력 JSON 스키마:
{
  "issue_type": "이슈 유형",
  "severity": "낮음 | 중간 | 높음",
  "customer_emotion": "고객 감정",
  "possible_causes": ["원인 후보 1", "원인 후보 2", "원인 후보 3"],
  "owner_team": "담당 부서",
  "reply_direction": "고객 응대 방향",
  "next_action": "다음 액션",
  "needs_human_review": true 또는 false
}
"""

few_shot_examples = """
예시 1:
고객 문의:
배송이 예정일보다 3일 늦어졌고, 알림도 받지 못했습니다.

좋은 출력:
{
  "issue_type": "배송 지연",
  "severity": "중간",
  "customer_emotion": "불만, 불안",
  "possible_causes": ["물류 지연 가능성", "배송 알림 누락 가능성", "주문량 증가 가능성"],
  "owner_team": "물류/CS",
  "reply_direction": "배송 지연에 대해 사과하고 현재 배송 상태와 예상 도착일을 안내한다.",
  "next_action": "배송 추적 정보를 확인한 뒤 고객에게 개별 안내한다.",
  "needs_human_review": false
}

예시 2:
고객 문의:
제품을 받았는데 포장이 찢어져 있고 본체에 흠집이 있습니다.

좋은 출력:
{
  "issue_type": "제품 파손/품질",
  "severity": "높음",
  "customer_emotion": "분노, 실망",
  "possible_causes": ["배송 중 파손 가능성", "포장재 부족 가능성", "출고 검수 누락 가능성"],
  "owner_team": "품질/물류/CS",
  "reply_direction": "불편에 대해 즉시 사과하고 교환 또는 환불 절차를 안내한다.",
  "next_action": "파손 사진을 확인한 뒤 교환 접수 또는 환불 절차를 진행한다.",
  "needs_human_review": true
}
"""

def clean_json_text(text):
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned

def analyze_voc(customer_message):
    prompt = f"""
{few_shot_examples}

이제 아래 고객 문의를 같은 기준으로 분석해라.

고객 문의:
{customer_message}
"""

    response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw_text = response.content[0].text
    cleaned_text = clean_json_text(raw_text)

    return json.loads(cleaned_text)

for index, customer_message in enumerate(customer_messages, start=1):
    print("\n" + "=" * 80)
    print(f"VOC #{index}")
    print("=" * 80)
    print("고객 문의:", customer_message)

    result = analyze_voc(customer_message)

    print("이슈 유형:", result["issue_type"])
    print("심각도:", result["severity"])
    print("담당 부서:", result["owner_team"])
    print("다음 액션:", result["next_action"])
    print("담당자 확인 필요:", result["needs_human_review"])

    if result["severity"] == "높음" or result["needs_human_review"] is True:
        print("⚠️ 담당자 확인 필요")
    else:
        print("✅ 일반 처리 가능")
