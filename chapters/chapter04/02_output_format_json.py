import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

customer_message = """
체중계가 앱이랑 연결이 계속 안 돼요.
어제부터 계속 시도했는데 실패하고,
AS 문의를 남겼는데 답변도 늦어서 답답합니다.
"""

prompt = f"""
너는 앳플리 VOC 분석 전문가다.

아래 고객 문의를 분석하고 반드시 JSON 형식으로만 출력해라.
JSON 앞뒤에 설명, 마크다운, 코드블록을 붙이지 마라.

고객 문의:
{customer_message}

출력 JSON 스키마:
{{
  "issue_type": "이슈 유형",
  "severity": "낮음 | 중간 | 높음",
  "customer_emotion": "고객 감정",
  "possible_causes": ["원인 후보 1", "원인 후보 2", "원인 후보 3"],
  "owner_team": "담당 부서",
  "reply_direction": "고객 응대 방향",
  "next_action": "다음 액션"
}}
"""

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1000,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

raw_text = response.content[0].text

print("\n===== RAW CLAUDE OUTPUT =====")
print(raw_text)

clean_text = raw_text.strip()
if clean_text.startswith("```"):
    clean_text = clean_text.split("\n", 1)[1]
if clean_text.endswith("```"):
    clean_text = clean_text.rsplit("```", 1)[0]

data = json.loads(clean_text.strip())

print("\n===== PARSED RESULT =====")
print("이슈 유형:", data["issue_type"])
print("심각도:", data["severity"])
print("담당 부서:", data["owner_team"])
print("다음 액션:", data["next_action"])
