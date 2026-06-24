import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

system_prompt = """
너는 앳플리 VOC 분석 전문가다.
고객 문의를 분석하고, 원인과 우선순위를 실무자가 바로 이해할 수 있게 정리한다.
답변은 항상 아래 형식으로 작성한다.

1. 핵심 요약
2. 원인 분석
3. 우선순위
4. 다음 액션
"""

messages = [
    {
        "role": "user",
        "content": "최근 2주 동안 체중계 앱 연결 오류 문의가 늘었어."
    },
    {
        "role": "assistant",
        "content": "앱 연결 오류 문의 증가를 확인했습니다. 원인 후보와 우선순위를 분석해보겠습니다."
    },
    {
        "role": "user",
        "content": "그럼 이 문제를 어떤 순서로 확인해야 해?"
    }
]

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1000,
    system=system_prompt,
    messages=messages
)

print(response.content[0].text)
