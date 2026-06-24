import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=3000,
    thinking={
        "type": "enabled",
        "budget_tokens": 1024
    },
    system="""
    너는 앳플리 VOC 분석 전문가다.

    고객 불만 원인을 분석하고
    우선순위를 정리한다.
    답변은 실무자가 바로 볼 수 있게 간결하고 구조적으로 작성한다.
    """,
    messages=[
        {
            "role": "user",
            "content": """
최근 2주 동안 앳플리 고객 문의에서 아래 이슈가 증가했다.

1. 배송 문의 증가
2. AS 문의 증가
3. 체중계 앱 연결 오류 증가

가능한 원인을 분석하고
우선순위를 정리해줘.
"""
        }
    ]
)

for block in response.content:
    print("\n====================")

    if block.type == "thinking":
        print("THINKING")
        print(block.thinking)

    elif block.type == "text":
        print("ANSWER")
        print(block.text)
