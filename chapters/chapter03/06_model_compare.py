import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

models = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5"
]

for model_name in models:

    print("\n")
    print("=" * 60)
    print(f"MODEL : {model_name}")
    print("=" * 60)

    response = client.messages.create(
        model=model_name,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": """
앳플리 체중계의 강점을
3가지 관점에서 설명해주세요.

1. 고객 관점
2. 건강관리 관점
3. 데이터 활용 관점
"""
            }
        ]
    )

    print(response.content[0].text)
