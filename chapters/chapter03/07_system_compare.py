import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

roles = [
    {
        "name": "고객센터 매니저",
        "system": """
너는 앳플리 고객센터 매니저다.
고객 불만을 최소화하는 관점으로 답변한다.
"""
    },
    {
        "name": "마케팅 팀장",
        "system": """
너는 앳플리 마케팅 팀장이다.
제품의 장점과 고객가치를 강조한다.
"""
    },
    {
        "name": "CEO",
        "system": """
너는 앳플리 CEO다.
사업성과 성장 관점에서 답변한다.
"""
    }
]

for role in roles:

    print("\n")
    print("=" * 60)
    print(role["name"])
    print("=" * 60)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=role["system"],
        messages=[
            {
                "role": "user",
                "content": """
최근 체중계 리뷰 수가 감소하고 있다.
어떻게 대응해야 할까?
"""
            }
        ]
    )

    print(response.content[0].text)
