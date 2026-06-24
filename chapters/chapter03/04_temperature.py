import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

for temp in [0, 0.7, 1]:
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        temperature=temp,
        messages=[
            {
                "role": "user",
                "content": "앳플리 체중계를 소개하는 짧은 마케팅 문구를 3개 만들어줘.",
            }
        ],
    )

    print(f"\n===== temperature={temp} =====\n")
    print(message.content[0].text)