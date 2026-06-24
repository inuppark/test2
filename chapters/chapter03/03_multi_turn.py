import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[
        {
            "role":"user",
            "content":"내 이름은 박인업이야."
        },
        {
            "role":"assistant",
            "content":"안녕하세요 박인업님. 기억하겠습니다."
        },
        {
            "role":"user",
            "content":"내 이름이 뭐야?"
        }
    ]
)

print(message.content[0].text)