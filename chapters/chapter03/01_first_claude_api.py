import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(
api_key=os.getenv("ANTHROPIC_API_KEY")
)

response = client.messages.create(
model="claude-sonnet-4-5",
max_tokens=200,
messages=[
{
"role": "user",
"content": "안녕 Claude. 나는 앳플리 AX 프로젝트를 시작하고 있어. 3줄로 응원해줘."
}
]
)

print(response.content[0].text)
