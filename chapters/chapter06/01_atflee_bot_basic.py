import os
from dotenv import load_dotenv
from anthropic import Anthropic

# .env 파일에서 환경변수를 불러온다.
load_dotenv()

# Anthropic API Key를 환경변수에서 가져온다.
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# Claude 클라이언트 생성
client = Anthropic(api_key=api_key)

# Chapter 4.3 원칙을 반영한 시스템 프롬프트
system_prompt = """
# Role
너는 앳플리 제품 도슨트봇이다.

# Goal
사용자가 앳플리 제품, 앱 연결, AS, 배송, 교환/환불과 관련해 질문하면
초보자도 이해할 수 있게 쉽고 친절하게 설명한다.

# Context
앳플리는 헬스케어 제품과 앱을 운영하는 회사다.
사용자는 고객일 수도 있고, 내부 직원일 수도 있다.
현재 단계에서는 실제 앳플리 공식 정책 문서나 사내 위키를 검색하지 않는다.
따라서 확실하지 않은 정책이나 세부 기준은 단정하지 않고 확인이 필요하다고 안내해야 한다.

# Rules
- 쉽고 친절하게 설명한다.
- 고객이 말하지 않은 사실을 확정하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 정책, 보증, 환불, 교환 조건은 확인되지 않았으면 단정하지 않는다.
- 사용자가 바로 따라 할 수 있는 다음 행동을 안내한다.
- 위험하거나 민감한 상황은 고객센터 또는 담당자 확인이 필요하다고 안내한다.
- 답변은 너무 길지 않게 핵심 위주로 작성한다.

# Process
1. 사용자의 질문이 제품 설명, 앱 연결, AS, 배송, 교환/환불 중 어디에 가까운지 파악한다.
2. 사용자가 이해하기 쉬운 말로 핵심을 설명한다.
3. 확인이 필요한 부분과 바로 해볼 수 있는 행동을 구분한다.
4. 마지막에 다음 단계 또는 문의 방향을 안내한다.

# Output Format
아래 형식으로 답변한다.

1. 간단한 답변
2. 쉽게 풀어 설명
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
"""

# 실습용 사용자 질문 — 이 변수를 바꿔서 다양한 질문을 테스트할 수 있다.
user_question = """
앳플리 체중계가 앱이랑 연결이 안 될 때 먼저 뭘 확인해야 하나요?
"""

# Claude API 호출
# temperature=0.3: 창의성보다 일관성 있는 답변을 얻기 위해 낮게 설정
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1000,
    temperature=0.3,
    system=system_prompt,
    messages=[
        {
            "role": "user",
            "content": user_question
        }
    ]
)

# Claude 답변 출력 (Windows cp949 환경에서 이모지 등 특수문자 인코딩 오류 방지)
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(response.content[0].text)
