import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

model_name = "claude-sonnet-4-5"

customer_message = """
체중계가 앱이랑 계속 연결이 안 됩니다.
고객센터에 문의를 남겼는데 답변이 늦고,
아이 건강 관리 때문에 매일 기록해야 하는데 며칠째 못 쓰고 있어요.
이럴 거면 왜 샀는지 모르겠습니다.
"""

normal_prompt = f"""
아래 고객 문의를 분석해줘.

고객 문의:
{customer_message}
"""

step_by_step_prompt = f"""
너는 앳플리 VOC 분석 전문가다.

아래 고객 문의를 분석하되, 반드시 다음 판단 순서를 적용해라.

판단 순서:
1. 고객 감정 파악
2. 이슈 유형 분류
3. 심각도 판단
4. 원인 후보 도출
5. 담당 부서 판단
6. 고객 응대 방향 제안
7. 내부 후속 액션 제안

주의:
- 내부적으로는 단계적으로 검토하되, 답변에는 최종 결과만 구조화해서 출력해라.
- 고객이 실제로 말하지 않은 내용을 확정하지 마라.
- 원인 후보는 가능성으로 표현해라.
- 심각도는 낮음/중간/높음 중 하나로 판단해라.

출력 형식:
1. 핵심 요약
2. 고객 감정
3. 이슈 유형
4. 심각도
5. 원인 후보
6. 담당 부서
7. 고객 응대 방향
8. 내부 후속 액션

고객 문의:
{customer_message}
"""

def ask_claude(title, prompt):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(response.content[0].text)

ask_claude("NORMAL PROMPT RESULT", normal_prompt)
ask_claude("STEP-BY-STEP PROMPT RESULT", step_by_step_prompt)
