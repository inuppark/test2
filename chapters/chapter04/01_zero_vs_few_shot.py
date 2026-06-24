import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

model_name = "claude-sonnet-4-5"

customer_message = """
체중계가 앱이랑 연결이 계속 안 돼요.
어제부터 계속 시도했는데 실패하고,
AS 문의를 남겼는데 답변도 늦어서 답답합니다.
"""

zero_shot_prompt = f"""
아래 고객 문의를 분석해줘.

고객 문의:
{customer_message}
"""

few_shot_prompt = f"""
너는 앳플리 VOC 분석 전문가다.
고객 문의를 아래 형식으로 분석한다.

분석 형식:
- 이슈 유형:
- 심각도:
- 고객 감정:
- 원인 후보:
- 담당 부서:
- 고객 응대 방향:
- 다음 액션:

예시 1:
고객 문의:
배송이 예정일보다 3일 늦어져고, 알림도 받지 못했습니다.

분석:
- 이슈 유형: 배송 지연
- 심각도: 중간
- 고객 감정: 불만, 불안
- 원인 후보: 물류 지연, 배송 알림 누락, 주문량 증가
- 담당 부서: 물류/CS
- 고객 응대 방향: 지연 사과, 현재 배송 상태 확인, 예상 도착일 안내
- 다음 액션: 배송 추적 확인 후 고객에게 개별 안내

예시 2:
고객 문의:
제품을 받았는데 포장이 찢어져 있고 본체에 흠집이 있습니다.

분석:
- 이슈 유형: 제품 파손/품질
- 심각도: 높음
- 고객 감정: 분노, 실망
- 원인 후보: 배송 중 파손, 포장재 부족, 출고 검수 누락
- 담당 부서: 물류/품질/CS
- 고객 응대 방향: 즉시 사과, 교환 또는 환불 절차 안내, 사진 확인 요청
- 다음 액션: 파손 사진 확인 후 교환 접수

이제 아래 고객 문의를 같은 형식으로 분석해줘.

고객 문의:
{customer_message}
"""

def ask_claude(title, prompt):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(response.content[0].text)

ask_claude("ZERO-SHOT RESULT", zero_shot_prompt)
ask_claude("FEW-SHOT RESULT", few_shot_prompt)
