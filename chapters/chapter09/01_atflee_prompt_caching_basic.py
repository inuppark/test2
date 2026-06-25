"""
Chapter 9-1: 앳플리 Prompt Caching 기초 실습

목적:
  반복적으로 재사용되는 긴 앳플리 위키 Context를 캐싱 대상으로 설정하고,
  같은 Context로 여러 질문을 보낼 때 cache_creation / cache_read 토큰을 확인한다.

핵심 개념:
  - Prompt Caching: 한 번 처리한 긴 Context를 서버가 캐싱해 두고,
    이후 동일 Context가 포함된 요청에서는 다시 처리하지 않고 캐시를 읽는다.
  - cache_creation_input_tokens: 이번 호출에서 캐시를 새로 만든 토큰 수
  - cache_read_input_tokens:    이번 호출에서 캐시에서 읽어 온 토큰 수
"""

import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

# 프로젝트 루트 경로를 sys.path에 추가해 utils 패키지를 안전하게 import할 수 있도록 한다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================
# API Key 및 클라이언트 초기화
# =========================

# .env 파일에서 환경 변수를 불러온다.
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

client = Anthropic(api_key=api_key)

# claude-sonnet-4-5: Prompt Caching을 지원하는 모델
model_name = "claude-sonnet-4-5"

# =========================
# 시스템 프롬프트
# =========================

system_prompt = """
# Role
너는 앳플리 위키 기반 제품/정책 안내 봇이다.

# Goal
사용자 질문에 대해 제공된 앳플리 위키 Context를 근거로 쉽고 안전하게 답변한다.

# Rules
- Context에 있는 정보만 확정적으로 말한다.
- Context에 없는 내용은 추측하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
- 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
- 답변 마지막에는 참고한 문서명을 표시한다.
"""


# =========================
# 함수 1: 앳플리 위키 Context 로드
# =========================

def load_atflee_wiki_context():
    """
    data/wiki 폴더의 모든 .md 문서를 읽어 하나의 긴 Context 문자열로 합친다.
    이 Context 전체를 캐싱 대상으로 사용한다.
    """
    wiki_dir = os.path.join(PROJECT_ROOT, "data", "wiki")

    if not os.path.exists(wiki_dir):
        return "data/wiki 폴더를 찾을 수 없습니다."

    context_parts = []

    # 파일명 오름차순으로 정렬해 일관된 순서를 보장한다.
    for file_name in sorted(os.listdir(wiki_dir)):
        if not file_name.endswith(".md"):
            continue

        file_path = os.path.join(wiki_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 각 문서 앞에 파일명 태그를 붙여서 Claude가 출처를 구분할 수 있게 한다.
        context_parts.append(f"[문서명: {file_name}]\n{content}")

    if not context_parts:
        return "data/wiki 문서가 없습니다."

    # 문서 사이에 구분선을 넣어 가독성을 높인다.
    return "\n\n---\n\n".join(context_parts)


# =========================
# 함수 2: usage 출력
# =========================

def print_usage(label, usage):
    """
    API 응답의 usage 정보를 출력한다.
    getattr로 안전하게 접근해 필드가 없어도 오류가 나지 않는다.

    usage 필드 설명:
      input_tokens               - 이번 호출에서 처리한 총 입력 토큰 수
      output_tokens              - 이번 호출에서 생성한 출력 토큰 수
      cache_creation_input_tokens - 이번 호출에서 새로 캐시를 만든 토큰 수 (첫 번째 호출에서 발생)
      cache_read_input_tokens    - 이번 호출에서 캐시에서 읽어 온 토큰 수 (두 번째 이후 호출에서 발생)
    """
    print(f"\n[{label} usage]")

    fields = [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ]

    for field in fields:
        value = getattr(usage, field, None)
        if value is not None:
            print(f"  {field}: {value}")
        else:
            print(f"  {field}: 지원되지 않거나 없음")


# =========================
# 함수 3: 캐싱된 Context로 질문
# =========================

def ask_with_cached_context(question, atflee_wiki_context, label):
    """
    앳플리 위키 Context에 cache_control을 적용해 Claude에게 질문을 보낸다.

    메시지 구조:
      content[0]: 긴 위키 Context (cache_control 적용 → 캐싱 대상)
      content[1]: 실제 사용자 질문 (캐싱 없음 → 매번 새로 전송)

    첫 번째 호출: Claude가 Context를 처리하면서 캐시를 생성한다.
    두 번째 호출: 동일한 Context가 오면 캐시에서 읽어 처리 속도와 비용이 절약된다.
    """
    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        # 캐싱 대상 블록: 긴 앳플리 위키 Context 전체
                        # cache_control을 "ephemeral"로 설정하면 Claude 서버가
                        # 이 블록을 캐싱해 두고 다음 요청에서 재사용한다.
                        "type": "text",
                        "text": f"<atflee_wiki_context>\n{atflee_wiki_context}\n</atflee_wiki_context>",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        # 캐싱 없음: 질문은 매번 다르므로 캐싱하지 않는다.
                        "type": "text",
                        "text": f"<user_question>\n{question}\n</user_question>"
                    }
                ]
            }
        ]
    )

    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print("[질문]")
    print(question)

    print("\n[답변]")
    print(response.content[0].text)

    # usage 정보 출력 (캐시 효과 확인 포인트)
    print_usage(label, response.usage)

    return response


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    # 1단계: 앳플리 위키 문서 전체를 하나의 Context로 합친다.
    print("[앳플리 위키 Context 로드 중...]")
    atflee_wiki_context = load_atflee_wiki_context()

    print("[앳플리 위키 Context 길이]")
    print(f"  {len(atflee_wiki_context):,} characters")
    print()

    # 2단계: 같은 Context로 두 가지 질문을 순서대로 보낸다.
    #
    # 기대 결과:
    #   첫 번째 호출 → cache_creation_input_tokens 발생 (캐시 생성 비용)
    #   두 번째 호출 → cache_read_input_tokens 발생   (캐시 읽기, 비용 절감)
    #
    # 실제 값은 모델 버전과 Context 크기에 따라 다를 수 있다.
    # Prompt Caching은 Context가 1,024 토큰 이상일 때 활성화된다.

    question_1 = "앳플리 체중계가 앱이랑 연결이 안 될 때 먼저 뭘 확인해야 해?"
    question_2 = "배송은 보통 얼마나 걸리고, 환불은 언제 처리돼?"

    ask_with_cached_context(question_1, atflee_wiki_context, "첫 번째 호출")
    ask_with_cached_context(question_2, atflee_wiki_context, "두 번째 호출")

    print("\n[실습 완료]")
    print("첫 번째 호출의 cache_creation_input_tokens와")
    print("두 번째 호출의 cache_read_input_tokens를 비교해보세요.")
