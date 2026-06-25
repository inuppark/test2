"""
Chapter 9-2: 앳플리 Prompt Caching usage 비교 리포트

목적:
  같은 앳플리 위키 Context로 질문 3개를 순서대로 호출하고,
  각 호출의 cache_creation_input_tokens / cache_read_input_tokens를 표로 비교한다.
  결과는 reports/chapter09 폴더에 JSON으로 저장한다.

9-1과의 차이:
  9-1: 2번 호출, usage를 콘솔에서만 확인
  9-2: 3번 호출, usage 비교 표 + JSON 리포트 저장
"""

import os
import sys
import json
from datetime import datetime
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

    for file_name in sorted(os.listdir(wiki_dir)):
        if not file_name.endswith(".md"):
            continue

        file_path = os.path.join(wiki_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        context_parts.append(f"[문서명: {file_name}]\n{content}")

    if not context_parts:
        return "data/wiki 문서가 없습니다."

    return "\n\n---\n\n".join(context_parts)


# =========================
# 함수 2: usage dict 변환
# =========================

def get_usage_dict(usage):
    """
    response.usage 객체에서 필요한 필드를 안전하게 꺼내 dict로 반환한다.
    getattr의 두 번째 인자로 0을 지정해 필드가 없어도 오류가 나지 않는다.
    'or 0'은 None이 반환되는 경우를 숫자 0으로 대체한다.
    """
    return {
        "input_tokens":                getattr(usage, "input_tokens",                0) or 0,
        "output_tokens":               getattr(usage, "output_tokens",               0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens",     0) or 0,
    }


# =========================
# 함수 3: 캐싱된 Context로 질문
# =========================

def ask_with_cached_context(question, atflee_wiki_context, call_index):
    """
    앳플리 위키 Context에 cache_control을 적용해 Claude에게 질문을 보낸다.

    content[0]: 긴 위키 Context (cache_control: ephemeral → 캐싱 대상)
    content[1]: 실제 사용자 질문 (캐싱 없음)

    반환값: 호출 번호, 질문, 답변 미리보기(300자), usage dict를 담은 dict
    """
    response = client.messages.create(
        model=model_name,
        max_tokens=800,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        # cache_control을 적용한 블록: 동일한 Context가 오면 캐시에서 읽는다.
                        "type": "text",
                        "text": f"<atflee_wiki_context>\n{atflee_wiki_context}\n</atflee_wiki_context>",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        # 캐싱하지 않는 블록: 질문은 매번 다르다.
                        "type": "text",
                        "text": f"<user_question>\n{question}\n</user_question>"
                    }
                ]
            }
        ]
    )

    usage = get_usage_dict(response.usage)

    # content 블록이 여러 개일 수 있으므로 text 블록만 합친다.
    answer_text = ""
    for block in response.content:
        if block.type == "text":
            answer_text += block.text

    return {
        "call_index":     call_index,
        "question":       question,
        "answer_preview": answer_text[:300],   # 리포트에는 앞 300자만 저장한다.
        "usage":          usage,
    }


# =========================
# 함수 4: usage 비교 표 출력
# =========================

def print_usage_table(results):
    """
    각 호출의 usage를 마크다운 테이블 형태로 콘솔에 출력한다.

    비교 포인트:
      - cache_creation_input_tokens: 첫 호출에서 값이 있으면 캐시가 생성된 것
      - cache_read_input_tokens:     이후 호출에서 값이 있으면 캐시에서 읽은 것
    """
    print("\n[프롬프트 캐싱 usage 비교]")
    print("| 호출 | input | output | cache_creation | cache_read |")
    print("|------|------:|-------:|---------------:|-----------:|")

    for result in results:
        usage = result["usage"]
        print(
            f"| {result['call_index']}번째  | "
            f"{usage['input_tokens']:>6} | "
            f"{usage['output_tokens']:>7} | "
            f"{usage['cache_creation_input_tokens']:>14} | "
            f"{usage['cache_read_input_tokens']:>10} |"
        )


# =========================
# 함수 5: usage 합계 계산
# =========================

def summarize_usage(results):
    """
    모든 호출의 usage를 합산해 반환한다.
    cache_creation과 cache_read를 합계로 보면 전체 캐싱 효과를 파악할 수 있다.
    """
    totals = {
        "input_tokens":                0,
        "output_tokens":               0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens":     0,
    }

    for result in results:
        usage = result["usage"]
        for key in totals:
            totals[key] += usage.get(key, 0)

    return totals


# =========================
# 함수 6: 리포트 저장
# =========================

def save_report(results, totals, context_length):
    """
    호출 결과와 usage 합계를 reports/chapter09 폴더에 JSON으로 저장한다.
    파일명에 날짜시간을 넣어 중복을 방지한다.
    """
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "chapter09")
    os.makedirs(reports_dir, exist_ok=True)

    created_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "created_at":               created_at,
        "chapter":                  "chapter09",
        "report_name":              "atflee_prompt_cache_usage_report",
        "model_name":               model_name,
        "context_length_characters": context_length,
        "results":                  results,
        "totals":                   totals,
        "interpretation": [
            "첫 번째 호출에서는 긴 Context에 대해 cache_creation_input_tokens가 발생할 수 있다.",
            "이후 동일한 Context를 재사용하면 cache_read_input_tokens가 발생할 수 있다.",
            "앳플리 봇처럼 같은 data/wiki 문서를 반복 사용하는 경우 Prompt Caching으로 비용 효율을 높일 수 있다.",
        ],
    }

    file_name = f"prompt_cache_usage_{file_timestamp}.json"
    file_path = os.path.join(reports_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return file_path


# =========================
# 함수 7: .gitignore 확인
# =========================

def check_reports_gitignore():
    """
    .gitignore에 reports/ 가 포함되어 있는지 확인한다.
    결과 파일이 GitHub에 올라가지 않도록 보장하기 위한 안전 체크이다.
    """
    gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")

    if not os.path.exists(gitignore_path):
        return False

    with open(gitignore_path, "r", encoding="utf-8") as file:
        gitignore_content = file.read()

    return "reports/" in gitignore_content or "reports" in gitignore_content


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    # 1단계: 앳플리 위키 문서 전체를 하나의 Context로 합친다.
    print("[앳플리 위키 Context 로드 중...]")
    atflee_wiki_context = load_atflee_wiki_context()
    context_length = len(atflee_wiki_context)

    print("[앳플리 위키 Context 길이]")
    print(f"  {context_length:,} characters")

    # 2단계: 같은 Context로 3개의 질문을 순서대로 보낸다.
    #
    # 기대 패턴:
    #   1번째 호출: cache_creation_input_tokens > 0  (캐시 생성)
    #   2번째 호출: cache_read_input_tokens > 0      (캐시 재사용)
    #   3번째 호출: cache_read_input_tokens > 0      (캐시 재사용)
    questions = [
        "앳플리 체중계가 앱이랑 연결이 안 될 때 먼저 뭘 확인해야 해?",
        "배송은 보통 얼마나 걸리고, 환불은 언제 처리돼?",
        "AS 접수 여부는 어떻게 확인해야 해?",
    ]

    results = []

    for index, question in enumerate(questions, start=1):
        print(f"\n[{index}번째 호출]")
        print(f"  질문: {question}")

        result = ask_with_cached_context(question, atflee_wiki_context, index)
        results.append(result)

        print(f"  usage: {result['usage']}")

    # 3단계: usage 비교 표 출력
    print_usage_table(results)

    # 4단계: usage 합계 계산
    totals = summarize_usage(results)

    print("\n[usage 합계]")
    for key, value in totals.items():
        print(f"  {key}: {value}")

    # 5단계: 리포트 저장
    saved_path = save_report(results, totals, context_length)

    print("\n[리포트 저장 완료]")
    print(f"  {saved_path}")

    # 6단계: .gitignore 보안 확인
    if check_reports_gitignore():
        print("\n[보안 확인]")
        print("  reports 폴더가 .gitignore에 포함되어 있습니다.")
    else:
        print("\n[주의]")
        print("  reports 폴더가 .gitignore에 포함되어 있는지 확인이 필요합니다.")

    print("\n[해석]")
    print("  - 첫 호출에서 cache_creation_input_tokens가 발생하면 캐시 생성이 된 것입니다.")
    print("  - 이후 호출에서 cache_read_input_tokens가 발생하면 캐시 재사용이 된 것입니다.")
    print("  - 반복되는 긴 Context를 쓰는 앳플리 봇/AX Console에 적용 가치가 있습니다.")
