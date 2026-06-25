"""
Chapter 9-3: Tool Use Agent Prompt Caching 확인 실습

목적:
  run_structured_agent를 동일한 질문으로 2번 실행해
  search_atflee_wiki 결과의 rag_context에 적용된 cache_control이
  두 번째 호출에서 cache_read_input_tokens로 나타나는지 확인한다.

핵심 개념:
  - search_atflee_wiki 도구 결과의 rag_context를 <cached_rag_context> text block으로
    분리해 cache_control: ephemeral을 적용한다.
  - 같은 rag_context가 두 번째 호출에서도 전달되면 캐시에서 읽어온다.
  - usage_logs에서 라운드별 cache_creation_input_tokens / cache_read_input_tokens를
    확인할 수 있다.
"""

import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

# 프로젝트 루트 경로를 sys.path에 추가한다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.tool_agent_utils import run_structured_agent

# =========================
# API Key 및 클라이언트 초기화
# =========================

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

client = Anthropic(api_key=api_key)

# claude-sonnet-4-5: Prompt Caching 지원 모델
model_name = "claude-sonnet-4-5"


# =========================
# 출력 헬퍼
# =========================

def print_usage_logs(label, usage_logs):
    """
    usage_logs를 라운드별로 출력한다.

    비교 포인트:
      첫 번째 실행 → 캐시가 없으므로 cache_creation_input_tokens 발생 가능
      두 번째 실행 → 동일한 rag_context 캐시를 읽어 cache_read_input_tokens 발생 가능
    """
    print(f"\n[{label} usage_logs]")

    if not usage_logs:
        print("  usage_logs 없음")
        return

    for log in usage_logs:
        print(f"  라운드 {log.get('round', '?')}: {log}")


def print_result_summary(label, output):
    """실행 결과에서 오류 여부와 사용 도구만 간략히 출력한다."""
    error = output.get("error")
    result = output.get("result")

    print(f"\n[{label} 결과 요약]")

    if error:
        print(f"  오류: {error}")
    elif result:
        print(f"  이슈 유형: {result.get('issue_type', '-')}")
        print(f"  심각도:   {result.get('severity', '-')}")
        print(f"  사용 도구: {output.get('used_tools', [])}")
    else:
        print("  결과 없음")


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    # 앳플리 봇 체중계 연결 + AS 지연 복합 문의 — 두 번 동일하게 실행한다.
    user_question = """
고객이 이렇게 문의했습니다.

"앳플리 체중계가 앱이랑 계속 연결이 안 되고,
AS 문의를 남겼는데 답변이 너무 늦어서 화가 납니다."

이 문의를 처리해서 업무 시스템에 넣을 수 있는 JSON으로 정리해줘.
"""

    # ── 첫 번째 실행 ──────────────────────────────────────────────
    # search_atflee_wiki가 처음 호출되면 rag_context 캐시가 생성된다.
    # cache_creation_input_tokens가 발생할 수 있다.
    print("=" * 50)
    print("첫 번째 실행")
    print("=" * 50)

    first = run_structured_agent(client, model_name, user_question)
    print_result_summary("첫 번째 실행", first)
    print_usage_logs("첫 번째 실행", first.get("usage_logs", []))

    # ── 두 번째 실행 ──────────────────────────────────────────────
    # 동일한 질문 → 동일한 search_wiki 결과 → 동일한 rag_context
    # cache_read_input_tokens가 나타나면 캐시 재사용이 된 것이다.
    print("\n" + "=" * 50)
    print("두 번째 실행")
    print("=" * 50)

    second = run_structured_agent(client, model_name, user_question)
    print_result_summary("두 번째 실행", second)
    print_usage_logs("두 번째 실행", second.get("usage_logs", []))

    # ── 비교 요약 ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("비교 요약")
    print("=" * 50)
    print("첫 번째 cache_creation 합계:",
          sum((log.get("cache_creation_input_tokens") or 0)
              for log in first.get("usage_logs", [])))
    print("두 번째 cache_read 합계:    ",
          sum((log.get("cache_read_input_tokens") or 0)
              for log in second.get("usage_logs", [])))
    print("\n  두 번째 실행의 cache_read 합계 > 0 이면 Prompt Caching이 정상 작동한 것입니다.")
