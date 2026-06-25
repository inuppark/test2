"""
Chapter 8-3: Claude Agent Loop 실습
한 번의 사용자 질문에서 Claude가 필요하면 여러 도구를 순서대로 사용한다.

[8-2 vs 8-3 차이]
8-2: 사용자 질문 → Claude 응답 → tool_use 1회 처리 → 최종 답변
     (도구를 한 번만 실행하고 끝낸다)

8-3: 사용자 질문 → Claude 응답 → tool_use 실행 → 결과 전달
     → Claude가 추가 도구 요청 가능 → 반복 → tool_use가 없으면 종료
     (최대 5회 루프로 여러 도구를 연속 사용할 수 있다)
"""

import os
import sys
import json
from dotenv import load_dotenv
from anthropic import Anthropic

# Windows cp949 터미널에서 한글·이모지 출력 시 UnicodeEncodeError를 방지한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==============================
# 프로젝트 루트 경로 설정
# ==============================
# 이 파일 위치: chapters/chapter08/ → 두 단계 위가 프로젝트 루트다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rag_utils import (
    search_wiki,
    build_rag_context,
    get_source_file_names,
    evaluate_rag_answer,
)


# ==============================
# API Key 설정
# ==============================

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

client = Anthropic(api_key=api_key)
model_name = "claude-sonnet-4-5"


# ==============================
# 도구(Tool) 정의
# ==============================
# 8-2와 동일한 세 가지 도구를 Claude에게 알려준다.
# Agent Loop에서도 도구 정의 자체는 변하지 않는다.

tools = [
    {
        "name": "search_atflee_wiki",
        "description": (
            "앳플리 제품, 앱, 배송, 환불, AS, 문의 관련 data/wiki 문서를 검색한다. "
            "제품 사용법, 앱 연결, 배송, 환불, AS, 고객센터 문의 관련 질문에 사용한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "앳플리 위키에서 검색할 사용자 질문 또는 최적화된 검색 쿼리"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "classify_voc",
        "description": (
            "고객 문의나 불만 문장을 VOC 유형, 심각도, 담당팀, 다음 액션으로 분류한다. "
            "고객 불만, AS 지연, 배송 지연, 앱 오류, 제품 파손 문의에 사용한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_message": {
                    "type": "string",
                    "description": "분류할 고객 문의 원문"
                }
            },
            "required": ["customer_message"]
        }
    },
    {
        "name": "evaluate_answer_quality",
        "description": (
            "앳플리 봇 답변이 안전한지 평가한다. "
            "참고 문서 포함 여부, 위험 단정 표현, 개인정보 입력 유도 여부를 점검한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "품질 평가할 답변 문장"
                },
                "source_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "답변이 참고해야 하는 문서 파일명 목록"
                }
            },
            "required": ["answer", "source_files"]
        }
    }
]


# ==============================
# 도구 실행 함수 1: search_atflee_wiki
# ==============================

def search_atflee_wiki(question):
    """
    앳플리 위키 문서를 검색하고 결과를 반환한다.
    8-2와 동일한 구현을 그대로 사용한다.
    """
    search_results = search_wiki(question, top_k=3)
    rag_context = build_rag_context(search_results)
    source_files = get_source_file_names(search_results)

    return {
        "source_files": source_files,
        "search_results": [
            {
                "file_name": r["file_name"],
                "score": r["score"],
                "snippet": r.get("snippet", "")
            }
            for r in search_results
        ],
        "rag_context": rag_context
    }


# ==============================
# 도구 실행 함수 2: classify_voc
# ==============================

# VOC 분류에 사용할 키워드 사전 (8-2와 동일)
_VOC_KEYWORDS = {
    "앱 연결":  ["연결", "앱", "블루투스", "연동", "페어링"],
    "배송":     ["배송", "택배", "송장", "지연", "출고", "도착"],
    "AS":       ["as", "에이에스", "수리", "고장", "불량", "파손"],
    "교환/환불": ["환불", "반품", "교환", "취소", "결제"],
}

_SEVERITY_KEYWORDS = [
    "화남", "화가", "짜증", "항의", "항의합니다",
    "환불", "고장", "안됨", "안돼", "늦", "지연", "실망", "최악"
]


def classify_voc(customer_message):
    """
    고객 문의 문장을 키워드 기반으로 분류한다.
    외부 API 없이 Python 규칙만으로 처리하는 간단한 구현이다.

    실제 서비스에서는 Claude API 또는 전용 분류 모델을 사용하는 것이 더 정확하다.
    """
    msg_lower = customer_message.lower()

    # 1. VOC 유형 분류: 키워드 매칭으로 유형을 결정한다.
    issue_type = "일반 문의"
    for voc_type, keywords in _VOC_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            issue_type = voc_type
            break

    # 2. 심각도 분류: 부정적 감정·상황 키워드가 있으면 "높음"
    severity = "높음" if any(kw in msg_lower for kw in _SEVERITY_KEYWORDS) else "중간"

    # 3. 담당팀 배정
    owner_map = {
        "앱 연결":  "앱/CS팀",
        "배송":     "물류/CS팀",
        "AS":       "CS/품질팀",
        "교환/환불": "CS/운영팀",
        "일반 문의": "CS팀",
    }
    owner_team = owner_map[issue_type]

    # 4. 다음 액션
    next_action = (
        f"고객에게 불편을 드려 죄송하다는 공감 표현을 먼저 전달한다. "
        f"사실 관계를 확인하고 필요한 정보를 수집한 뒤 {owner_team}에 전달한다."
    )

    # 5. 인간 검토 필요 여부: 심각도 높음이거나 AS/교환/환불 유형
    needs_human_review = severity == "높음" or issue_type in ("AS", "교환/환불")

    return {
        "issue_type": issue_type,
        "severity": severity,
        "owner_team": owner_team,
        "next_action": next_action,
        "needs_human_review": needs_human_review
    }


# ==============================
# 도구 실행 함수 3: evaluate_answer_quality
# ==============================

def evaluate_answer_quality_tool(answer, source_files):
    """
    utils.rag_utils.evaluate_rag_answer를 호출해 답변 품질을 평가한다.
    7-6에서 만든 평가 함수를 도구로 재사용하는 예시다.
    """
    return evaluate_rag_answer(answer, source_files)


# ==============================
# 도구 라우터
# ==============================
# Claude가 요청한 도구 이름에 따라 해당 Python 함수를 실행한다.
# Agent Loop에서 매 라운드마다 이 함수가 호출된다.

def run_tool(tool_name, tool_input):
    """도구 이름을 보고 대응하는 함수를 실행한다."""
    if tool_name == "search_atflee_wiki":
        return search_atflee_wiki(tool_input["question"])
    elif tool_name == "classify_voc":
        return classify_voc(tool_input["customer_message"])
    elif tool_name == "evaluate_answer_quality":
        return evaluate_answer_quality_tool(
            tool_input["answer"],
            tool_input["source_files"]
        )
    else:
        return {"error": f"알 수 없는 도구입니다: {tool_name}"}


# ==============================
# System Prompt
# ==============================
# Chapter 4.3 원칙을 반영한 에이전트 역할 정의.
# Agent Loop에서는 여러 도구를 연속 사용하는 흐름을 명시한다.

system_prompt = """
# Role
너는 앳플리 Tool Use 기반 업무 지원 에이전트다.

# Goal
사용자 요청을 해결하기 위해 필요한 도구를 순서대로 사용한다.
앳플리 제품/정책 안내, VOC 분류, 답변 품질 평가를 함께 수행할 수 있다.

# Context
너에게는 세 가지 도구가 있다.
- search_atflee_wiki: 앳플리 위키 검색
- classify_voc: 고객 문의/VOC 분류
- evaluate_answer_quality: 답변 품질 평가

# Tool Selection Rules
- 제품, 앱 연결, 배송, 환불, AS, 문의 방법을 묻는 질문에는 search_atflee_wiki를 사용한다.
- 고객 불만, VOC 분석, 담당팀 분류, 심각도 판단 요청에는 classify_voc를 사용한다.
- 답변이 안전한지, 문서 근거가 있는지, 위험 표현이 있는지 확인하는 요청에는 evaluate_answer_quality를 사용한다.
- 고객 문의를 처리해야 하는 복합 요청이면 classify_voc로 분류한 뒤 search_atflee_wiki로 관련 정책을 확인할 수 있다.
- 최종 답변을 만들었다면 필요 시 evaluate_answer_quality로 안전성을 점검할 수 있다.
- 단, 불필요한 도구 호출은 반복하지 않는다.

# Safety Rules
- 도구 결과에 없는 내용을 확정하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 단정하지 않는다.
- 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.

# Output Format
최종 답변은 아래 형식으로 작성한다.

1. 처리 요약
2. 사용한 도구
3. 도구 결과 요약
4. 최종 답변
5. 다음 액션
"""


# ==============================
# Agent Loop 함수
# ==============================
# 8-3의 핵심: 한 번의 질문에서 Claude와 Python이 여러 번 대화하며
# 필요한 도구를 모두 사용한 뒤 최종 답변을 출력한다.

def run_agent_loop(user_question, max_tool_rounds=5):
    """
    Agent Loop를 실행한다.

    흐름:
    1. 사용자 질문을 messages에 넣고 Claude에게 전달한다.
    2. Claude 응답에 tool_use가 있으면 Python이 도구를 실행한다.
    3. tool_result를 messages에 추가하고 Claude에게 다시 전달한다.
    4. Claude가 tool_use를 요청하지 않으면 최종 답변으로 보고 루프를 종료한다.
    5. max_tool_rounds에 도달하면 강제로 루프를 종료한다.

    매개변수:
    - user_question: 사용자가 입력한 질문 문자열
    - max_tool_rounds: 최대 도구 사용 횟수 (기본값 5)
    """

    print("=" * 60)
    print(f"사용자 질문:\n{user_question}")
    print("=" * 60)

    # messages에 대화 기록을 누적한다.
    # 8-2와 달리, 도구 결과가 쌓이면서 Claude가 이전 맥락을 모두 참고한다.
    messages = [
        {"role": "user", "content": user_question}
    ]

    # 이번 질문 처리에서 사용한 도구 이름을 순서대로 기록한다.
    used_tools = []

    # ── Agent Loop 시작 ──────────────────────────────────────
    for round_index in range(max_tool_rounds):

        # Claude에게 현재까지의 대화 전체를 전달한다.
        response = client.messages.create(
            model=model_name,
            max_tokens=1500,
            temperature=0.2,
            system=system_prompt,
            tools=tools,
            messages=messages
        )

        # Claude 응답을 messages에 추가한다.
        # assistant 역할로 누적해야 다음 라운드에서 맥락이 유지된다.
        messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )

        # 응답에서 tool_use 블록만 추출한다.
        # Claude가 여러 도구를 한 번에 요청하는 경우도 처리한다.
        tool_use_blocks = [
            block for block in response.content
            if block.type == "tool_use"
        ]

        # tool_use가 없으면 → 최종 답변 출력 후 루프 종료
        if not tool_use_blocks:
            print("\n[최종 답변]")
            print("-" * 60)
            for block in response.content:
                if block.type == "text":
                    print(block.text)

            print("\n[사용한 도구]")
            if used_tools:
                for tool_name in used_tools:
                    print(f"  - {tool_name}")
            else:
                print("  - 사용한 도구 없음")
            return  # 루프 정상 종료

        # ── 도구 실행 단계 ────────────────────────────────────
        # 이번 라운드에서 실행한 도구 결과들을 모아서 한 번에 전달한다.
        tool_results_content = []

        for tool_block in tool_use_blocks:
            tool_name   = tool_block.name
            tool_input  = tool_block.input
            tool_use_id = tool_block.id

            # 사용 도구 기록 (나중에 요약 출력에 사용)
            used_tools.append(tool_name)

            print(f"\n[라운드 {round_index + 1} 도구 요청]")
            print(f"  도구명: {tool_name}")
            print(f"  입력:   {json.dumps(tool_input, ensure_ascii=False)}")

            # Python이 직접 도구를 실행한다.
            tool_result = run_tool(tool_name, tool_input)

            # 도구 실행 결과를 콘솔에 요약 출력한다.
            # rag_context는 길어서 생략하고 핵심 정보만 보여준다.
            print("  [도구 실행 결과 요약]")
            result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
            print(result_str[:1500])  # 너무 길면 1500자까지만 출력

            # tool_result를 다음 라운드용 메시지로 준비한다.
            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(tool_result, ensure_ascii=False, indent=2)
                }
            )

        # 모든 도구 결과를 하나의 user 메시지로 묶어 messages에 추가한다.
        # Claude는 다음 라운드에서 이 결과를 보고 추가 도구가 필요한지 판단한다.
        messages.append(
            {
                "role": "user",
                "content": tool_results_content
            }
        )

    # ── 루프 강제 종료 ─────────────────────────────────────────
    # max_tool_rounds에 도달했지만 Claude가 계속 tool_use를 요청하는 경우
    print("\n[중단]")
    print(f"최대 도구 사용 라운드 {max_tool_rounds}회에 도달했습니다.")
    print("도구 호출이 반복되고 있어 종료합니다.")


# ==============================
# 실습용 복합 업무 질문
# ==============================
# classify_voc + search_atflee_wiki를 순서대로 사용하고
# 필요하면 evaluate_answer_quality까지 사용하는 흐름을 유도한다.

user_question = """
고객이 이렇게 문의했어.

"앳플리 체중계가 앱이랑 계속 연결이 안 되고,
AS 문의를 남겼는데 답변이 너무 늦어서 화가 납니다."

이 문의를 처리해줘.

1. VOC 유형과 심각도를 분류하고
2. 관련 앳플리 정책/가이드를 찾아보고
3. 고객에게 보낼 답변 방향을 제안해줘.
"""

# Agent Loop 실행
run_agent_loop(user_question, max_tool_rounds=5)
