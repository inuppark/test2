"""
Chapter 8-2: Claude Multi-Tool Use 실습
여러 도구를 제공하고 Claude가 상황에 맞는 도구를 스스로 선택하게 만든다.

[8-1 vs 8-2 차이]
8-1: 도구 1개 (search_atflee_wiki) — Claude가 항상 같은 도구를 사용
8-2: 도구 3개 — Claude가 사용자 요청 의도를 파악해 적합한 도구를 선택
     - 제품/정책 질문 → search_atflee_wiki
     - VOC 분류 요청  → classify_voc
     - 답변 평가 요청 → evaluate_answer_quality
"""

import os
import sys
import json

# ==============================
# 프로젝트 루트 경로 설정
# ==============================
# 이 파일 위치: chapters/chapter08/ → 두 단계 위가 프로젝트 루트다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from anthropic import Anthropic

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
# 3개 도구를 Claude에게 알려준다.
# Claude는 사용자 요청을 보고 어떤 도구가 필요한지 스스로 결정한다.

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
    8-1과 동일한 구현이며, 멀티 도구 환경에서도 그대로 사용한다.
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

# VOC 분류에 사용할 키워드 사전
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

    # 1. VOC 유형 분류
    issue_type = "일반 문의"
    for voc_type, keywords in _VOC_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            issue_type = voc_type
            break

    # 2. 심각도 분류
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

system_prompt = """
# Role
너는 앳플리 Tool Use 기반 업무 지원 에이전트다.

# Goal
사용자 요청을 보고 필요한 도구를 선택해
앳플리 제품/정책 안내, VOC 분류, 답변 품질 평가를 지원한다.

# Context
너에게는 세 가지 도구가 있다.
- search_atflee_wiki: 앳플리 위키 검색
- classify_voc: 고객 문의/VOC 분류
- evaluate_answer_quality: 답변 품질 평가

# Tool Selection Rules
- 제품, 앱 연결, 배송, 환불, AS, 문의 방법을 묻는 질문에는 search_atflee_wiki를 사용한다.
- 고객 불만, VOC 분석, 담당팀 분류, 심각도 판단 요청에는 classify_voc를 사용한다.
- 답변이 안전한지, 문서 근거가 있는지, 위험 표현이 있는지 확인하는 요청에는 evaluate_answer_quality를 사용한다.
- 도구가 필요 없는 일반 질문에는 바로 답변해도 된다.

# Safety Rules
- 도구 결과에 없는 내용을 확정하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 단정하지 않는다.
- 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.

# Output Format
도구 사용 후 최종 답변은 아래 형식으로 작성한다.

1. 사용한 도구
2. 도구 결과 요약
3. 최종 답변
4. 다음 액션
"""


# ==============================
# 사용자 질문 처리 함수
# ==============================

def process_user_question(user_question):
    """
    사용자 질문을 처리하는 전체 흐름.

    1. Claude에게 질문 + 도구 목록 전달
    2. Claude가 tool_use를 요청하면 해당 도구 실행
    3. 도구 결과를 포함해 Claude에게 재요청
    4. 최종 답변 출력
    """
    print("=" * 60)
    print(f"사용자 질문:\n{user_question}")
    print("=" * 60)

    # Step 1: 첫 번째 Claude 호출
    # Claude는 응답에서 tool_use 블록으로 도구 사용 의사를 전달한다.
    first_response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0.2,
        system=system_prompt,
        tools=tools,
        messages=[
            {"role": "user", "content": user_question}
        ]
    )

    # tool_use 블록 탐색 (여러 개면 첫 번째만 처리)
    tool_use_block = None
    for block in first_response.content:
        if block.type == "tool_use":
            tool_use_block = block
            break

    # Step 2: 도구 실행
    if tool_use_block is None:
        # Claude가 도구 없이 바로 답변한 경우
        print("\n[안내] Claude가 도구를 요청하지 않았습니다.")
        print("\n[Claude 답변]")
        for block in first_response.content:
            if hasattr(block, "text"):
                print(block.text)
        return

    # Claude가 도구를 요청한 경우
    tool_name   = tool_use_block.name
    tool_input  = tool_use_block.input
    tool_use_id = tool_use_block.id

    print(f"\n[Claude tool_use 요청]")
    print(f"도구명: {tool_name}")
    print(f"입력:   {json.dumps(tool_input, ensure_ascii=False)}")

    # Python 코드가 도구를 실행한다.
    tool_result = run_tool(tool_name, tool_input)

    # 도구 결과 요약 출력
    print(f"\n[도구 실행 결과]")
    if tool_name == "search_atflee_wiki":
        for item in tool_result.get("search_results", []):
            print(f"  - {item['file_name']} / 점수: {item['score']}")
            snippet = item.get("snippet", "").replace("\n", " ")[:80]
            if snippet:
                print(f"    스니펫: {snippet}...")
    else:
        # classify_voc / evaluate_answer_quality 결과는 JSON으로 출력
        print(json.dumps(tool_result, ensure_ascii=False, indent=2))

    # Step 3: 두 번째 Claude 호출
    # 첫 번째 응답 + 도구 결과를 함께 전달해 최종 답변을 생성한다.
    final_response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        tools=tools,
        messages=[
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": first_response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(tool_result, ensure_ascii=False, indent=2)
                    }
                ]
            }
        ]
    )

    # Step 4: 최종 답변 출력
    print("\n[최종 답변]")
    print("-" * 60)
    for block in final_response.content:
        if hasattr(block, "text"):
            print(block.text)
    print()


# ==============================
# 실습용 테스트 질문 3개
# ==============================
# 각 질문은 서로 다른 도구를 트리거하도록 설계했다.
# Q1 → search_atflee_wiki  (제품/앱 관련 질문)
# Q2 → classify_voc        (고객 불만 VOC 분류)
# Q3 → evaluate_answer_quality (답변 안전성 평가)

test_questions = [
    "앳플리 체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
    "고객이 AS 답변이 너무 늦고 제품이 계속 고장난다고 화를 냈어. VOC 분류해줘.",
    (
        "아래 답변이 안전한지 확인해줘. "
        "답변: AS 접수 완료되었습니다. 주문번호를 입력해주세요. "
        "참고 문서: atflee_contact_guide.md"
    ),
]

# 3개 질문을 순서대로 실행한다.
for question in test_questions:
    process_user_question(question)
    print()
