"""
Chapter 8-5: 구조화 결과 파일 저장 실습
8-4에서 만든 Tool Use + 구조화 JSON 결과를 reports 폴더에 저장한다.

[8-4 vs 8-5 차이]
8-4: Agent Loop → 도구 실행 → JSON 파싱 → 콘솔 출력으로 끝난다.
8-5: Agent Loop → 도구 실행 → JSON 파싱 → 콘솔 출력 + 파일 저장까지 한다.
     저장된 파일은 나중에 재검토하거나 업무 자동화 파이프라인에서 불러올 수 있다.
     reports/ 폴더는 .gitignore에 포함되어 GitHub에 올라가지 않는다.
"""

import os
import sys
import json
from datetime import datetime
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
# 8-4와 동일한 세 가지 도구를 유지한다.

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
    """앳플리 위키 문서를 검색하고 결과를 반환한다."""
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
    """고객 문의 문장을 키워드 기반으로 분류한다."""
    msg_lower = customer_message.lower()

    issue_type = "일반 문의"
    for voc_type, keywords in _VOC_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            issue_type = voc_type
            break

    severity = "높음" if any(kw in msg_lower for kw in _SEVERITY_KEYWORDS) else "중간"

    owner_map = {
        "앱 연결":  "앱/CS팀",
        "배송":     "물류/CS팀",
        "AS":       "CS/품질팀",
        "교환/환불": "CS/운영팀",
        "일반 문의": "CS팀",
    }
    owner_team = owner_map[issue_type]

    next_action = (
        f"고객에게 불편을 드려 죄송하다는 공감 표현을 먼저 전달한다. "
        f"사실 관계를 확인하고 필요한 정보를 수집한 뒤 {owner_team}에 전달한다."
    )

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
    """utils.rag_utils.evaluate_rag_answer를 호출해 답변 품질을 평가한다."""
    return evaluate_rag_answer(answer, source_files)


# ==============================
# 도구 라우터
# ==============================

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
# 유틸 함수: clean_json_text
# ==============================

def clean_json_text(text):
    """Claude 응답에서 JSON 코드블록 마커(```json, ```)를 제거한다."""
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


# ==============================
# 유틸 함수: extract_text_from_response
# ==============================

def extract_text_from_response(response):
    """Claude 응답에서 text 블록만 골라 하나의 문자열로 합친다."""
    text_parts = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)

    return "\n".join(text_parts).strip()


# ==============================
# 유틸 함수: print_structured_result
# ==============================

def print_structured_result(result):
    """구조화된 결과 dict를 필드별로 콘솔에 출력한다."""
    print("\n[구조화 결과]")
    print("-" * 60)
    print(f"요약:            {result.get('summary')}")
    print(f"이슈 유형:       {result.get('issue_type')}")
    print(f"심각도:          {result.get('severity')}")
    print(f"담당팀:          {result.get('owner_team')}")
    print(f"사용 도구:       {result.get('used_tools')}")
    print(f"참고 문서:       {result.get('source_files')}")
    print(f"고객 답변 방향:  {result.get('customer_reply_direction')}")
    print(f"내부 다음 액션:  {result.get('internal_next_action')}")
    print(f"사람 검토 필요:  {result.get('needs_human_review')}")
    print(f"안전 메모:       {result.get('safety_notes')}")
    print("-" * 60)


# ==============================
# System Prompt (8-4와 동일)
# ==============================

system_prompt = """
# Role
너는 앳플리 Tool Use 기반 업무 지원 에이전트다.

# Goal
사용자 문의를 처리하기 위해 필요한 도구를 사용하고,
최종 결과를 업무 자동화에 사용할 수 있는 JSON으로 구조화한다.

# Context
너에게는 세 가지 도구가 있다.
- search_atflee_wiki: 앳플리 위키 검색
- classify_voc: 고객 문의/VOC 분류
- evaluate_answer_quality: 답변 품질 평가

# Tool Selection Rules
- 고객 불만, VOC 분석, 담당팀 분류, 심각도 판단 요청에는 classify_voc를 사용한다.
- 제품, 앱 연결, 배송, 환불, AS, 문의 방법을 묻는 질문에는 search_atflee_wiki를 사용한다.
- 답변이 안전한지, 문서 근거가 있는지, 위험 표현이 있는지 확인하는 요청에는 evaluate_answer_quality를 사용한다.
- 복합 고객 문의는 classify_voc로 분류한 뒤 search_atflee_wiki로 관련 정책을 확인한다.
- 필요하면 evaluate_answer_quality로 안전성을 점검한다.

# Safety Rules
- 도구 결과에 없는 내용을 확정하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 단정하지 않는다.
- 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
- 위험하거나 확정이 필요한 내용은 needs_human_review를 true로 둔다.

# Final Output Rules
최종 답변은 반드시 JSON만 출력한다.
마크다운 설명, 일반 문장, 코드블록 설명은 출력하지 않는다.
필드명은 아래 JSON Schema를 따른다.

{
  "summary": "문의 핵심 요약",
  "issue_type": "앱 연결 | 배송 | AS | 교환/환불 | 일반 문의 | 복합",
  "severity": "낮음 | 중간 | 높음",
  "owner_team": "담당팀",
  "used_tools": ["사용한 도구명"],
  "source_files": ["참고 문서 파일명"],
  "customer_reply_direction": "고객에게 답변할 방향",
  "internal_next_action": "내부 담당자가 해야 할 다음 액션",
  "needs_human_review": true,
  "safety_notes": ["주의할 점"]
}
"""


# ==============================
# Agent Loop + 구조화 출력 함수 (8-4와 동일)
# ==============================

def run_structured_agent(user_question, max_tool_rounds=5):
    """
    Agent Loop를 실행하고 최종 결과를 JSON dict로 반환한다.
    8-4와 동일한 구조이며, 반환값을 save_structured_result에 전달한다.
    """
    print("=" * 60)
    print(f"사용자 질문:\n{user_question.strip()}")
    print("=" * 60)

    messages = [{"role": "user", "content": user_question}]
    used_tools = []

    for round_index in range(max_tool_rounds):

        response = client.messages.create(
            model=model_name,
            max_tokens=1800,
            temperature=0.2,
            system=system_prompt,
            tools=tools,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [
            block for block in response.content
            if block.type == "tool_use"
        ]

        # tool_use가 없으면 → 최종 답변 처리
        if not tool_use_blocks:
            final_text = extract_text_from_response(response)

            print("\n[Claude 최종 원문]")
            print(final_text)

            cleaned = clean_json_text(final_text)

            try:
                result = json.loads(cleaned)
                # Python이 직접 추적한 used_tools 목록으로 덮어쓴다.
                result["used_tools"] = used_tools
                print_structured_result(result)
                return result

            except json.JSONDecodeError as error:
                print("\n[JSON 파싱 실패]")
                print(f"오류: {error}")
                print("\n[정리되지 않은 최종 답변]")
                print(final_text)
                return None

        # 도구 실행
        tool_results_content = []

        for tool_block in tool_use_blocks:
            tool_name   = tool_block.name
            tool_input  = tool_block.input
            tool_use_id = tool_block.id

            used_tools.append(tool_name)

            print(f"\n[라운드 {round_index + 1} 도구 요청]")
            print(f"  도구명: {tool_name}")
            print(f"  입력:   {json.dumps(tool_input, ensure_ascii=False)}")

            tool_result = run_tool(tool_name, tool_input)

            print("  [도구 실행 결과 요약]")
            print(json.dumps(tool_result, ensure_ascii=False, indent=2)[:1500])

            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(tool_result, ensure_ascii=False, indent=2)
                }
            )

        messages.append({"role": "user", "content": tool_results_content})

    print("\n[중단]")
    print(f"최대 도구 사용 라운드 {max_tool_rounds}회에 도달했습니다.")
    return None


# ==============================
# 추가 함수 1: save_structured_result
# ==============================
# 8-5의 핵심 기능.
# 구조화된 결과에 메타데이터를 붙여 reports/chapter08 폴더에 JSON 파일로 저장한다.

def save_structured_result(result, user_question):
    """
    구조화된 결과를 reports/chapter08 폴더에 JSON 파일로 저장한다.

    파일명 예시: atflee_tool_result_20260625_143022.json
    파일 내용: 메타데이터(created_at, chapter, agent_name, user_question) + result

    반환값: 저장된 파일의 전체 경로
    """
    # reports/chapter08 폴더가 없으면 자동으로 만든다.
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "chapter08")
    os.makedirs(reports_dir, exist_ok=True)

    # 현재 시각을 두 가지 형식으로 준비한다.
    # created_at: 사람이 읽기 쉬운 형식 (파일 내부에 기록)
    # file_timestamp: 파일명에 쓸 형식 (공백 없이)
    created_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 메타데이터와 실제 결과를 합쳐 저장할 dict를 만든다.
    result_with_metadata = {
        "created_at":  created_at,
        "chapter":     "chapter08",
        "agent_name":  "atflee_tool_use_agent",
        "user_question": user_question.strip(),
        "result":      result
    }

    # 파일명과 전체 경로를 구성한다.
    file_name = f"atflee_tool_result_{file_timestamp}.json"
    file_path = os.path.join(reports_dir, file_name)

    # JSON 파일로 저장한다.
    # ensure_ascii=False: 한글이 \uXXXX 이스케이프 없이 그대로 저장된다.
    # indent=2: 사람이 읽기 쉽게 들여쓰기를 적용한다.
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(result_with_metadata, file, ensure_ascii=False, indent=2)

    return file_path


# ==============================
# 추가 함수 2: check_reports_gitignore
# ==============================
# reports 폴더가 .gitignore에 포함되어 있는지 확인한다.
# 고객 문의 결과 파일이 GitHub에 올라가는 사고를 방지하기 위한 안전망이다.

def check_reports_gitignore():
    """
    프로젝트 루트의 .gitignore에 reports/ 또는 reports 항목이 있는지 확인한다.
    True: 포함됨 (안전) / False: 없거나 .gitignore 파일 자체가 없음 (주의 필요)
    """
    gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")

    if not os.path.exists(gitignore_path):
        return False

    with open(gitignore_path, "r", encoding="utf-8") as file:
        gitignore_content = file.read()

    return "reports/" in gitignore_content or "reports" in gitignore_content


# ==============================
# 실습용 복합 업무 질문
# ==============================

user_question = """
고객이 이렇게 문의했어.

"앳플리 체중계가 앱이랑 계속 연결이 안 되고,
AS 문의를 남겼는데 답변이 너무 늦어서 화가 납니다."

이 문의를 처리해서 업무 시스템에 넣을 수 있는 JSON으로 정리하고 저장해줘.
"""


# ==============================
# 실행 진입점
# ==============================

if __name__ == "__main__":
    # 1. Agent Loop 실행 → JSON 구조화
    result = run_structured_agent(user_question)

    if result:
        print("\n[JSON 파싱 성공]")

        # 2. 결과를 파일로 저장
        saved_path = save_structured_result(result, user_question)
        print(f"\n[저장 완료]")
        print(f"저장 경로: {saved_path}")

        # 3. .gitignore 보안 확인
        if check_reports_gitignore():
            print("\n[보안 확인]")
            print("reports 폴더가 .gitignore에 포함되어 있습니다.")
            print("저장된 파일은 git에 추적되지 않습니다.")
        else:
            print("\n[주의]")
            print("reports 폴더가 .gitignore에 포함되어 있는지 확인이 필요합니다.")
            print(".gitignore에 'reports/' 항목을 추가하세요.")
    else:
        print("\n[JSON 파싱 실패 또는 결과 없음]")
        print("저장하지 않았습니다.")
