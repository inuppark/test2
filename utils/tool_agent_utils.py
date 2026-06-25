"""
utils/tool_agent_utils.py
앳플리 Tool Use Agent 공통 모듈.

apps/ax_console_v0.py 와 chapters/chapter08/ 실습 파일에서 공통으로 사용한다.

설계 원칙:
- Anthropic client는 이 모듈 안에서 직접 만들지 않는다.
- client와 model_name은 호출하는 쪽에서 전달받는다.
- reports 저장 경로는 project_root를 기준으로 만든다.
"""

import os
import json
from datetime import datetime

try:
    # 패키지 모드(utils.tool_agent_utils)로 임포트될 때 사용하는 상대 임포트.
    # sys.path 설정 없이도 동일 패키지 내 모듈을 안정적으로 찾는다.
    from .rag_utils import (
        search_wiki,
        build_rag_context,
        get_source_file_names,
        evaluate_rag_answer,
    )
except ImportError:
    # 스크립트로 직접 실행되거나 경로 설정이 다를 때 절대 임포트로 폴백한다.
    from utils.rag_utils import (
        search_wiki,
        build_rag_context,
        get_source_file_names,
        evaluate_rag_answer,
    )


# ==============================
# 도구(Tool) 정의
# ==============================

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
# System Prompt
# ==============================
# 8-4/8-5와 동일한 방향 유지.
# 최종 답변은 JSON만 출력하도록 강하게 지시한다.

_SYSTEM_PROMPT = """
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
# 도구 실행 함수들
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


def evaluate_answer_quality_tool(answer, source_files):
    """utils.rag_utils.evaluate_rag_answer를 호출해 답변 품질을 평가한다."""
    return evaluate_rag_answer(answer, source_files)


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
# 유틸 함수
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


def extract_text_from_response(response):
    """Claude 응답에서 text 블록만 골라 하나의 문자열로 합친다."""
    text_parts = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)

    return "\n".join(text_parts).strip()


# ==============================
# Agent Loop 함수
# ==============================

def run_structured_agent(client, model_name, user_question, max_tool_rounds=5):
    """
    Agent Loop를 실행하고 구조화된 결과를 반환한다.

    client와 model_name은 호출하는 쪽에서 전달받는다.
    이 모듈 안에서 직접 Anthropic 클라이언트를 생성하지 않는다.

    반환값:
    {
        "result":     파싱된 JSON dict (파싱 실패 시 None),
        "raw_text":   Claude 최종 원문 텍스트,
        "used_tools": 사용한 도구 이름 목록,
        "tool_logs":  라운드별 도구 실행 로그,
        "error":      오류 메시지 (정상 처리 시 None)
    }
    """
    messages   = [{"role": "user", "content": user_question}]
    used_tools = []
    tool_logs  = []

    for round_index in range(max_tool_rounds):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=1800,
                temperature=0.2,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages
            )
        except Exception as error:
            return {
                "result":     None,
                "raw_text":   "",
                "used_tools": used_tools,
                "tool_logs":  tool_logs,
                "error":      f"Claude API 호출 오류: {error}"
            }

        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [
            block for block in response.content
            if block.type == "tool_use"
        ]

        # tool_use가 없으면 → 최종 답변 처리
        if not tool_use_blocks:
            final_text = extract_text_from_response(response)
            cleaned    = clean_json_text(final_text)

            try:
                result = json.loads(cleaned)
                result["used_tools"] = used_tools
                return {
                    "result":     result,
                    "raw_text":   final_text,
                    "used_tools": used_tools,
                    "tool_logs":  tool_logs,
                    "error":      None
                }
            except json.JSONDecodeError as error:
                return {
                    "result":     None,
                    "raw_text":   final_text,
                    "used_tools": used_tools,
                    "tool_logs":  tool_logs,
                    "error":      f"JSON 파싱 실패: {error}"
                }

        # 도구 실행
        tool_results_content = []

        for tool_block in tool_use_blocks:
            tool_name   = tool_block.name
            tool_input  = tool_block.input
            tool_use_id = tool_block.id

            used_tools.append(tool_name)

            tool_result = run_tool(tool_name, tool_input)

            # 라운드별 로그를 기록한다.
            tool_logs.append({
                "round":       round_index + 1,
                "tool_name":   tool_name,
                "tool_input":  tool_input,
                "tool_result": tool_result
            })

            tool_results_content.append({
                "type":        "tool_result",
                "tool_use_id": tool_use_id,
                "content":     json.dumps(tool_result, ensure_ascii=False, indent=2)
            })

        messages.append({"role": "user", "content": tool_results_content})

    return {
        "result":     None,
        "raw_text":   "",
        "used_tools": used_tools,
        "tool_logs":  tool_logs,
        "error":      f"최대 도구 사용 라운드 {max_tool_rounds}회에 도달했습니다."
    }


# ==============================
# 결과 저장 함수
# ==============================

def save_structured_result(project_root, result, user_question):
    """
    구조화된 결과를 reports/chapter08 폴더에 JSON 파일로 저장한다.

    project_root: 프로젝트 루트 경로 (호출하는 쪽에서 전달)
    result:       run_structured_agent()["result"] 값
    user_question: 원본 사용자 질문

    반환값: 저장된 파일의 전체 경로 문자열
    """
    reports_dir = os.path.join(project_root, "reports", "chapter08")
    os.makedirs(reports_dir, exist_ok=True)

    created_at     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_with_metadata = {
        "created_at":    created_at,
        "chapter":       "chapter08",
        "agent_name":    "atflee_tool_use_agent",
        "user_question": user_question.strip(),
        "result":        result
    }

    file_name = f"atflee_tool_result_{file_timestamp}.json"
    file_path = os.path.join(reports_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result_with_metadata, f, ensure_ascii=False, indent=2)

    return file_path


# ==============================
# 이력 목록 조회 함수
# ==============================

def list_saved_results(project_root):
    """
    reports/chapter08 폴더에 저장된 .json 파일 목록을 읽어 반환한다.

    반환값: 파일명 내림차순(최신 우선) 정렬된 목록
    [
        {
            "file_name":  "atflee_tool_result_20260625_131613.json",
            "file_path":  "...(전체 경로)...",
            "created_at": "2026-06-25 13:16:13",
            "summary":    "...",
            "issue_type": "복합",
            "severity":   "높음",
            "owner_team": "앱/CS팀"
        },
        ...
    ]
    폴더가 없거나 파일이 없으면 빈 리스트를 반환한다.
    """
    reports_dir = os.path.join(project_root, "reports", "chapter08")

    if not os.path.exists(reports_dir):
        return []

    items = []

    for file_name in sorted(os.listdir(reports_dir), reverse=True):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(reports_dir, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = data.get("result", {}) or {}

            # created_at: 파일 내부 값 우선, 없으면 파일 수정 시간으로 대체한다.
            created_at = data.get("created_at")
            if not created_at:
                mtime = os.path.getmtime(file_path)
                created_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            items.append({
                "file_name":  file_name,
                "file_path":  file_path,
                "created_at": created_at,
                "summary":    result.get("summary", ""),
                "issue_type": result.get("issue_type", ""),
                "severity":   result.get("severity", ""),
                "owner_team": result.get("owner_team", ""),
            })

        except Exception:
            # 읽기/파싱 실패 파일은 건너뛴다.
            continue

    return items


# ==============================
# 이력 파일 로드 함수
# ==============================

def load_saved_result(file_path):
    """
    선택한 JSON 파일을 읽어 dict로 반환한다.

    파일이 없거나 JSON 파싱에 실패하면 None을 반환한다.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
