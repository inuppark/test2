"""
Chapter 11 선택 실습 11-4: Claude + MCP Tool Agent

Claude tool_use API와 앳플리 MCP 서버(02_atflee_mcp_server_basic.py)를 연결한다.

흐름:
  1. 사용자 질문과 MCP 도구 스키마를 Claude에 전달
  2. Claude가 tool_use 블록으로 사용할 도구와 입력을 선택
  3. Python이 FastMCP Client로 해당 MCP tool을 실행
  4. tool_result를 Claude에 반환 → Claude가 최종 답변 생성

이번 단계에서는 UPSTAGE_API_KEY를 사용하지 않는다.
ANTHROPIC_API_KEY는 .env에서 읽는다.

실행:
  python chapters/chapter11/04_atflee_claude_mcp_tool_agent.py
"""

import os
import sys
import json
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from anthropic import Anthropic
from fastmcp import Client

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports", "chapter11")
SERVER_PATH  = os.path.join(PROJECT_ROOT, "chapters", "chapter11", "02_atflee_mcp_server_basic.py")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CLAUDE_MODEL_NAME = "claude-sonnet-4-5"

# ==============================
# API Key 로드
# ==============================
load_dotenv()
_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise RuntimeError("ANTHROPIC_API_KEY가 설정되어 있지 않습니다. .env를 확인하세요.")

anthropic_client = Anthropic(api_key=_api_key)

# ==============================
# Claude tool schema 정의
# ==============================
TOOLS = [
    {
        "name": "get_atflee_status",
        "description": "앳플리 MCP 서버 상태와 사용 가능한 도구 목록을 확인합니다.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_atflee_tools",
        "description": "앳플리 MCP 서버에서 사용할 수 있는 도구 목록을 확인합니다.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_atflee_wiki",
        "description": (
            "앳플리 제품, 앱, 배송, 교환/반품, AS, 고객센터 관련 질문에 대해 "
            "data/wiki 문서를 검색합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "검색할 사용자 질문",
                },
                "top_k": {
                    "type": "integer",
                    "description": "반환할 검색 결과 수",
                    "default": 3,
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "classify_atflee_voc",
        "description": (
            "고객 문의를 유형, 심각도, 담당팀, 사람 검토 필요 여부로 분류합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_message": {
                    "type": "string",
                    "description": "분류할 고객 문의 원문",
                },
            },
            "required": ["customer_message"],
        },
    },
]

# ==============================
# 시스템 프롬프트
# ==============================
SYSTEM_PROMPT = """
# Role
너는 앳플리 MCP Tool Agent v0이다.

# Goal
사용자 질문을 보고 필요한 MCP 도구를 선택해 앳플리 위키 검색 또는 VOC 분류를 수행한다.

# Available Tools
- search_atflee_wiki: 앳플리 위키 검색
- classify_atflee_voc: 고객 문의 분류
- get_atflee_status: MCP 서버 상태 확인
- list_atflee_tools: 도구 목록 확인

# Tool Selection Rules
- 제품, 앱, 배송, 교환/반품, AS, 고객센터 관련 질문이면 search_atflee_wiki를 사용한다.
- 고객 불만, VOC 분류, 심각도 판단, 담당팀 판단이 필요하면 classify_atflee_voc를 사용한다.
- 서버 상태나 도구 목록 질문이면 get_atflee_status 또는 list_atflee_tools를 사용한다.
- 하나의 질문에 여러 의도가 있으면 필요한 도구를 2개 이상 사용할 수 있다.

# Safety Rules
- 실제 주문 상태, 배송 상태, AS 접수 상태를 단정하지 않는다.
- 주문번호, 연락처, 주소 등 개인정보는 공개 채팅에 입력하지 않도록 안내한다.
- 도구 결과에 없는 내용은 추측하지 않는다.

# Output Format (최종 답변 시)
1. 간단한 답변
2. 사용한 MCP 도구
3. 도구 결과 요약
4. 근거가 되는 앳플리 정보
5. 바로 해볼 수 있는 것
6. 확인이 필요한 것
""".strip()


# ==============================
# 출력 helper
# ==============================
def _p(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


# ==============================
# MCP 결과 정규화
# ==============================
def normalize_mcp_result(result) -> dict | list | str:
    """
    FastMCP CallToolResult를 JSON 직렬화 가능한 형태로 변환한다.
    result.data 가 dict/list이면 그대로 반환하고,
    content[0].text를 JSON 파싱하거나 문자열로 폴백한다.
    """
    # FastMCP 3.4.2: result.data 가 이미 Python 객체
    if hasattr(result, "data") and result.data is not None:
        return result.data

    # content 텍스트 폴백
    if hasattr(result, "content") and result.content:
        texts = [b.text for b in result.content if hasattr(b, "text")]
        combined = "\n".join(texts)
        try:
            return json.loads(combined)
        except Exception:
            return combined

    return {"raw": str(result)}


# ==============================
# MCP tool 실행
# ==============================
async def call_mcp_tool(mcp_client: Client, tool_name: str, tool_input: dict):
    """MCP 서버에 tool call을 실행하고 결과를 반환한다."""
    try:
        result = await mcp_client.call_tool(tool_name, tool_input)
        return normalize_mcp_result(result), None
    except Exception as exc:
        return None, str(exc)


# ==============================
# 에이전트 단일 질문 처리
# ==============================
async def run_agent_for_question(user_message: str, mcp_client: Client) -> dict:
    """
    Claude tool_use + MCP Client 루프를 실행하고 결과 dict를 반환한다.
    """
    _p("=" * 60)
    _p(f"[User]")
    _p(user_message)

    # ---- 1차 Claude 호출 ----
    response1 = anthropic_client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1200,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=[{"role": "user", "content": user_message}],
    )

    _p(f"\n[Claude 1차 응답]")
    _p(f"  stop_reason: {response1.stop_reason}")

    # tool_use 블록 추출
    tool_use_blocks = [b for b in response1.content if b.type == "tool_use"]
    tool_names_requested = [b.name for b in tool_use_blocks]
    _p(f"  tool_use 요청: {tool_names_requested if tool_names_requested else '없음 (직접 답변)'}")

    tool_use_records = []
    mcp_results      = []
    tool_result_msgs = []

    for block in tool_use_blocks:
        _p(f"\n[Claude tool_use 요청]")
        _p(f"  도구명: {block.name}")
        _p(f"  입력:   {json.dumps(block.input, ensure_ascii=False)}")

        # MCP tool 실행
        data, err = await call_mcp_tool(mcp_client, block.name, block.input)

        if err:
            mcp_payload = {"error": err}
            _p(f"\n[MCP tool 실행 오류]")
            _p(f"  {err}")
        else:
            mcp_payload = data
            _p(f"\n[MCP tool 실행 결과]")
            try:
                _p(json.dumps(mcp_payload, ensure_ascii=False, indent=2)[:600])
            except Exception:
                _p(str(mcp_payload)[:600])

        tool_use_records.append({
            "tool_name":  block.name,
            "tool_input": block.input,
            "tool_use_id": block.id,
        })
        mcp_results.append({"tool_name": block.name, "result": mcp_payload})

        tool_result_msgs.append({
            "type":        "tool_result",
            "tool_use_id": block.id,
            "content":     json.dumps(mcp_payload, ensure_ascii=False),
        })

    # ---- 최종 답변 ----
    if tool_result_msgs:
        # 2차 Claude 호출: tool_result 전달
        messages2 = [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": response1.content},
            {"role": "user",      "content": tool_result_msgs},
        ]
        response2 = anthropic_client.messages.create(
            model=CLAUDE_MODEL_NAME,
            max_tokens=1500,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages2,
        )
        final_answer = next(
            (b.text for b in response2.content if hasattr(b, "text")), ""
        )
    else:
        # tool_use 없음 → 1차 응답 그대로 사용
        final_answer = next(
            (b.text for b in response1.content if hasattr(b, "text")), ""
        )

    _p(f"\n[Claude 최종 답변]")
    _p(final_answer)

    _p(f"\n[Summary]")
    _p(f"  - 사용한 도구:        {', '.join(tool_names_requested) if tool_names_requested else '없음'}")
    _p(f"  - tool_use 개수:      {len(tool_use_blocks)}")
    _p(f"  - MCP 실행 성공 여부: {'성공' if not any('error' in str(r.get('result', {})) for r in mcp_results) else '일부 오류'}")

    return {
        "user_message":    user_message,
        "tool_uses":       tool_use_records,
        "mcp_results":     mcp_results,
        "final_answer":    final_answer,
        "tool_use_count":  len(tool_use_blocks),
    }


# ==============================
# 리포트 저장
# ==============================
def save_report(all_results: list) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"atflee_claude_mcp_tool_agent_{ts}.json"
    filepath = os.path.join(REPORTS_DIR, filename)

    payload = {
        "server_path":   SERVER_PATH,
        "model":         CLAUDE_MODEL_NAME,
        "test_messages": [r["user_message"] for r in all_results],
        "results":       all_results,
        "summary": {
            "total_questions": len(all_results),
            "total_tool_uses": sum(r["tool_use_count"] for r in all_results),
        },
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath


# ==============================
# 실행부
# ==============================
async def main():
    test_messages = [
        "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
        "제품이 불량 같고 교환하고 싶어요. VOC 분류도 해줘.",
        "AS 접수됐는지 확인해줘.",
        "앳플리 MCP 서버에서 사용할 수 있는 도구 목록 알려줘.",
    ]

    _p("[Chapter 11-4] 앳플리 Claude + MCP Tool Agent")
    _p(f"모델: {CLAUDE_MODEL_NAME}")
    _p(f"MCP 서버: {SERVER_PATH}")
    _p("")

    all_results = []

    # MCP 클라이언트 연결을 한 번만 열고 모든 질문에 재사용
    async with Client(SERVER_PATH) as mcp_client:
        for msg in test_messages:
            result = await run_agent_for_question(msg, mcp_client)
            all_results.append(result)
            _p("")

    filepath = save_report(all_results)

    _p("=" * 60)
    _p(f"[리포트 저장]")
    _p(f"  {filepath}")
    _p("")
    _p("[실습 완료]")
    _p("Claude tool_use가 선택한 도구를 MCP Client로 실행하는 에이전트 구조가 완성되었습니다.")
    _p("다음 단계: 11-5 - AX Console 또는 Claude Desktop과 MCP 서버 연결")


if __name__ == "__main__":
    asyncio.run(main())
