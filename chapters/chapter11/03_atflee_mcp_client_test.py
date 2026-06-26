"""
Chapter 11 선택 실습 11-3: 앳플리 MCP 클라이언트 테스트

11-2에서 만든 FastMCP 기반 앳플리 MCP 서버(02_atflee_mcp_server_basic.py)에
Python MCP 클라이언트로 연결해 등록된 도구 목록을 확인하고
실제 tool call을 실행한다.

이번 단계에서는 Claude API, ANTHROPIC_API_KEY, UPSTAGE_API_KEY를 사용하지 않는다.
목표는 MCP 서버와 MCP 클라이언트 연결 구조를 이해하는 것이다.

실행:
  python chapters/chapter11/03_atflee_mcp_client_test.py

연결 방식:
  FastMCP Client(SERVER_PATH) 를 사용해 서버 스크립트를 stdio 방식으로 자동 실행.
  별도 터미널에서 서버를 수동 실행할 필요 없다.
"""

import os
import sys
import json
import asyncio
from datetime import datetime

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports", "chapter11")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SERVER_PATH = os.path.join(PROJECT_ROOT, "chapters", "chapter11", "02_atflee_mcp_server_basic.py")

from fastmcp import Client


# ==============================
# 출력 helper
# ==============================
def _safe_print(text: str) -> None:
    """Windows cp949 콘솔 인코딩 오류를 방지해 출력한다."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


def _print_section(title: str) -> None:
    _safe_print(f"\n[{title}]")


def _print_json(data) -> None:
    """dict 또는 CallToolResult content를 보기 좋게 출력한다."""
    if data is None:
        _safe_print("  (결과 없음)")
        return

    # CallToolResult 처리: structuredContent 우선, 없으면 content 텍스트
    if hasattr(data, "structuredContent") and data.structuredContent:
        payload = data.structuredContent
    elif hasattr(data, "content"):
        # content는 TextContent / ImageContent 등의 리스트
        texts = []
        for block in data.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        payload_str = "\n".join(texts)
        try:
            payload = json.loads(payload_str)
        except Exception:
            _safe_print(payload_str)
            return
    else:
        payload = data

    try:
        _safe_print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        _safe_print(str(payload))


def _extract_dict(result) -> dict:
    """CallToolResult에서 dict를 추출한다."""
    if result is None:
        return {}

    if hasattr(result, "structuredContent") and result.structuredContent:
        return result.structuredContent if isinstance(result.structuredContent, dict) else {}

    if hasattr(result, "content"):
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        try:
            return json.loads("\n".join(texts))
        except Exception:
            return {}

    if isinstance(result, dict):
        return result

    return {}


# ==============================
# 메인 비동기 흐름
# ==============================
async def run_client_test() -> dict:
    """
    MCP 서버에 연결해 4개의 tool call을 실행하고
    전체 결과를 dict로 반환한다.
    """
    _safe_print("=" * 60)
    _safe_print("[Chapter 11-3] 앳플리 MCP 클라이언트 테스트")
    _safe_print("=" * 60)

    _print_section("Server")
    _safe_print(SERVER_PATH)

    summary = {
        "server_connect":   "실패",
        "tools_list":       "실패",
        "tool_call":        "실패",
    }

    tools_list_raw    = []
    status_result_raw = {}
    tools_result_raw  = {}
    wiki_result_raw   = {}
    voc_result_raw    = {}

    async with Client(SERVER_PATH) as client:
        summary["server_connect"] = "성공"

        # ---- A. tool 목록 조회 ----
        _print_section("Tools / List")
        tools = await client.list_tools()
        tools_list_raw = [{"name": t.name, "description": t.description} for t in tools]

        if tools:
            summary["tools_list"] = "성공"
            for t in tools:
                _safe_print(f"  - {t.name}")
        else:
            _safe_print("  (도구 없음)")

        # ---- B. get_atflee_status ----
        _safe_print("\n[Tool Call] get_atflee_status")
        status_raw = await client.call_tool("get_atflee_status", {})
        status_result_raw = _extract_dict(status_raw)
        _print_json(status_raw)

        # ---- C. list_atflee_tools ----
        _safe_print("\n[Tool Call] list_atflee_tools")
        tools_raw = await client.call_tool("list_atflee_tools", {})
        tools_result_raw = _extract_dict(tools_raw)
        _print_json(tools_raw)

        # ---- D. search_atflee_wiki ----
        wiki_question = "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?"
        _safe_print("\n[Tool Call] search_atflee_wiki")
        _safe_print(f"  질문: {wiki_question}")

        wiki_raw = await client.call_tool(
            "search_atflee_wiki",
            {"question": wiki_question, "top_k": 3},
        )
        wiki_result_raw = _extract_dict(wiki_raw)

        _safe_print("  TOP 3:")
        for idx, item in enumerate(wiki_result_raw.get("results", []), start=1):
            snippet = item.get("snippet", "")[:60].replace("\n", " ")
            _safe_print(
                f"  {idx}. {item.get('source_file')} / "
                f"score={item.get('score')} / "
                f"snippet: {snippet}..."
            )

        # ---- E. classify_atflee_voc ----
        voc_message = "제품이 불량 같고 교환하고 싶어요."
        _safe_print("\n[Tool Call] classify_atflee_voc")
        _safe_print(f"  문의: {voc_message}")

        voc_raw = await client.call_tool(
            "classify_atflee_voc",
            {"customer_message": voc_message},
        )
        voc_result_raw = _extract_dict(voc_raw)

        _safe_print(f"  intent:             {voc_result_raw.get('intent')}")
        _safe_print(f"  severity:           {voc_result_raw.get('severity')}")
        _safe_print(f"  owner_team:         {voc_result_raw.get('owner_team')}")
        _safe_print(f"  needs_human_review: {voc_result_raw.get('needs_human_review')}")
        _safe_print(f"  reason:             {voc_result_raw.get('reason')}")

        summary["tool_call"] = "성공"

    # ---- Summary ----
    _print_section("Summary")
    _safe_print(f"  - MCP 서버 연결:    {summary['server_connect']}")
    _safe_print(f"  - tool 목록 조회:   {summary['tools_list']}")
    _safe_print(f"  - tool call 실행:   {summary['tool_call']}")
    _safe_print("")
    _safe_print("  [11-2 vs 11-3 차이]")
    _safe_print("  - 11-2: FastMCP 서버를 만들고 @mcp.tool 데코레이터로 도구를 등록하는 단계")
    _safe_print("  - 11-3: MCP 클라이언트가 stdio로 서버를 자동 실행하고 도구를 원격 호출하는 단계")
    _safe_print("  - 핵심 차이: 11-2는 서버 코드, 11-3은 클라이언트 코드 - 같은 도구를 반대 방향에서 본다")

    return {
        "server_path":               SERVER_PATH,
        "tools":                     tools_list_raw,
        "status_result":             status_result_raw,
        "tools_result":              tools_result_raw,
        "wiki_search_result":        wiki_result_raw,
        "voc_classification_result": voc_result_raw,
        "summary":                   summary,
    }


# ==============================
# 리포트 저장
# ==============================
def save_report(result: dict) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"atflee_mcp_client_test_{ts}.json"
    filepath = os.path.join(REPORTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return filepath


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    result   = asyncio.run(run_client_test())
    filepath = save_report(result)

    _safe_print(f"\n[리포트 저장]")
    _safe_print(f"  {filepath}")
    _safe_print("")
    _safe_print("[실습 완료]")
    _safe_print("MCP 클라이언트에서 서버 도구를 stdio로 연결해 호출하는 구조가 완성되었습니다.")
    _safe_print("다음 단계: 11-4 - Claude tool_use API와 MCP 서버를 연결하는 에이전트 구현")
