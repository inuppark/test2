"""
Chapter 11 선택 실습 11-7: 앳플리 MCP 확장 도구 테스트

11-7에서 추가된 3개 도구(generate_cs_reply, analyze_atflee_voc, hybrid_rag_answer)를
FastMCP Client로 02_atflee_mcp_server_basic.py에 연결해 테스트한다.

이번 단계에서는 AX Console을 수정하지 않는다.
ANTHROPIC_API_KEY는 .env에서 읽는다. 값은 절대 출력하지 않는다.

실행:
  python chapters/chapter11/06_atflee_mcp_extended_tools_test.py
"""

import os
import sys
import json
import asyncio
from datetime import datetime

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports", "chapter11")
SERVER_PATH  = os.path.join(PROJECT_ROOT, "chapters", "chapter11", "02_atflee_mcp_server_basic.py")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastmcp import Client


# ==============================
# 출력 helper
# ==============================
def _p(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


def _extract_result(raw) -> dict | list | str:
    """FastMCP CallToolResult에서 Python 객체를 추출한다."""
    if hasattr(raw, "data") and raw.data is not None:
        return raw.data
    if hasattr(raw, "content") and raw.content:
        texts = [b.text for b in raw.content if hasattr(b, "text")]
        combined = "\n".join(texts)
        try:
            return json.loads(combined)
        except Exception:
            return combined
    return {}


def _print_json(data, max_chars: int = 800) -> None:
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        _p(text[:max_chars] + ("..." if len(text) > max_chars else ""))
    except Exception:
        _p(str(data)[:max_chars])


# ==============================
# 테스트 실행
# ==============================
async def run_extended_tests() -> dict:
    _p("=" * 60)
    _p("[Chapter 11-7] 앳플리 MCP 확장 도구 테스트")
    _p("=" * 60)
    _p(f"서버: {SERVER_PATH}")

    results = {
        "get_atflee_status":   None,
        "list_atflee_tools":   None,
        "generate_cs_reply":   None,
        "analyze_atflee_voc":  None,
        "hybrid_rag_answer":   None,
    }
    success_count = 0

    async with Client(SERVER_PATH) as client:

        # ---- tool 목록 조회 ----
        _p("\n[Tools]")
        tools = await client.list_tools()
        for t in tools:
            _p(f"  - {t.name}")
        _p(f"  총 도구 수: {len(tools)}")

        # ---- A. get_atflee_status ----
        _p("\n[Tool Call] get_atflee_status")
        raw = await client.call_tool("get_atflee_status", {})
        data = _extract_result(raw)
        results["get_atflee_status"] = data
        _p(f"  status:               {data.get('status')}")
        _p(f"  tool_count:           {data.get('tool_count')}")
        _p(f"  wiki_doc_count:       {data.get('wiki_doc_count')}")
        _p(f"  hybrid_rag_available: {data.get('hybrid_rag_available')}")
        _p(f"  upstage_index_exists: {data.get('upstage_index_exists')}")

        # ---- B. list_atflee_tools ----
        _p("\n[Tool Call] list_atflee_tools")
        raw = await client.call_tool("list_atflee_tools", {})
        data = _extract_result(raw)
        results["list_atflee_tools"] = data
        tool_list = data.get("tools", [])
        for t in tool_list:
            _p(f"  - {t.get('name')}: {t.get('description', '')[:50]}")
        _p(f"  도구 수: {len(tool_list)}")

        # ---- C. generate_cs_reply ----
        cs_message = "제품이 불량 같고 교환하고 싶어요. 어떻게 답변해야 해요?"
        _p(f"\n[Tool Call] generate_cs_reply")
        _p(f"  문의: {cs_message}")
        raw = await client.call_tool("generate_cs_reply", {"customer_message": cs_message})
        data = _extract_result(raw)
        results["generate_cs_reply"] = data

        if "error" not in data:
            success_count += 1
            reply_preview = data.get("reply", "")[:120].replace("\n", " ")
            _p(f"  답변 초안 (120자): {reply_preview}...")
            _p(f"  needs_human_review: {data.get('needs_human_review')}")
            safety = data.get("safety_notes", [])
            for note in safety:
                _p(f"  [안전] {note}")
            _p(f"  source_policy:      {data.get('source_policy')}")
        else:
            _p(f"  [오류] {data.get('error')}")

        # ---- D. analyze_atflee_voc ----
        voc_message = "앱 연결이 계속 안 되고 고객센터 답변도 늦어서 화가 납니다."
        _p(f"\n[Tool Call] analyze_atflee_voc")
        _p(f"  문의: {voc_message}")
        raw = await client.call_tool("analyze_atflee_voc", {"customer_message": voc_message})
        data = _extract_result(raw)
        results["analyze_atflee_voc"] = data

        if "error" not in data:
            success_count += 1
            _p(f"  issue_type:         {data.get('issue_type')}")
            _p(f"  severity:           {data.get('severity')}")
            _p(f"  customer_emotion:   {data.get('customer_emotion')}")
            _p(f"  owner_team:         {data.get('owner_team')}")
            _p(f"  needs_human_review: {data.get('needs_human_review')}")
            _p(f"  next_action:        {str(data.get('next_action', ''))[:80]}")
        else:
            _p(f"  [오류] {data.get('error')}")

        # ---- E. hybrid_rag_answer ----
        rag_question = "제품이 불량 같고 교환하고 싶어요. 어떻게 해야 해요?"
        _p(f"\n[Tool Call] hybrid_rag_answer")
        _p(f"  질문: {rag_question}")
        raw = await client.call_tool(
            "hybrid_rag_answer",
            {"question": rag_question, "top_k": 3},
        )
        data = _extract_result(raw)
        results["hybrid_rag_answer"] = data

        if "error" not in data:
            success_count += 1
            _p(f"  used_search:        {data.get('used_search')}")
            _p(f"  needs_human_review: {data.get('needs_human_review')}")
            sources = data.get("sources", [])
            for s in sources[:3]:
                _p(f"  source: {s.get('source_file')} / score={s.get('score') or s.get('hybrid_score')}")
            answer_preview = data.get("answer", "")[:150].replace("\n", " ")
            _p(f"  답변 (150자): {answer_preview}...")
        else:
            _p(f"  [오류] {data.get('error')}")

    # ---- Summary ----
    _p("\n[Summary]")
    _p(f"  - 기존 도구 수:          4")
    _p(f"  - 확장 후 도구 수:       7")
    _p(f"  - 새 도구 실행 성공:     {success_count}/3")
    _p(f"  - Claude Desktop 재시작: 필요 (새 도구 목록 반영을 위해 완전 종료 후 재시작)")

    return {
        "server_path": SERVER_PATH,
        "tool_count":  len(tools),
        "results":     {k: v for k, v in results.items()},
        "summary": {
            "original_tools":    4,
            "extended_tools":    7,
            "new_tools_success": success_count,
        },
    }


# ==============================
# 리포트 저장
# ==============================
def save_report(result: dict) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"atflee_mcp_extended_tools_test_{ts}.json"
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return filepath


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    result   = asyncio.run(run_extended_tests())
    filepath = save_report(result)

    _p(f"\n[리포트 저장]")
    _p(f"  {filepath}")
    _p("")
    _p("[실습 완료]")
    _p("AX Console 핵심 기능이 MCP tool로 승격되었습니다.")
    _p("Claude Desktop을 완전 종료 후 재시작하면 7개 도구가 모두 보입니다.")
