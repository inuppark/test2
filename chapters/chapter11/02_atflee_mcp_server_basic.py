"""
Chapter 11 선택 실습 11-2: 앳플리 MCP 서버 Basic

11-1에서 Python 코드 안에서 직접 실행했던 도구들을
MCP(Model Context Protocol) 서버 형태로 외부에 노출한다.

이 서버는 Claude Desktop 또는 MCP 클라이언트가 연결해
표준 프로토콜로 앳플리 도구를 호출할 수 있게 한다.

실행:
  python chapters/chapter11/02_atflee_mcp_server_basic.py

연결:
  Claude Desktop 설정 파일에 이 서버를 등록하거나
  11-3에서 만들 MCP 클라이언트로 연결한다.
"""

import os
import sys
import json
from typing import Dict, List, Any

from fastmcp import FastMCP

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
WIKI_DIR     = os.path.join(PROJECT_ROOT, "data", "wiki")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# MCP 서버 생성
mcp = FastMCP("atflee-mcp-server-v0")

# 의도 분류 규칙 테이블
_INTENT_RULES = [
    ("앱 연결",      ["앱", "연결", "블루투스", "페어링", "bluetooth", "sync"]),
    ("배송",         ["배송", "택배", "언제 와", "도착", "shipping"]),
    ("교환/반품",    ["교환", "반품", "환불", "return", "refund"]),
    ("AS",           ["as", "a/s", "수리", "고장", "접수"]),
    ("제품 품질",    ["불량", "파손", "하자", "결함", "망가"]),
    ("고객센터 문의", ["전화", "고객센터", "문의", "연락처", "상담"]),
]

_HIGH_SEVERITY_KEYWORDS = ["불량", "파손", "하자", "고장", "as", "a/s", "환불", "교환"]
_MID_SEVERITY_KEYWORDS  = ["배송", "연결", "앱", "블루투스"]

_OWNER_TEAM_MAP = {
    "앱 연결":      "앱/CS팀",
    "배송":         "물류/CS팀",
    "교환/반품":    "CS/물류팀",
    "AS":           "CS/품질팀",
    "제품 품질":    "품질/CS팀",
    "고객센터 문의": "CS팀",
    "기타":         "CS팀",
}


# ==============================
# 내부 helper
# ==============================
def read_wiki_files() -> List[Dict[str, str]]:
    """
    data/wiki 폴더의 .md 파일을 읽어 리스트로 반환한다.
    파일이 없으면 빈 리스트를 반환한다.
    """
    if not os.path.isdir(WIKI_DIR):
        return []

    documents = []
    for filename in sorted(os.listdir(WIKI_DIR)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(WIKI_DIR, filename)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        documents.append({"source_file": filename, "content": content})

    return documents


def _extract_snippet(question: str, content: str, max_chars: int = 300) -> str:
    """질문 토큰이 처음 등장하는 위치 근처의 텍스트를 snippet으로 반환한다."""
    tokens = question.lower().split()
    content_lower = content.lower()

    best_pos = len(content)
    for token in tokens:
        pos = content_lower.find(token)
        if 0 <= pos < best_pos:
            best_pos = pos

    start = max(0, best_pos - 50)
    return content[start : start + max_chars].replace("\n", " ").strip()


# ==============================
# MCP tool 1: 서버 상태
# ==============================
@mcp.tool
def get_atflee_status() -> Dict[str, Any]:
    """
    앳플리 MCP 서버 v0의 현재 상태를 반환한다.
    서버가 살아있는지, 어떤 도구를 제공하는지 확인할 때 사용한다.
    """
    wiki_count = len(read_wiki_files())

    return {
        "service":         "atflee-mcp-server-v0",
        "status":          "ready",
        "description":     "앳플리 위키 검색, VOC 분류, 도구 목록 확인을 제공하는 MCP 서버입니다.",
        "wiki_doc_count":  wiki_count,
        "wiki_dir":        WIKI_DIR,
        "available_tools": [
            "get_atflee_status",
            "list_atflee_tools",
            "search_atflee_wiki",
            "classify_atflee_voc",
        ],
    }


# ==============================
# MCP tool 2: 도구 목록
# ==============================
@mcp.tool
def list_atflee_tools() -> Dict[str, Any]:
    """
    앳플리 MCP 서버에서 제공하는 도구 목록과 설명을 반환한다.
    MCP 클라이언트가 사용 가능한 도구를 파악할 때 호출한다.
    """
    return {
        "tools": [
            {
                "name":        "get_atflee_status",
                "description": "앳플리 MCP 서버 상태 확인",
            },
            {
                "name":        "list_atflee_tools",
                "description": "사용 가능한 앳플리 MCP 도구 목록 확인",
            },
            {
                "name":        "search_atflee_wiki",
                "description": "data/wiki 문서에서 질문과 관련 있는 앳플리 정보를 검색",
            },
            {
                "name":        "classify_atflee_voc",
                "description": "고객 문의를 유형, 심각도, 담당팀 기준으로 간단 분류",
            },
        ]
    }


# ==============================
# MCP tool 3: 위키 검색
# ==============================
@mcp.tool
def search_atflee_wiki(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    질문과 관련 있는 앳플리 위키 문서를 키워드 방식으로 검색한다.
    data/wiki 폴더의 .md 파일을 질문 토큰 매칭으로 스코어링해 상위 top_k개를 반환한다.

    Args:
        question: 검색할 질문 또는 키워드
        top_k: 반환할 최대 문서 수 (기본값 3)
    """
    documents = read_wiki_files()

    if not documents:
        return {
            "question": question,
            "top_k":    top_k,
            "results":  [],
            "note":     f"data/wiki 폴더({WIKI_DIR})에 문서가 없습니다.",
        }

    tokens = [t for t in question.lower().split() if len(t) > 1]

    scored = []
    for doc in documents:
        content_lower = doc["content"].lower()
        score = sum(1 for token in tokens if token in content_lower)
        snippet = _extract_snippet(question, doc["content"])
        scored.append(
            {
                "source_file": doc["source_file"],
                "score":       score,
                "snippet":     snippet,
            }
        )

    scored.sort(key=lambda x: -x["score"])

    return {
        "question": question,
        "top_k":    top_k,
        "results":  scored[:top_k],
    }


# ==============================
# MCP tool 4: VOC 분류
# ==============================
@mcp.tool
def classify_atflee_voc(customer_message: str) -> Dict[str, Any]:
    """
    고객 문의를 간단한 규칙 기반으로 분류한다.
    의도, 심각도, 담당팀, 사람 검토 필요 여부를 반환한다.

    Args:
        customer_message: 분류할 고객 문의 원문
    """
    message_lower = customer_message.lower()

    # 의도 분류
    detected_intent = "기타"
    for intent_name, keywords in _INTENT_RULES:
        if any(kw in message_lower for kw in keywords):
            detected_intent = intent_name
            break

    # 심각도 판단
    if any(kw in message_lower for kw in _HIGH_SEVERITY_KEYWORDS):
        severity = "높음"
    elif any(kw in message_lower for kw in _MID_SEVERITY_KEYWORDS):
        severity = "중간"
    else:
        severity = "낮음"

    # 담당팀
    owner_team = _OWNER_TEAM_MAP.get(detected_intent, "CS팀")

    # 사람 검토 필요 여부
    high_review_intents = {"AS", "제품 품질", "교환/반품"}
    needs_human_review  = detected_intent in high_review_intents or severity == "높음"

    # 분류 이유
    if detected_intent != "기타":
        reason = f"'{detected_intent}' 관련 키워드가 감지되었습니다."
    else:
        reason = "명확한 키워드가 감지되지 않아 기타로 분류되었습니다."

    if needs_human_review:
        reason += " 사람 검토가 필요합니다."

    return {
        "customer_message":   customer_message,
        "intent":             detected_intent,
        "severity":           severity,
        "owner_team":         owner_team,
        "needs_human_review": needs_human_review,
        "reason":             reason,
    }


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    print("[앳플리 MCP 서버 v0]")
    print(f"서버명: atflee-mcp-server-v0")
    print(f"위키 경로: {WIKI_DIR}")
    print()
    print("등록된 도구:")
    print("  1. get_atflee_status    - 서버 상태 확인")
    print("  2. list_atflee_tools    - 도구 목록 반환")
    print("  3. search_atflee_wiki   - data/wiki 문서 검색")
    print("  4. classify_atflee_voc  - 고객 문의 VOC 분류")
    print()
    print("이 파일은 MCP 서버로 실행됩니다.")
    print("Claude Desktop 또는 MCP 클라이언트에서 연결해 사용할 수 있습니다.")
    print("Ctrl+C로 서버를 종료할 수 있습니다.")
    print()
    mcp.run()
