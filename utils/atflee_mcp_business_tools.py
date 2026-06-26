"""
utils/atflee_mcp_business_tools.py

앳플리 MCP 서버 확장 도구(Chapter 11-7)에서 사용하는 비즈니스 로직 공통 유틸.

제공 함수:
  generate_cs_reply_core  — CS 답변 초안 생성
  analyze_atflee_voc_core — VOC 구조화 분석 (Claude 기반)
  hybrid_rag_answer_core  — 하이브리드 또는 키워드 RAG 기반 답변 생성
  get_hybrid_rag_status   — 하이브리드 RAG 사용 가능 여부 반환

보안:
  ANTHROPIC_API_KEY, UPSTAGE_API_KEY는 .env에서만 읽는다.
  키 값은 로그나 반환 dict에 절대 포함하지 않는다.
"""

import os
import sys
import json
import re

from dotenv import load_dotenv

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
_UPSTAGE_API_KEY   = os.getenv("UPSTAGE_API_KEY")

CLAUDE_MODEL_NAME = "claude-sonnet-4-5"

_HIGH_REVIEW_KEYWORDS = ["불량", "파손", "하자", "고장", "환불", "교환", "반품", "as", "a/s", "수리"]
_PRIVACY_KEYWORDS     = ["주문번호", "연락처", "주소", "전화번호", "이메일"]

_POLICY_KEYWORD_MAP = {
    "atflee_delivery_refund_policy": ["배송", "배달", "택배", "반품", "교환", "환불", "반환"],
    "atflee_app_guide":              ["앱", "연결", "블루투스", "페어링", "sync", "bluetooth"],
    "atflee_contact_guide":          ["고객센터", "전화", "1:1", "상담", "문의"],
    "atflee_product_guide":          ["제품", "품질", "불량", "파손", "수리", "as", "a/s"],
}


# ==============================
# 내부 helper
# ==============================
def _get_anthropic_client():
    """Anthropic 클라이언트를 반환한다. API Key가 없으면 None을 반환한다."""
    if not _ANTHROPIC_API_KEY:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=_ANTHROPIC_API_KEY)
    except ImportError:
        return None


def _clean_json_text(text: str) -> str:
    """Claude 응답에서 마크다운 코드블록을 제거하고 JSON 문자열만 반환한다."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _infer_source_policy(message: str) -> list:
    """문의 내용에서 관련 가능성이 있는 정책 문서명을 추론한다."""
    message_lower = message.lower()
    matches = []
    for doc, keywords in _POLICY_KEYWORD_MAP.items():
        if any(kw in message_lower for kw in keywords):
            matches.append(doc)
    return matches if matches else ["atflee_contact_guide"]


def _infer_safety_notes(message: str, reply: str = "") -> list:
    """문의 내용과 답변에서 안전 주의사항을 생성한다."""
    notes = []
    msg_lower = message.lower()

    if any(kw in msg_lower for kw in _HIGH_REVIEW_KEYWORDS):
        notes.append("불량/교환/환불/AS 관련 문의는 사람 검토 후 처리가 필요합니다.")

    if any(kw in message for kw in _PRIVACY_KEYWORDS):
        notes.append("개인정보(주문번호, 연락처, 주소 등)는 공개 채팅에 입력하지 않도록 안내해야 합니다.")

    if reply and any(phrase in reply for phrase in ["배송이 완료", "접수가 완료", "처리됐습니다"]):
        notes.append("확정되지 않은 상태를 단정한 표현이 포함될 수 있으므로 발송 전 검토가 필요합니다.")

    return notes


# ==============================
# CS 답변 초안 생성
# ==============================
def generate_cs_reply_core(customer_message: str, tone: str = "정중하고 친절하게") -> dict:
    """
    고객 문의를 받아 앳플리 기준 CS 답변 초안을 생성한다.

    반환:
      reply: 고객에게 전달할 답변 초안
      safety_notes: 발송 전 확인 권고 사항
      needs_human_review: 사람 검토 필요 여부
      source_policy: 참고된 정책 문서 후보 목록
    """
    anthropic_client = _get_anthropic_client()
    if not anthropic_client:
        return {
            "reply":              "",
            "safety_notes":       ["ANTHROPIC_API_KEY가 설정되어 있지 않습니다. .env를 확인하세요."],
            "needs_human_review": True,
            "source_policy":      [],
            "error":              "ANTHROPIC_API_KEY 없음",
        }

    needs_human_review = any(kw in customer_message.lower() for kw in _HIGH_REVIEW_KEYWORDS)
    source_policy      = _infer_source_policy(customer_message)

    system_prompt = f"""# Role
너는 앳플리 고객센터 매니저다.

# Goal
고객 문의에 대해 {tone} CS 답변 초안을 작성한다.

# Context
앳플리는 헬스케어 제품(체중계, 체성분계 등)과 앱을 운영하는 회사다.
고객은 배송, 앱 연결, 제품 품질, AS, 환불/교환 등으로 불편을 겪을 수 있다.

# Rules
- 고객 불편에 먼저 공감한다.
- 필요한 경우 사과 표현을 포함한다.
- 고객이 말하지 않은 사실을 확정하지 않는다.
- 실제 주문번호, 배송 상태, AS 접수번호 등 확인되지 않은 정보를 지어내지 않는다.
- 개인정보(주문번호, 연락처, 주소 등)는 공개 채팅에 입력하지 말고 고객센터 1:1 문의로 전달하도록 안내한다.
- 다음 처리 절차를 명확히 안내한다.
- 고객에게 바로 보낼 수 있는 톤으로 작성한다.

# Output
답변문만 작성한다. JSON, 마크다운 코드블록, 추가 설명은 넣지 않는다."""

    user_prompt = f"""아래 고객 문의에 대한 CS 답변 초안을 작성해줘.

고객 문의:
{customer_message}"""

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1000,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    reply       = response.content[0].text
    safety_notes = _infer_safety_notes(customer_message, reply)

    return {
        "reply":              reply,
        "safety_notes":       safety_notes,
        "needs_human_review": needs_human_review,
        "source_policy":      source_policy,
    }


# ==============================
# VOC 구조화 분석
# ==============================
def analyze_atflee_voc_core(customer_message: str) -> dict:
    """
    고객 문의를 Claude로 구조화 분석해 반환한다.
    classify_atflee_voc보다 더 풍부한 분석(감정, 원인, 응대 방향 포함)을 제공한다.

    반환 dict 키:
      issue_type, severity, customer_emotion, possible_causes,
      owner_team, reply_direction, next_action, needs_human_review
    """
    anthropic_client = _get_anthropic_client()
    if not anthropic_client:
        return {
            "issue_type":       "분석 불가",
            "severity":         "알 수 없음",
            "customer_emotion": "알 수 없음",
            "possible_causes":  [],
            "owner_team":       "CS팀",
            "reply_direction":  "",
            "next_action":      "",
            "needs_human_review": True,
            "error":            "ANTHROPIC_API_KEY 없음",
        }

    system_prompt = """# Role
너는 앳플리 VOC Agent v1이다.

# Goal
고객 문의를 읽고 이슈 유형, 심각도, 고객 감정, 원인 후보, 담당 부서, 응대 방향, 다음 액션을 분석한다.

# Context
앳플리는 헬스케어 제품과 앱을 운영하는 회사다.
고객 문의에는 배송, 제품 품질, 앱 연결, AS, 환불/교환 관련 이슈가 포함될 수 있다.

# Rules
- 실무자가 바로 처리할 수 있도록 구조화된 JSON으로만 답한다.
- 고객이 말하지 않은 사실을 확정하지 않는다.
- 원인은 반드시 가능성으로 표현한다.
- 심각도는 낮음, 중간, 높음 중 하나로 판단한다.
- 사람이 확인해야 하는 이슈는 needs_human_review를 true로 표시한다.
- JSON 앞뒤에 설명, 마크다운, 코드블록을 붙이지 않는다.

# Output Format
{
  "issue_type": "이슈 유형",
  "severity": "낮음 | 중간 | 높음",
  "customer_emotion": "고객 감정",
  "possible_causes": ["원인 후보 1", "원인 후보 2"],
  "owner_team": "담당 부서",
  "reply_direction": "고객 응대 방향",
  "next_action": "다음 액션",
  "needs_human_review": true 또는 false
}"""

    few_shot = """예시:
고객 문의: 제품을 받았는데 포장이 찢어져 있고 본체에 흠집이 있습니다.
출력:
{
  "issue_type": "제품 파손/품질",
  "severity": "높음",
  "customer_emotion": "분노, 실망",
  "possible_causes": ["배송 중 파손 가능성", "포장재 부족 가능성"],
  "owner_team": "품질/물류/CS",
  "reply_direction": "불편에 대해 즉시 사과하고 교환 또는 환불 절차를 안내한다.",
  "next_action": "파손 사진을 확인한 뒤 교환 접수 또는 환불 절차를 진행한다.",
  "needs_human_review": true
}"""

    user_prompt = f"""{few_shot}

이제 아래 고객 문의를 같은 기준으로 분석해라.

고객 문의:
{customer_message}"""

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1200,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = response.content[0].text
    try:
        result = json.loads(_clean_json_text(raw_text))
    except Exception:
        result = {
            "issue_type":       "파싱 오류",
            "severity":         "알 수 없음",
            "customer_emotion": "알 수 없음",
            "possible_causes":  [],
            "owner_team":       "CS팀",
            "reply_direction":  raw_text[:200],
            "next_action":      "",
            "needs_human_review": True,
        }

    return result


# ==============================
# 하이브리드 RAG 상태 확인
# ==============================
def get_hybrid_rag_status() -> dict:
    """
    하이브리드 RAG 사용 가능 여부를 반환한다.
    UPSTAGE_API_KEY와 Upstage 인덱스 파일 모두 있어야 hybrid 사용 가능.
    """
    upstage_key_present = bool(_UPSTAGE_API_KEY)

    try:
        from utils.upstage_rag_utils import get_upstage_index_status
        index_status = get_upstage_index_status()
    except Exception:
        index_status = {"exists": False}

    hybrid_available = upstage_key_present and index_status.get("exists", False)

    return {
        "hybrid_available":      hybrid_available,
        "upstage_key_present":   upstage_key_present,
        "upstage_index_exists":  index_status.get("exists", False),
        "upstage_chunk_count":   index_status.get("chunk_count", 0),
    }


# ==============================
# 하이브리드 RAG 답변 생성
# ==============================
def hybrid_rag_answer_core(question: str, top_k: int = 3) -> dict:
    """
    질문에 대해 하이브리드 RAG 또는 키워드 검색 기반으로 앳플리 문서를 검색하고 답변한다.

    UPSTAGE_API_KEY + Upstage 인덱스가 모두 있으면 hybrid_rag_utils를 사용한다.
    하나라도 없으면 키워드 검색(search_wiki) + Claude 답변으로 폴백한다.

    반환:
      answer: Claude 최종 답변
      used_search: "hybrid" 또는 "keyword_fallback"
      sources: 사용된 문서 목록
      needs_human_review: 사람 검토 필요 여부
      safety_notes: 안전 주의사항
    """
    anthropic_client = _get_anthropic_client()
    if not anthropic_client:
        return {
            "answer":             "",
            "used_search":        "none",
            "sources":            [],
            "needs_human_review": True,
            "safety_notes":       ["ANTHROPIC_API_KEY가 설정되어 있지 않습니다."],
            "error":              "ANTHROPIC_API_KEY 없음",
        }

    rag_status = get_hybrid_rag_status()
    used_search = "hybrid" if rag_status["hybrid_available"] else "keyword_fallback"

    sources      = []
    context_text = ""

    if rag_status["hybrid_available"]:
        # --- Hybrid 경로 ---
        from utils.hybrid_rag_utils import (
            search_hybrid_rag,
            build_hybrid_answer_context,
            filter_hybrid_results_for_answer,
        )
        payload = search_hybrid_rag(
            question=question,
            upstage_api_key=_UPSTAGE_API_KEY,
            top_k=top_k,
        )
        hybrid_results = payload.get("hybrid_results", [])
        if payload.get("error"):
            used_search = "keyword_fallback"
        else:
            filtered = filter_hybrid_results_for_answer(hybrid_results, max_items=top_k)
            context_text = build_hybrid_answer_context(hybrid_results, max_items=top_k)
            sources = [
                {
                    "source_file":  r["source_file"],
                    "hybrid_score": round(r.get("hybrid_score", 0), 4),
                    "sources":      r.get("sources", []),
                }
                for r in filtered
            ]

    if used_search == "keyword_fallback":
        # --- 키워드 폴백 경로 ---
        from utils.rag_utils import search_wiki
        keyword_results = search_wiki(question, top_k=top_k)
        context_parts = []
        for r in keyword_results:
            file_name = r.get("file_name", "")
            snippet   = r.get("snippet", "")
            score     = r.get("score", 0)
            context_parts.append(f"[출처: {file_name} / score: {score}]\n{snippet}")
            sources.append({"source_file": file_name, "score": score})
        context_text = "\n\n---\n\n".join(context_parts)

    # --- Claude 답변 생성 ---
    tag = "hybrid_rag_context" if used_search == "hybrid" else "keyword_rag_context"
    search_label = "하이브리드 RAG (키워드 + Upstage 의미 검색)" if used_search == "hybrid" else "키워드 검색 (Upstage 인덱스 없음)"

    system_prompt = f"""# Role
너는 앳플리 봇이다.

# Goal
사용자 질문에 대해 검색된 앳플리 위키/정책 문서를 근거로 안전하게 답변한다.
검색 방식: {search_label}

# Rules
- <{tag}> 안에 있는 정보만 확정적으로 말한다.
- Context에 없는 내용은 추측하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 개인정보(주문번호, 연락처, 주소 등)는 공개 채팅에 입력하지 않도록 안내한다.
- 검색 결과가 부족하면 "정확한 확인이 필요합니다"라고 말한다.

# Output Format
1. 간단한 답변
2. 근거가 되는 앳플리 정보
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 문서"""

    user_content = (
        f"<{tag}>\n{context_text}\n</{tag}>\n\n"
        f"<user_question>\n{question}\n</user_question>"
    )

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    answer = response.content[0].text
    needs_human_review = any(kw in question.lower() for kw in _HIGH_REVIEW_KEYWORDS)
    safety_notes       = _infer_safety_notes(question, answer)

    return {
        "answer":             answer,
        "used_search":        used_search,
        "sources":            sources,
        "needs_human_review": needs_human_review,
        "safety_notes":       safety_notes,
    }
