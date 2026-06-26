"""
Chapter 11 선택 실습 11-1: 앳플리 ReAct Agent Basic

ReAct = Reasoning + Acting
AI가 사용자 요청을 보고:
  1. Thought  — 무엇이 필요한지 판단
  2. Action   — 도구를 선택하고 실행
  3. Observation — 도구 결과를 확인
  4. Final Answer — 최종 답변 생성

이번 실습은 Claude의 tool_use 기능을 직접 사용하는 대신,
Python 코드가 ReAct 흐름을 명시적으로 보여주는 방식으로 구현한다.
(Chapter 8 Tool Use Agent와의 차이: 흐름이 코드에 명확히 드러남)

사전 준비:
  1. .env에 ANTHROPIC_API_KEY 설정
  2. data/wiki/*.md 파일 존재 확인
"""

import os
import sys
import json
import datetime

from dotenv import load_dotenv
from anthropic import Anthropic

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.rag_utils import search_wiki

CLAUDE_MODEL_NAME = "claude-sonnet-4-5"
REPORTS_DIR       = os.path.join(PROJECT_ROOT, "reports", "chapter11")

INTENT_KEYWORDS = {
    "앱 연결":    ["앱", "연결", "블루투스", "페어링", "sync", "싱크", "bluetooth"],
    "배송":       ["배송", "택배", "도착", "운송", "shipping"],
    "교환/반품":  ["교환", "반품", "환불", "refund", "return", "반환"],
    "AS":         ["as", "a/s", "수리", "서비스센터", "고장", "접수"],
    "제품 품질":  ["불량", "파손", "품질", "결함", "망가", "깨", "찌그러"],
    "고객센터 문의": ["전화번호", "고객센터", "연락처", "contact", "상담"],
}

SAFETY_DANGER_PATTERNS = [
    ("배송 완료로 단정",   ["배송이 완료됐습니다", "배송 완료되었습니다", "이미 도착했습니다"]),
    ("AS 접수 단정",       ["접수가 완료됐습니다", "접수되었습니다", "AS가 완료됐습니다"]),
    ("주문 상태 단정",     ["주문이 처리됐습니다", "결제가 완료됐습니다", "출고됐습니다"]),
    ("개인정보 요청",      ["주문번호를 입력해주세요", "연락처를 알려주세요", "주소를 입력해주세요"]),
]


# ==============================
# 도구 1: 앳플리 위키 검색
# ==============================
def search_atflee_wiki(question: str, top_k: int = 3) -> dict:
    """
    data/wiki 문서에서 질문과 관련 있는 문서를 검색한다.
    utils.rag_utils.search_wiki를 재사용한다.
    """
    results = search_wiki(question, top_k=top_k)

    formatted = []
    for rank, r in enumerate(results, start=1):
        formatted.append(
            {
                "rank":        rank,
                "source_file": r["file_name"],
                "score":       r["score"],
                "snippet":     r.get("snippet", "")[:300],
            }
        )

    return {
        "tool":    "search_atflee_wiki",
        "query":   question,
        "results": formatted,
        "count":   len(formatted),
    }


# ==============================
# 도구 2: 고객 의도 분류
# ==============================
def classify_customer_intent(message: str) -> dict:
    """
    고객 문의의 의도를 키워드 기반으로 분류한다.
    외부 API 없이 로컬 규칙으로 빠르게 판단한다.
    """
    message_lower = message.lower()

    detected_intents = []
    for intent_name, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in message_lower:
                detected_intents.append(intent_name)
                break

    if not detected_intents:
        detected_intents = ["기타"]

    primary_intent = detected_intents[0]

    high_severity_intents = {"AS", "제품 품질", "교환/반품"}
    severity = "높음" if primary_intent in high_severity_intents else "중간"

    needs_human = primary_intent in high_severity_intents

    return {
        "tool":              "classify_customer_intent",
        "intent":            primary_intent,
        "all_intents":       detected_intents,
        "severity":          severity,
        "needs_human_review": needs_human,
        "reason":            f"'{primary_intent}' 관련 키워드가 감지되었습니다.",
    }


# ==============================
# 도구 3: 답변 안전성 평가
# ==============================
def evaluate_safety(answer: str) -> dict:
    """
    Claude 답변에 위험 표현이 있는지 점검한다.
    실제 주문/배송/AS 상태를 단정하거나 개인정보를 요구하는 표현을 검사한다.
    """
    warnings = []

    for pattern_name, danger_phrases in SAFETY_DANGER_PATTERNS:
        for phrase in danger_phrases:
            if phrase in answer:
                warnings.append(f"[{pattern_name}] '{phrase}' 감지")
                break

    if "참고 문서" not in answer and "출처" not in answer and "source_file" not in answer:
        warnings.append("[참고 문서 누락] 답변에 참고 문서가 표시되지 않았습니다.")

    return {
        "tool":     "evaluate_safety",
        "safe":     len(warnings) == 0,
        "warnings": warnings,
    }


# ==============================
# ReAct 계획: 규칙 기반 도구 선택
# ==============================
def decide_action_plan(user_message: str, intent_result: dict) -> list:
    """
    사용자 메시지와 의도 분류 결과를 보고 실행할 도구 목록을 결정한다.

    규칙:
    - 항상: classify_customer_intent (이미 실행됨 → 이번엔 계획에만 표시)
    - 제품/앱/배송/교환/AS/고객센터 관련이면: search_atflee_wiki
    - 최종 답변 후 항상: evaluate_safety
    """
    search_needed_intents = {
        "앱 연결", "배송", "교환/반품", "AS", "제품 품질", "고객센터 문의"
    }

    plan = ["classify_customer_intent"]

    intent = intent_result.get("intent", "기타")
    if intent in search_needed_intents or intent == "기타":
        plan.append("search_atflee_wiki")

    plan.append("generate_final_answer")
    plan.append("evaluate_safety")

    return plan


# ==============================
# Thought 메시지 생성
# ==============================
def build_thought(user_message: str, intent_result: dict) -> str:
    """
    ReAct의 Thought 단계: 사용자 의도와 필요한 도구를 설명한다.
    """
    intent   = intent_result.get("intent", "기타")
    severity = intent_result.get("severity", "-")
    needs_hr = intent_result.get("needs_human_review", False)

    thought = (
        f"사용자 문의는 '{intent}' 유형으로 판단됩니다. "
        f"심각도: {severity}. "
    )

    if needs_hr:
        thought += "담당자 확인이 필요한 사안입니다. "

    thought += (
        "앳플리 위키 문서를 검색해 근거를 찾고, "
        "Claude가 안전하고 정확한 답변을 생성합니다."
    )

    return thought


# ==============================
# Claude 최종 답변 생성
# ==============================
def ask_claude_react_final_answer(
    user_message:  str,
    intent_result: dict,
    wiki_results:  dict,
) -> str:
    """
    ReAct 흐름의 Final Answer 단계.
    의도 분류 결과와 위키 검색 결과를 Claude에게 전달해 최종 답변을 생성한다.
    """
    intent_summary = (
        f"문의 유형: {intent_result.get('intent')}, "
        f"심각도: {intent_result.get('severity')}, "
        f"담당자 필요: {intent_result.get('needs_human_review')}"
    )

    wiki_context_parts = []
    for r in wiki_results.get("results", []):
        wiki_context_parts.append(
            f"[{r['rank']}위] {r['source_file']} (score={r['score']})\n{r['snippet']}"
        )
    wiki_context = "\n\n".join(wiki_context_parts) if wiki_context_parts else "검색 결과 없음"

    system_prompt = """
# Role
너는 앳플리 ReAct Agent v0이다.

# Goal
사용자 문의를 보고 내부 도구 결과를 참고해 안전하고 근거 기반으로 답변한다.

# Context
이 답변은 ReAct 흐름(Thought → Action → Observation → Final Answer)으로 생성된다.
도구 결과는 아래 <intent_result>와 <wiki_search_result>로 제공된다.

# Rules
- 제공된 도구 결과에 없는 내용은 단정하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 주문번호, 연락처, 주소 등 개인정보는 공개 채팅에 입력하지 않도록 안내한다.
- 교환/반품/AS 가능 여부는 고객센터나 마이페이지 확인이 필요하다고 안내한다.
- 답변 마지막에 참고 문서를 표시한다.
- 검색 결과가 없으면 "정확한 확인이 필요합니다"라고 말한다.

# Output Format
1. 간단한 답변
2. 판단한 문의 유형
3. 근거가 되는 앳플리 정보
4. 바로 해볼 수 있는 것
5. 확인이 필요한 것
6. 참고 문서
"""

    user_content = (
        f"<intent_result>\n{intent_summary}\n</intent_result>\n\n"
        f"<wiki_search_result>\n{wiki_context}\n</wiki_search_result>\n\n"
        f"<user_question>\n{user_message}\n</user_question>"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}]
    )

    return response.content[0].text


# ==============================
# ReAct 흐름 한 질문 실행
# ==============================
def run_react_agent(user_message: str) -> dict:
    """
    한 질문에 대해 ReAct 흐름 전체를 실행하고 결과를 반환한다.
    """
    separator = "=" * 70

    print(f"\n{separator}")
    print(f"[User]\n{user_message}")

    # ── Step 1: Action 1 — 의도 분류
    intent_result = classify_customer_intent(user_message)

    # ── Step 2: Thought — 판단 설명
    thought = build_thought(user_message, intent_result)
    print(f"\n[Thought]\n{thought}")

    # ── Step 3: Action Plan
    action_plan = decide_action_plan(user_message, intent_result)
    print(f"\n[Action Plan]")
    for step, tool in enumerate(action_plan, start=1):
        print(f"  {step}. {tool}")

    # ── Step 4: Action 1 출력
    print(f"\n[Action 1] classify_customer_intent")
    print(f"  intent   : {intent_result['intent']}")
    print(f"  severity : {intent_result['severity']}")
    print(f"  needs_hr : {intent_result['needs_human_review']}")
    print(f"  reason   : {intent_result['reason']}")

    # ── Step 5: Action 2 — 위키 검색
    wiki_results = search_atflee_wiki(user_message, top_k=3)
    print(f"\n[Action 2] search_atflee_wiki")
    for r in wiki_results["results"]:
        print(f"  {r['rank']}위 {r['source_file']} (score={r['score']})")
        if r["snippet"]:
            snippet_preview = r["snippet"][:80].replace("\n", " ")
            print(f"       └ {snippet_preview}...")

    # ── Step 6: Observation
    observation = (
        f"의도 분류 완료 ('{intent_result['intent']}'), "
        f"위키 검색 완료 ({wiki_results['count']}개 문서 발견)."
    )
    if intent_result["needs_human_review"]:
        observation += " 담당자 확인이 필요한 사안입니다."
    print(f"\n[Observation]\n{observation}")

    # ── Step 7: Final Answer
    print(f"\n[Action 3] generate_final_answer")
    print("  Claude에게 의도 분류 + 위키 검색 결과를 전달해 답변 생성 중...")

    final_answer = ask_claude_react_final_answer(
        user_message, intent_result, wiki_results
    )

    print(f"\n[Final Answer]")
    try:
        print(final_answer)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(final_answer.encode(encoding, errors="replace").decode(encoding))

    # ── Step 8: Safety Check
    safety_result = evaluate_safety(final_answer)
    print(f"\n[Safety Check]")
    if safety_result["safe"]:
        print("  safe: True - 위험 표현 없음")
    else:
        print(f"  safe: False")
        for w in safety_result["warnings"]:
            print(f"  ⚠ {w}")

    print(separator)

    return {
        "user_message":  user_message,
        "thought":       thought,
        "action_plan":   action_plan,
        "intent_result": intent_result,
        "wiki_results":  wiki_results,
        "observation":   observation,
        "final_answer":  final_answer,
        "safety_check":  safety_result,
    }


# ==============================
# 리포트 저장
# ==============================
def save_report(all_results: list) -> str:
    """
    ReAct Agent 실행 결과를 reports/chapter11/ 에 JSON으로 저장한다.
    reports/는 Git에 포함하지 않는다.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"atflee_react_agent_basic_{timestamp}.json"
    filepath  = os.path.join(REPORTS_DIR, filename)

    report = {
        "generated_at": timestamp,
        "model":        CLAUDE_MODEL_NAME,
        "agent":        "앳플리 ReAct Agent v0",
        "results":      all_results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return filepath


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    load_dotenv()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("[오류] ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
        sys.exit(1)

    client = Anthropic(api_key=anthropic_key)

    print("[Chapter 11 선택 실습 11-1] 앳플리 ReAct Agent Basic")
    print("-" * 60)
    print(f"모델: {CLAUDE_MODEL_NAME}")
    print("API Key 로드 완료 (값은 보안상 출력하지 않습니다.)")
    print()
    print("ReAct 흐름: Thought → Action → Observation → Final Answer")
    print()

    test_messages = [
        "체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
        "제품이 불량 같고 교환하고 싶어요.",
        "AS 접수됐는지 확인해줘.",
        "고객센터 전화번호 알려줘.",
    ]

    all_results = []

    for message in test_messages:
        result = run_react_agent(message)
        all_results.append(result)

    # 리포트 저장
    report_path = save_report(all_results)
    print(f"\n[리포트 저장 완료] {report_path}")
    print("(reports/ 폴더는 Git에 포함되지 않습니다.)")

    print("\n[실습 완료]")
    print("ReAct 흐름 (Thought → Action → Observation → Final Answer)이 완성되었습니다.")
    print("다음 단계: 11-2 - Claude tool_use API로 MCP 기반 에이전트 구현")
