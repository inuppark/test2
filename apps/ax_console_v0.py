import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

# Streamlit Cloud는 apps/ 하위를 실행 디렉토리로 삼아 프로젝트 루트를 못 찾는 경우가 있다.
# 이 파일 위치(apps/)의 한 단계 위가 프로젝트 루트이므로 sys.path에 명시적으로 추가한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# utils.rag_utils에서 앳플리 봇 RAG 공통 함수를 가져온다.
from utils.rag_utils import (
    search_wiki,
    build_rag_context,
    get_source_file_names,
    format_search_results_for_display,
    evaluate_rag_answer,
    TOP_K,
)

# utils.tool_agent_utils에서 Tool Use Agent 공통 함수를 가져온다.
from utils.tool_agent_utils import (
    run_structured_agent,
    save_structured_result,
    list_saved_results,
    load_saved_result,
)

# utils.vector_rag_utils에서 TF-IDF 벡터 검색 공통 함수를 가져온다. (Chapter 10-5)
from utils.vector_rag_utils import (
    search_similar_chunks,
    build_vector_rag_context,
    get_source_chunks,
    summarize_query_vector,
    format_vector_results_for_display,
)

# .env 파일 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="앳플리 AX Console v0",
    page_icon="🧭",
    layout="wide"
)

# API Key 가져오기
# 로컬: .env의 ANTHROPIC_API_KEY 사용
# Streamlit Cloud: st.secrets["ANTHROPIC_API_KEY"] 사용
def get_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    return os.getenv("ANTHROPIC_API_KEY")

api_key = get_api_key()

if not api_key:
    st.error("ANTHROPIC_API_KEY가 설정되어 있지 않습니다. 로컬에서는 .env, 배포 환경에서는 Streamlit Secrets에 등록해주세요.")
    st.stop()

# Claude 클라이언트 생성
client = Anthropic(api_key=api_key)
model_name = "claude-sonnet-4-5"


def clean_json_text(text):
    """
    Claude가 JSON을 코드블록으로 감싸서 반환하는 경우를 대비해
    ```json, ``` 같은 표시를 제거한다.
    """
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def call_claude(system_prompt, user_prompt, max_tokens=1200, temperature=0.3):
    """
    Claude API를 호출하는 공통 함수.
    """
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.content[0].text


def analyze_voc(customer_message):
    """
    고객 문의를 VOC 기준으로 분석하고 JSON으로 반환한다.
    """
    system_prompt = """
# Role
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

# Process
1. 고객 감정을 파악한다.
2. 이슈 유형을 분류한다.
3. 심각도를 판단한다.
4. 원인 후보를 도출한다.
5. 담당 부서를 판단한다.
6. 고객 응대 방향을 제안한다.
7. 내부 후속 액션을 제안한다.

# Output Format
{
  "issue_type": "이슈 유형",
  "severity": "낮음 | 중간 | 높음",
  "customer_emotion": "고객 감정",
  "possible_causes": ["원인 후보 1", "원인 후보 2", "원인 후보 3"],
  "owner_team": "담당 부서",
  "reply_direction": "고객 응대 방향",
  "next_action": "다음 액션",
  "needs_human_review": true 또는 false
}
"""

    few_shot_examples = """
예시 1:
고객 문의:
배송이 예정일보다 3일 늦어졌고, 알림도 받지 못했습니다.

좋은 출력:
{
  "issue_type": "배송 지연",
  "severity": "중간",
  "customer_emotion": "불만, 불안",
  "possible_causes": ["물류 지연 가능성", "배송 알림 누락 가능성", "주문량 증가 가능성"],
  "owner_team": "물류/CS",
  "reply_direction": "배송 지연에 대해 사과하고 현재 배송 상태와 예상 도착일을 안내한다.",
  "next_action": "배송 추적 정보를 확인한 뒤 고객에게 개별 안내한다.",
  "needs_human_review": false
}

예시 2:
고객 문의:
제품을 받았는데 포장이 찢어져 있고 본체에 흠집이 있습니다.

좋은 출력:
{
  "issue_type": "제품 파손/품질",
  "severity": "높음",
  "customer_emotion": "분노, 실망",
  "possible_causes": ["배송 중 파손 가능성", "포장재 부족 가능성", "출고 검수 누락 가능성"],
  "owner_team": "품질/물류/CS",
  "reply_direction": "불편에 대해 즉시 사과하고 교환 또는 환불 절차를 안내한다.",
  "next_action": "파손 사진을 확인한 뒤 교환 접수 또는 환불 절차를 진행한다.",
  "needs_human_review": true
}
"""

    user_prompt = f"""
{few_shot_examples}

이제 아래 고객 문의를 같은 기준으로 분석해라.

고객 문의:
{customer_message}
"""

    raw_text = call_claude(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1200,
        temperature=0
    )

    cleaned_text = clean_json_text(raw_text)
    return json.loads(cleaned_text), raw_text


def generate_cs_reply(customer_message):
    """
    고객 문의를 바탕으로 CS 답변 초안을 생성한다.
    """
    system_prompt = """
# Role
너는 앳플리 고객센터 매니저다.

# Goal
고객 문의에 대해 정중하고 실무적인 CS 답변 초안을 작성한다.

# Context
앳플리는 헬스케어 제품과 앱을 운영하는 회사다.
고객은 배송, 앱 연결, 제품 품질, AS 답변 지연 등으로 불편을 겪을 수 있다.

# Rules
- 고객 불편에 먼저 공감한다.
- 필요한 경우 사과 표현을 포함한다.
- 고객이 말하지 않은 사실을 확정하지 않는다.
- 원인은 단정하지 말고 확인이 필요하다고 표현한다.
- 실제 주문번호, 배송상태, AS 접수번호 등 확인되지 않은 정보는 지어내지 않는다.
- 고객에게 다음 절차를 명확히 안내한다.
- 답변은 고객에게 바로 보낼 수 있는 톤으로 작성한다.

# Process
1. 고객의 불편 상황을 요약한다.
2. 공감과 사과를 표현한다.
3. 확인이 필요한 정보를 안내한다.
4. 다음 처리 절차를 설명한다.
5. 정중하게 마무리한다.

# Output Format
고객에게 보낼 수 있는 답변문 형태로 작성한다.
"""

    user_prompt = f"""
아래 고객 문의에 대한 CS 답변 초안을 작성해줘.

고객 문의:
{customer_message}
"""

    return call_claude(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1000,
        temperature=0.3
    )


atflee_system_prompt = """
# Role
너는 앳플리 위키 기반 RAG 답변 봇이다.

# Goal
사용자 질문에 대해 검색된 앳플리 위키 문서를 근거로 정확하고 친절하게 답변한다.

# Context
너에게 제공되는 <rag_context>는 data/wiki 문서 중 사용자 질문과 관련도가 높은 문서만 검색해서 가져온 것이다.
너는 이 문서 내용을 우선 근거로 사용한다.

# Rules
- <rag_context>에 있는 정보만 확정적으로 말한다.
- <rag_context>에 없는 내용은 추측하지 않는다.
- <rag_context> 안에 [관련 스니펫]이 포함되어 있으면 해당 부분을 우선 참고한다.
- 단, 스니펫만 보고 단정하지 말고 [문서 전체] 내용과 함께 판단한다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
- 정책, 보증, 교환/환불, AS 조건은 확실하지 않으면 "정확한 확인이 필요합니다"라고 말한다.
- 개인정보, 주문번호, 연락처 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
- 고객이 바로 해볼 수 있는 다음 행동을 안내한다.
- 답변은 초보자도 이해할 수 있게 쉽게 작성한다.
- 제품 사용법은 가능한 경우 사용자 매뉴얼과 문제 해결 FAQ 확인을 함께 안내한다.

# Process
1. 사용자 질문의 의도를 파악한다.
2. <rag_context>의 [관련 스니펫]을 먼저 확인하고, [문서 전체]와 함께 근거를 찾는다.
3. 확실한 정보와 확인이 필요한 정보를 구분한다.
4. 사용자가 바로 할 수 있는 행동을 안내한다.
5. 마지막에 참고한 문서명을 표시한다.

# Output Format
아래 형식으로 답변한다.

1. 간단한 답변
2. 근거가 되는 앳플리 위키 정보
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 문서
"""


def build_atflee_messages(user_question, rag_context, source_files):
    """
    앳플리 봇 탭용 messages를 구성한다.
    rag_context를 cache_control이 붙은 별도 text block으로 분리해
    반복 사용 시 Prompt Caching 효과를 얻는다.
    이전 대화 내역도 함께 전달해 맥락을 유지한다.
    """
    messages = []

    # 이전 대화 내역은 string content 그대로 전달한다.
    for message in st.session_state.atflee_bot_messages:
        messages.append(message)

    # 현재 메시지: rag_context는 캐싱 대상 블록, 질문은 일반 블록으로 분리한다.
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"<rag_context>\n{rag_context}\n</rag_context>",
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": (
                    f"<source_files>\n{', '.join(source_files)}\n</source_files>\n\n"
                    f"<user_question>\n{user_question}\n</user_question>"
                )
            }
        ]
    })

    return messages


def ask_atflee_bot(user_question, rag_context, source_files):
    """
    RAG로 검색된 문서를 Context로 Claude에게 전달하고 (답변 텍스트, usage dict)를 반환한다.
    """
    response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.2,
        system=atflee_system_prompt,
        messages=build_atflee_messages(user_question, rag_context, source_files)
    )

    return response.content[0].text, get_usage_dict(response.usage)


def call_ax_tutor(messages):
    """
    대화 기억이 있는 앳플리 AX 학습 챗봇용 Claude 호출 함수.
    """
    system_prompt = """
# Role
너는 앳플리 AX 프로젝트를 함께 학습하고 설계하는 AI 튜터이자 실무 파트너다.

# Goal
사용자가 Claude API, Streamlit, VOC Agent, RAG, LangGraph, AI-native 회사 전환을 이해하고 직접 구현할 수 있도록 돕는다.

# Context
사용자는 "조코딩의 바이브코딩 1인창업" 책을 따라 실습하고 있다.
최종 목표는 앳플리를 AI-native 회사로 전환하는 것이다.
현재는 Claude API와 Streamlit을 활용해 VOC Agent와 AX 실습 환경을 만들고 있다.

# Rules
- 초보자도 이해할 수 있게 쉽게 설명한다.
- 가능한 한 앳플리 AX 프로젝트와 연결해서 설명한다.
- 사용자가 바로 실행할 수 있는 코드나 명령어를 우선 제공한다.
- 모르는 사실은 확정하지 않고 확인이 필요하다고 말한다.
- 답변은 너무 길지 않게 핵심부터 설명한다.

# Process
1. 사용자의 질문 의도를 파악한다.
2. 현재 학습 단계와 앳플리 AX 목표를 연결한다.
3. 필요한 경우 쉬운 비유를 사용한다.
4. 실행 가능한 다음 단계를 제안한다.
"""

    response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.3,
        system=system_prompt,
        messages=messages
    )

    return response.content[0].text


# =========================
# 벡터 RAG Claude 답변 함수 (Chapter 10-6)
# =========================

def ask_claude_with_vector_rag(question, vector_results):
    """
    TF-IDF 벡터 검색 결과를 Context로 Claude에게 답변을 요청한다.
    build_vector_rag_context는 utils.vector_rag_utils에서 가져온다.
    """
    vector_rag_context = build_vector_rag_context(vector_results)

    system_prompt = """
# Role
너는 앳플리 벡터 RAG 답변 봇이다.

# Goal
사용자 질문에 대해 TF-IDF 벡터 검색으로 찾은 앳플리 위키 청크를 근거로 쉽고 안전하게 답변한다.

# Rules
* <vector_rag_context>에 있는 정보만 확정적으로 말한다.
* Context에 없는 내용은 추측하지 않는다.
* 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
* 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
* 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
* 답변 마지막에는 참고한 source_file과 chunk_id를 표시한다.
* 검색 결과의 유사도가 낮으면 "정확한 확인이 필요합니다"라고 안내한다.

# Output Format
아래 형식으로 답변한다.

1. 간단한 답변
2. 근거가 되는 앳플리 위키 청크
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 청크
"""

    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"<vector_rag_context>\n{vector_rag_context}\n</vector_rag_context>\n\n"
                    f"<user_question>\n{question}\n</user_question>"
                ),
            }
        ],
    )

    return response.content[0].text


# =========================
# Prompt Caching 헬퍼
# =========================

def get_usage_dict(usage):
    """response.usage 에서 캐싱 관련 필드를 안전하게 꺼낸다."""
    return {
        "input_tokens":                getattr(usage, "input_tokens",                None),
        "output_tokens":               getattr(usage, "output_tokens",               None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens",     None),
    }


def render_usage_expander(usage_dict):
    """캐싱 사용량을 expander로 표시한다."""
    with st.expander("프롬프트 캐싱 사용량"):
        for key, value in usage_dict.items():
            if value is None:
                st.write(f"- {key}: 지원되지 않거나 없음")
            else:
                st.write(f"- {key}: {value}")


# =========================
# Tool Use Agent 결과 렌더링 헬퍼
# =========================

def render_tool_agent_result(result):
    """핵심 요약 카드 → 요약 → 고객 답변 방향 → 내부 다음 액션 → 안전 메모 → 도구/문서."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("이슈 유형", result.get("issue_type", "-"))
    col2.metric("심각도", result.get("severity", "-"))
    col3.metric("담당팀", result.get("owner_team", "-"))
    col4.metric("사람 검토", "필요" if result.get("needs_human_review") else "불필요")

    if result.get("severity") == "높음":
        st.warning("심각도 높음 — 우선 처리가 필요합니다.")
    if result.get("needs_human_review"):
        st.warning("담당자 확인 필요")

    summary = result.get("summary", "")
    if summary:
        st.info(summary)

    st.subheader("고객 답변 방향")
    st.write(result.get("customer_reply_direction", "-"))

    st.subheader("내부 다음 액션")
    st.write(result.get("internal_next_action", "-"))

    safety_notes = result.get("safety_notes", [])
    if safety_notes:
        st.subheader("안전 메모")
        for note in safety_notes:
            st.warning(note)

    with st.expander("사용 도구 및 참고 문서"):
        st.markdown("**사용 도구**")
        used_tools = result.get("used_tools", [])
        if used_tools:
            for tool in used_tools:
                st.write(f"- {tool}")
        else:
            st.write("-")
        st.markdown("**참고 문서**")
        source_files = result.get("source_files", [])
        if source_files:
            for f in source_files:
                st.write(f"- {f}")
        else:
            st.write("-")


def render_tool_logs(tool_logs):
    """도구 실행 로그를 expander 안에 라운드별로 렌더링한다."""
    if not tool_logs:
        return
    with st.expander("도구 실행 로그"):
        for log in tool_logs:
            st.markdown(f"**라운드 {log['round']} — {log['tool_name']}**")
            st.markdown("입력")
            st.json(log.get("tool_input", {}))
            st.markdown("결과")
            tool_result = log.get("tool_result", {})
            if isinstance(tool_result, dict):
                display = {}
                for k, v in tool_result.items():
                    if k == "rag_context" and isinstance(v, str) and len(v) > 500:
                        display[k] = v[:500] + " ...(이하 생략)"
                    else:
                        display[k] = v
                st.json(display)
            else:
                st.code(str(tool_result)[:1200])
            st.divider()


def render_saved_result_detail(saved_data, file_name="result.json", download_key_prefix="history"):
    """저장 결과 파일을 실행 결과와 같은 구조로 렌더링한다."""
    result_data = saved_data.get("result") or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("이슈 유형", result_data.get("issue_type", "-"))
    col2.metric("심각도", result_data.get("severity", "-"))
    col3.metric("담당팀", result_data.get("owner_team", "-"))
    col4.metric("사람 검토", "필요" if result_data.get("needs_human_review") else "불필요")

    st.caption(
        f"저장 일시: {saved_data.get('created_at', '-')}  |  "
        f"에이전트: {saved_data.get('agent_name', '-')}"
    )

    if result_data.get("severity") == "높음":
        st.warning("심각도 높음 — 우선 처리가 필요합니다.")
    if result_data.get("needs_human_review"):
        st.warning("담당자 확인 필요")

    summary = result_data.get("summary", "")
    if summary:
        st.info(summary)

    st.subheader("고객 답변 방향")
    st.write(result_data.get("customer_reply_direction", "-"))

    st.subheader("내부 다음 액션")
    st.write(result_data.get("internal_next_action", "-"))

    safety_notes = result_data.get("safety_notes", [])
    if safety_notes:
        st.subheader("안전 메모")
        for note in safety_notes:
            st.warning(note)

    with st.expander("사용 도구 및 참고 문서"):
        st.markdown("**사용 도구**")
        used_tools = result_data.get("used_tools", [])
        if used_tools:
            for tool in used_tools:
                st.write(f"- {tool}")
        else:
            st.write("-")
        st.markdown("**참고 문서**")
        source_files = result_data.get("source_files", [])
        if source_files:
            for f in source_files:
                st.write(f"- {f}")
        else:
            st.write("-")

    with st.expander("원본 고객 문의"):
        st.write(saved_data.get("user_question", "-"))

    with st.expander("전체 JSON"):
        st.json(saved_data)

    st.download_button(
        label="이 결과 JSON 다운로드",
        data=json.dumps(saved_data, ensure_ascii=False, indent=2),
        file_name=file_name,
        mime="application/json",
        key=f"{download_key_prefix}_download"
    )


# =========================
# Streamlit UI 시작
# =========================

st.title("앳플리 AX Console v0")
st.caption("VOC 분석, CS 답변 초안, AX 학습 챗봇을 하나의 화면에서 실험하는 초기 콘솔입니다.")

tab_chat, tab_voc, tab_cs, tab_atflee, tab_tool_agent, tab_vector_rag = st.tabs(
    ["AX 학습 챗봇", "VOC 분석", "CS 답변 초안", "앳플리 봇", "Tool Use Agent", "벡터 RAG"]
)


# =========================
# 1. AX 학습 챗봇 탭
# =========================
with tab_chat:
    st.subheader("AX 학습 챗봇")
    st.write("Claude API, Streamlit, VOC Agent, RAG, LangGraph, 앳플리 AX 프로젝트에 대해 질문할 수 있습니다.")

    if "ax_chat_messages" not in st.session_state:
        st.session_state.ax_chat_messages = []

    with st.sidebar:
        st.header("AX Console 설정")
        st.write("AX 챗봇 대화 수:", len(st.session_state.ax_chat_messages))

        if st.button("AX 챗봇 대화 초기화"):
            st.session_state.ax_chat_messages = []
            st.rerun()

    for message in st.session_state.ax_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("앳플리 AX에 대해 질문해보세요")

    if user_input:
        st.session_state.ax_chat_messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Claude가 답변 중입니다..."):
                try:
                    assistant_reply = call_ax_tutor(st.session_state.ax_chat_messages)
                    st.write(assistant_reply)

                    st.session_state.ax_chat_messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_reply
                        }
                    )
                except Exception as error:
                    st.error(f"Claude 호출 중 오류가 발생했습니다: {error}")


# =========================
# 2. VOC 분석 탭
# =========================
with tab_voc:
    st.subheader("VOC 분석")
    st.write("고객 문의를 입력하면 이슈 유형, 심각도, 담당 부서, 다음 액션을 분석합니다.")

    sample_voc = """체중계가 앱이랑 계속 연결이 안 됩니다.
AS 문의도 남겼는데 답변이 늦어서 너무 답답합니다."""

    customer_message = st.text_area(
        "고객 문의 입력",
        value=sample_voc,
        height=160,
        key="voc_customer_message"
    )

    if st.button("VOC 분석하기", type="primary"):
        if not customer_message.strip():
            st.warning("고객 문의를 입력해주세요.")
        else:
            with st.spinner("VOC를 분석하고 있습니다..."):
                try:
                    result, raw_text = analyze_voc(customer_message)

                    severity = result.get("severity", "")
                    needs_human_review = result.get("needs_human_review", False)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("이슈 유형", result.get("issue_type", "-"))

                    with col2:
                        st.metric("심각도", severity)

                    with col3:
                        st.metric("담당 부서", result.get("owner_team", "-"))

                    if severity == "높음" or needs_human_review is True:
                        st.warning("담당자 확인 필요")
                    else:
                        st.success("일반 처리 가능")

                    st.markdown("### 고객 감정")
                    st.write(result.get("customer_emotion", "-"))

                    st.markdown("### 원인 후보")
                    possible_causes = result.get("possible_causes", [])
                    if isinstance(possible_causes, list):
                        for cause in possible_causes:
                            st.write(f"- {cause}")
                    else:
                        st.write(possible_causes)

                    st.markdown("### 고객 응대 방향")
                    st.write(result.get("reply_direction", "-"))

                    st.markdown("### 다음 액션")
                    st.write(result.get("next_action", "-"))

                    with st.expander("원본 JSON 보기"):
                        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

                    with st.expander("Claude 원본 응답 보기"):
                        st.code(raw_text)

                except Exception as error:
                    st.error(f"분석 중 오류가 발생했습니다: {error}")


# =========================
# 3. CS 답변 초안 탭
# =========================
with tab_cs:
    st.subheader("CS 답변 초안")
    st.write("고객 문의를 바탕으로 고객에게 보낼 수 있는 정중한 답변 초안을 생성합니다.")

    sample_cs = """체중계가 앱이랑 계속 연결이 안 됩니다.
AS 문의도 남겼는데 답변이 늦어서 너무 답답합니다."""

    cs_customer_message = st.text_area(
        "고객 문의 입력",
        value=sample_cs,
        height=160,
        key="cs_customer_message"
    )

    if st.button("CS 답변 생성하기", type="primary"):
        if not cs_customer_message.strip():
            st.warning("고객 문의를 입력해주세요.")
        else:
            with st.spinner("CS 답변 초안을 생성하고 있습니다..."):
                try:
                    reply = generate_cs_reply(cs_customer_message)

                    st.markdown("### CS 답변 초안")
                    st.write(reply)

                except Exception as error:
                    st.error(f"CS 답변 생성 중 오류가 발생했습니다: {error}")


# =========================
# 4. 앳플리 봇 탭
# =========================
with tab_atflee:
    st.subheader("앳플리 봇")
    st.write("질문과 관련 있는 data/wiki 문서를 검색해 답변합니다.")
    st.info("현재 앳플리 봇은 data/wiki 문서와 공식몰 공개 정보를 바탕으로 답변합니다.")

    if "atflee_bot_messages" not in st.session_state:
        st.session_state.atflee_bot_messages = []

    if st.button("앳플리 봇 대화 초기화", key="atflee_reset"):
        st.session_state.atflee_bot_messages = []
        st.rerun()

    for message in st.session_state.atflee_bot_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    atflee_user_input = st.chat_input("앳플리 제품/앱/정책에 대해 질문해보세요", key="atflee_chat_input")

    if atflee_user_input:
        st.session_state.atflee_bot_messages.append(
            {
                "role": "user",
                "content": atflee_user_input
            }
        )

        with st.chat_message("user"):
            st.write(atflee_user_input)

        with st.chat_message("assistant"):
            with st.spinner("앳플리 봇이 답변 중입니다..."):
                try:
                    # 1단계: 질문과 관련 있는 문서를 검색한다.
                    atflee_search_results = search_wiki(atflee_user_input, top_k=TOP_K)

                    if not atflee_search_results:
                        st.warning("data/wiki 폴더에 문서가 없어 답변을 생성할 수 없습니다.")
                    else:
                        # 2단계: 검색된 문서만 RAG Context로 합친다.
                        atflee_rag_context = build_rag_context(atflee_search_results)
                        atflee_source_files = get_source_file_names(atflee_search_results)

                        # 검색된 참고 문서와 관련 스니펫을 expander로 표시한다.
                        with st.expander("검색된 참고 문서"):
                            for result in atflee_search_results:
                                st.markdown(f"**{result['file_name']}** / 점수: {result['score']}")
                                st.caption(result.get("snippet", ""))

                        # 3단계: 검색된 문서를 Claude에게 전달해 답변을 받는다.
                        atflee_reply, atflee_usage = ask_atflee_bot(
                            atflee_user_input, atflee_rag_context, atflee_source_files
                        )
                        st.write(atflee_reply)

                        # 4단계: 답변 품질을 체크한다.
                        atflee_eval = evaluate_rag_answer(atflee_reply, atflee_source_files)
                        with st.expander("답변 품질 체크"):
                            st.write(f"점수: {atflee_eval['score']}점 / 상태: {atflee_eval['status']}")
                            for check in atflee_eval["checks"]:
                                if check["passed"]:
                                    st.success(f"✔ {check['name']}: {check['message']}")
                                else:
                                    st.warning(f"△ {check['name']}: {check['message']}")

                        # 5단계: 프롬프트 캐싱 사용량을 표시한다.
                        render_usage_expander(atflee_usage)
                        st.caption(
                            "반복되는 문서 Context는 Prompt Caching을 통해 비용 효율을 높일 수 있습니다."
                        )

                        st.session_state.atflee_bot_messages.append(
                            {
                                "role": "assistant",
                                "content": atflee_reply
                            }
                        )

                except Exception as error:
                    st.error(f"앳플리 봇 답변 생성 중 오류가 발생했습니다: {error}")


# =========================
# 5. Tool Use Agent 탭
# =========================
with tab_tool_agent:
    st.subheader("앳플리 Tool Use Agent")
    st.write(
        "고객 문의를 입력하면 Claude가 필요한 도구를 사용해 "
        "VOC 분류, 앳플리 위키 검색, 구조화 JSON 생성을 수행합니다."
    )

    _DEFAULT_QUESTION = """고객이 이렇게 문의했습니다.

"앳플리 체중계가 앱이랑 계속 연결이 안 되고,
AS 문의를 남겼는데 답변이 너무 늦어서 화가 납니다."

이 문의를 처리해서 업무 시스템에 넣을 수 있는 JSON으로 정리해줘."""

    agent_user_question = st.text_area(
        "고객 문의 입력",
        value=_DEFAULT_QUESTION,
        height=180,
        key="tool_agent_question"
    )

    if st.button("Tool Use Agent 실행", type="primary", key="tool_agent_run"):
        if not agent_user_question.strip():
            st.warning("고객 문의를 입력해주세요.")
        else:
            with st.spinner("Tool Use Agent가 도구를 사용해 처리 중입니다..."):
                try:
                    agent_output = run_structured_agent(
                        client, model_name, agent_user_question
                    )
                except Exception as error:
                    agent_output = {
                        "result":     None,
                        "raw_text":   "",
                        "used_tools": [],
                        "tool_logs":  [],
                        "error":      str(error)
                    }

            # 1. 실행 성공/실패 메시지
            if agent_output.get("error"):
                st.error(f"오류: {agent_output['error']}")

            result = agent_output.get("result")

            if result:
                st.success("분석 완료 — JSON 파싱 성공")

                # 2~7. 핵심 요약 카드 → 요약 → 고객 답변 방향 → 내부 액션 → 안전 메모 → 도구/문서
                render_tool_agent_result(result)

                # 8. 다운로드/저장 버튼 같은 줄 배치
                st.markdown("---")
                col_dl, col_save = st.columns(2)
                with col_dl:
                    st.download_button(
                        label="JSON 다운로드",
                        data=json.dumps(result, ensure_ascii=False, indent=2),
                        file_name="atflee_tool_result.json",
                        mime="application/json",
                        key="tool_agent_download"
                    )
                with col_save:
                    if st.button("reports 폴더에 저장", key="tool_agent_save"):
                        try:
                            saved_path = save_structured_result(
                                _PROJECT_ROOT, result, agent_user_question
                            )
                            st.success(f"저장 완료: {saved_path}")
                            st.info(
                                "Streamlit Cloud에서는 파일이 영구 저장되지 않을 수 있습니다. "
                                "로컬 실행 시에는 정상 저장됩니다. "
                                "JSON 다운로드 버튼을 기본으로 사용하세요."
                            )
                        except Exception as save_error:
                            st.error(f"저장 오류: {save_error}")

            else:
                st.warning("JSON 파싱에 실패했습니다. Claude 원문을 확인하세요.")
                raw_text_fail = agent_output.get("raw_text", "")
                if raw_text_fail:
                    st.text(raw_text_fail)

            # 9. 도구 실행 로그 expander
            render_tool_logs(agent_output.get("tool_logs", []))

            # 10. Claude 최종 원문 expander
            raw_text = agent_output.get("raw_text", "")
            if raw_text:
                with st.expander("Claude 최종 원문"):
                    st.code(raw_text, language="json")

            # 11. 프롬프트 캐싱 사용량 expander
            usage_logs = agent_output.get("usage_logs", [])
            with st.expander("프롬프트 캐싱 사용량"):
                if not usage_logs:
                    st.write("아직 usage 정보가 없습니다.")
                else:
                    any_cache_read = False
                    for log in usage_logs:
                        cache_read = log.get("cache_read_input_tokens") or 0
                        if cache_read > 0:
                            any_cache_read = True
                        st.write(f"**라운드 {log.get('round', '?')}**")
                        st.json({
                            "input_tokens":                log.get("input_tokens"),
                            "output_tokens":               log.get("output_tokens"),
                            "cache_creation_input_tokens": log.get("cache_creation_input_tokens"),
                            "cache_read_input_tokens":     log.get("cache_read_input_tokens"),
                        })
                    if any_cache_read:
                        st.success("캐시 읽기 사용 확인")

    # 보안 안내
    st.caption("주문번호, 연락처, 주소 등 개인정보는 공개 채팅에 입력하지 마세요.")

    # =========================
    # 저장된 처리 이력 섹션
    # =========================
    st.markdown("---")
    st.subheader("저장된 처리 이력")
    st.write("reports/chapter08에 저장된 Tool Use Agent JSON 결과를 다시 확인할 수 있습니다.")
    st.info(
        "Streamlit Cloud에서는 reports 폴더 저장 결과가 영구 보관되지 않을 수 있습니다. "
        "중요한 결과는 JSON 다운로드 버튼으로 따로 저장하세요."
    )

    saved_results = list_saved_results(_PROJECT_ROOT)

    if not saved_results:
        st.info("아직 저장된 Tool Use Agent 결과가 없습니다.")
    else:
        def _make_label(item):
            parts = [
                item.get("created_at", ""),
                item.get("issue_type", "-"),
                item.get("severity", "-"),
                item.get("owner_team", "-"),
                item.get("file_name", ""),
            ]
            return " | ".join(parts)

        options       = saved_results
        option_labels = [_make_label(item) for item in options]

        selected_label = st.selectbox(
            "결과 파일 선택",
            option_labels,
            key="history_selectbox"
        )

        selected_index = option_labels.index(selected_label)
        selected_item  = options[selected_index]

        saved_data = load_saved_result(selected_item["file_path"])

        if saved_data is None:
            st.error("파일을 읽는 데 실패했습니다.")
        else:
            st.subheader("저장 결과 상세")
            render_saved_result_detail(
                saved_data,
                file_name=selected_item["file_name"],
                download_key_prefix="history"
            )


# =========================
# 6. 벡터 RAG 탭 (Chapter 10-6)
# =========================
with tab_vector_rag:
    st.subheader("앳플리 벡터 RAG")
    st.write(
        "TF-IDF 벡터 검색으로 data/wiki 청크를 찾고, "
        "선택된 청크를 근거로 Claude가 답변합니다."
    )
    st.caption(
        "현재 버전은 학습용 TF-IDF 벡터 검색입니다. "
        "실제 의미 기반 검색은 추후 임베딩 API로 고도화할 수 있습니다."
    )

    # 벡터 인덱스 존재 여부 확인
    _VECTOR_INDEX_PATH = os.path.join(_PROJECT_ROOT, "data", "rag", "atflee_tfidf_vector_index.json")

    if not os.path.exists(_VECTOR_INDEX_PATH):
        st.error(
            "벡터 인덱스가 없습니다. 먼저 Chapter 10-1, 10-2를 실행해 인덱스를 생성하세요."
        )
        st.code(
            "python chapters/chapter10/01_atflee_chunk_wiki_documents.py\n"
            "python chapters/chapter10/02_atflee_build_tfidf_vector_index.py",
            language="bash"
        )
    else:
        # 인덱스를 세션 캐시에 올려두어 매 질문마다 파일을 다시 읽지 않는다.
        if "vector_index_payload" not in st.session_state:
            import json as _json
            with open(_VECTOR_INDEX_PATH, "r", encoding="utf-8") as _f:
                st.session_state.vector_index_payload = _json.load(_f)

        _index = st.session_state.vector_index_payload
        st.caption(
            f"인덱스 로드 완료 — 청크 수: {_index.get('chunk_count')} / "
            f"Vocabulary: {_index.get('vocabulary_size')} / "
            f"생성: {_index.get('created_at')}"
        )

        # 질문 입력
        vector_rag_question = st.text_input(
            "질문을 입력하세요",
            value="체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?",
            key="vector_rag_question"
        )

        if st.button("벡터 RAG 답변 생성", type="primary", key="vector_rag_run"):
            if not vector_rag_question.strip():
                st.warning("질문을 입력해주세요.")
            else:
                with st.spinner("벡터 검색 및 Claude 답변 생성 중..."):
                    try:
                        # 1단계: TF-IDF 벡터 검색
                        search_output = search_similar_chunks(
                            vector_rag_question,
                            index_payload=_index,
                            top_k=3,
                        )

                        _error        = search_output.get("error")
                        _vr_results   = search_output.get("results", [])
                        _query_vector = search_output.get("query_vector", {})

                        if _error:
                            st.error(f"벡터 검색 오류: {_error}")
                        else:
                            # 2단계: 질문 벡터 토큰 표시
                            _top_tokens = summarize_query_vector(_query_vector, top_n=10)
                            with st.expander("질문 벡터 토큰 (상위 10개)"):
                                if _top_tokens:
                                    for _tok, _w in _top_tokens:
                                        st.write(f"- {_tok}: {_w:.4f}")
                                else:
                                    st.warning(
                                        "질문 벡터가 비어 있습니다. "
                                        "인덱스 vocabulary에 없는 단어만 포함된 질문일 수 있습니다."
                                    )

                            # 3단계: 벡터 검색 결과 TOP 3 표시
                            _max_sim = max((r["similarity"] for r in _vr_results), default=0)
                            if _max_sim < 0.05:
                                st.warning(
                                    "검색 유사도가 낮습니다. 답변 품질이 낮을 수 있습니다."
                                )

                            with st.expander("벡터 검색 TOP 3"):
                                for _rank, _res in enumerate(_vr_results, start=1):
                                    st.markdown(
                                        f"**{_rank}위** `{_res['source_file']}` / "
                                        f"`{_res['chunk_id']}` / 유사도: `{_res['similarity']:.4f}`"
                                    )
                                    st.caption(_res["text"][:300])
                                    st.divider()

                            # 4단계: 키워드 검색 vs 벡터 검색 비교
                            with st.expander("키워드 검색 vs 벡터 검색 비교"):
                                _kw_col, _vr_col = st.columns(2)

                                with _kw_col:
                                    st.markdown("**키워드 검색 TOP 3**")
                                    try:
                                        _kw_results = search_wiki(vector_rag_question, top_k=3)
                                        if _kw_results:
                                            for _kr in _kw_results:
                                                st.write(
                                                    f"- {_kr.get('file_name', '-')} "
                                                    f"(score: {_kr.get('score', 0)})"
                                                )
                                        else:
                                            st.write("결과 없음")
                                    except Exception as _kw_err:
                                        st.write(f"키워드 검색 오류: {_kw_err}")

                                with _vr_col:
                                    st.markdown("**벡터 검색 TOP 3**")
                                    for _vline in format_vector_results_for_display(_vr_results):
                                        st.write(_vline)

                            # 5단계: Claude 답변 생성
                            st.markdown("### Claude 벡터 RAG 답변")
                            _vr_answer = ask_claude_with_vector_rag(
                                vector_rag_question, _vr_results
                            )
                            st.write(_vr_answer)

                    except Exception as _vr_err:
                        st.error(f"벡터 RAG 처리 중 오류가 발생했습니다: {_vr_err}")

        # 보안 안내
        st.caption("주문번호, 연락처, 주소 등 개인정보는 공개 채팅에 입력하지 마세요.")
