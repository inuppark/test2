import os
import json
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

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
# Streamlit UI 시작
# =========================

st.title("앳플리 AX Console v0")
st.caption("VOC 분석, CS 답변 초안, AX 학습 챗봇을 하나의 화면에서 실험하는 초기 콘솔입니다.")

tab_chat, tab_voc, tab_cs = st.tabs(["AX 학습 챗봇", "VOC 분석", "CS 답변 초안"])


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
