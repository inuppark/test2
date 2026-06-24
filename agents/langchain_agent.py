from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import os

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from agents.rag_agent import answer_with_rag

load_dotenv()

@tool
def get_project_status() -> str:
    """현재 AI Agent Portal 프로젝트 상태를 요약합니다."""
    return """
현재 프로젝트 상태:
- VOC Agent: 엑셀 VOC 분석, 분류, 심각도, 담당부서, 액션 추천 가능
- PDF Agent: PyMuPDF 기반 PDF 텍스트 전처리, 문서별 요약, 통합 요약 가능
- Meeting Agent: 회의록 요약, 결정사항, 액션아이템, 리스크 정리 가능
- Image Agent: 이미지와 고객 VOC를 함께 분석 가능
- RAG Agent: reports와 data/wiki를 검색해서 근거 기반 답변 가능
- LangChain Agent: Tool을 사용해 프로젝트 상태, 보고서, RAG 검색을 수행 가능
- Portal: 여러 Agent 통합 실행 및 로그 대시보드 제공
다음 후보:
- LangChain + RAG 통합
- LangGraph 멀티에이전트 흐름
- Agent 실행 자동화
"""

@tool
def list_recent_reports(limit: int = 10) -> str:
    """reports 폴더의 최근 생성 보고서 파일 목록을 보여줍니다."""
    report_dir = Path("reports")

    if not report_dir.exists():
        return "reports 폴더가 없습니다."

    files = sorted(
        [p for p in report_dir.iterdir() if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not files:
        return "최근 보고서 파일이 없습니다."

    rows = []

    for path in files[:limit]:
        rows.append(f"- {path.name} ({round(path.stat().st_size / 1024, 1)} KB)")

    return "\n".join(rows)

@tool
def read_latest_report(agent_type: str = "all") -> str:
    """
    reports 폴더에서 최신 보고서 파일 내용을 읽습니다.
    agent_type 예시: all, voc, pdf, meeting, image
    """
    report_dir = Path("reports")

    if not report_dir.exists():
        return "reports 폴더가 없습니다."

    files = [p for p in report_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

    if agent_type:
        keyword = agent_type.lower()

        if keyword != "all":
            files = [
                p for p in files
                if keyword in p.name.lower()
            ]

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        return f"{agent_type} 조건에 맞는 TXT 보고서가 없습니다."

    latest = files[0]

    try:
        text = latest.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = latest.read_text(encoding="utf-8-sig")

    max_chars = 6000

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...문서가 길어 앞부분만 읽었습니다."

    return f"""
파일명: {latest.name}
경로: {latest}
크기: {round(latest.stat().st_size / 1024, 1)} KB

[보고서 내용]
{text}
"""

@tool
def search_company_knowledge(question: str) -> str:
    """
    reports와 data/wiki를 RAG로 검색해 근거 기반 답변을 생성합니다.
    최근 보고서, 사내 기준, 고객 응대 기준, 제품 하자 처리, 배송, VOC 관련 질문에 사용합니다.
    """
    answer, search_results, confidence = answer_with_rag(
        question=question,
        top_k=5,
        search_scope="전체"
    )

    sources = []
    for item in search_results:
        sources.append(
            f"- {item['source_file']} / {item['source_group']} / {item['report_type']} / score={round(item['score'], 4)}"
        )

    source_text = "\n".join(sources)

    return f"""
[RAG 답변]
{answer}

[RAG 신뢰도]
- 신뢰도: {confidence.get("level")}
- 최고 유사도: {confidence.get("top_score")}
- 판단 이유: {confidence.get("reason")}

[검색 근거]
{source_text}
"""

@tool
def search_policy_only(question: str) -> str:
    """
    data/wiki 사내 기준 문서만 RAG로 검색합니다.
    정책, 기준, 절차, 담당부서, AS, 배송, 품질 처리 기준 질문에 사용합니다.
    """
    answer, search_results, confidence = answer_with_rag(
        question=question,
        top_k=5,
        search_scope="사내위키"
    )

    sources = []
    for item in search_results:
        sources.append(
            f"- {item['source_file']} / score={round(item['score'], 4)}"
        )

    return f"""
[사내 기준 문서 검색 답변]
{answer}

[신뢰도]
- 신뢰도: {confidence.get("level")}
- 최고 유사도: {confidence.get("top_score")}
- 판단 이유: {confidence.get("reason")}

[근거 문서]
{chr(10).join(sources)}
"""

@tool
def search_reports_only(question: str) -> str:
    """
    reports 업무 보고서만 RAG로 검색합니다.
    최근 VOC, PDF 요약, 회의록, 이미지 분석 결과 등 현재 업무 결과 질문에 사용합니다.
    """
    answer, search_results, confidence = answer_with_rag(
        question=question,
        top_k=5,
        search_scope="보고서 전체"
    )

    sources = []
    for item in search_results:
        sources.append(
            f"- {item['source_file']} / {item['report_type']} / score={round(item['score'], 4)}"
        )

    return f"""
[업무 보고서 검색 답변]
{answer}

[신뢰도]
- 신뢰도: {confidence.get("level")}
- 최고 유사도: {confidence.get("top_score")}
- 판단 이유: {confidence.get("reason")}

[근거 보고서]
{chr(10).join(sources)}
"""

@tool
def get_agent_guide(agent_name: str) -> str:
    """Agent 이름을 입력받아 사용 가이드를 제공합니다. 예: VOC, PDF, Meeting, Image, RAG, LangChain"""
    name = agent_name.lower()

    if "voc" in name:
        return """
VOC Agent 사용 가이드:
1. VOC 엑셀 파일을 업로드합니다.
2. VOC 또는 고객문의 컬럼을 자동 인식합니다.
3. VOC 분류, 심각도, 담당부서, 후속 액션을 생성합니다.
4. 보고서와 CSV를 다운로드합니다.
추천 데이터: 고객 문의, 불만, 리뷰, CS 상담 내역
"""

    if "pdf" in name:
        return """
PDF Agent 사용 가이드:
1. PDF 파일을 업로드합니다.
2. PyMuPDF로 텍스트를 추출합니다.
3. 문서별 요약과 여러 문서 통합 요약을 생성합니다.
4. 보고서를 다운로드합니다.
추천 데이터: 매뉴얼, 논문, 제안서, 내부 문서
"""

    if "meeting" in name or "회의" in name:
        return """
Meeting Agent 사용 가이드:
1. 회의록 텍스트를 입력하거나 TXT 파일을 업로드합니다.
2. 회의 요약, 결정사항, 액션아이템, 리스크를 정리합니다.
3. 액션아이템 CSV를 다운로드합니다.
추천 데이터: 회의록, 미팅 메모, 업무 논의 기록
"""

    if "image" in name or "이미지" in name:
        return """
Image Agent 사용 가이드:
1. 제품, 불량, 포장, 상세페이지 이미지를 업로드합니다.
2. 고객 VOC나 상황 설명을 함께 입력합니다.
3. 이미지와 텍스트를 비교 분석합니다.
4. CS/품질/마케팅/물류 관점의 후속 액션을 확인합니다.
추천 데이터: 제품 사진, 불량 사진, 포장 사진, 상세페이지 캡처
"""

    if "rag" in name:
        return """
RAG Agent 사용 가이드:
1. 사내위키 문서를 업로드합니다.
2. RAG 인덱스를 업데이트합니다.
3. 검색 범위를 선택합니다.
4. 질문하면 reports와 data/wiki에서 근거를 검색해 답변합니다.
추천 질문:
- 제품 하자 문의가 들어오면 어떤 절차로 처리해야 해?
- 최근 보고서와 사내 기준을 함께 봤을 때 우선 업무는 뭐야?
"""

    if "langchain" in name or "랭체인" in name:
        return """
LangChain Agent 사용 가이드:
1. 질문을 입력합니다.
2. Agent가 필요한 Tool을 선택합니다.
3. 프로젝트 상태, 보고서 목록, 보고서 내용, RAG 검색을 수행합니다.
추천 질문:
- 최근 보고서와 사내 기준을 함께 봐서 우선 처리 업무 알려줘
- 제품 하자 처리 기준 알려줘
- 최신 보고서 목록 보여줘
"""

    return "지원 Agent: VOC, PDF, Meeting, Image, RAG, LangChain"

def build_langchain_agent():
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    tools = [
        get_project_status,
        list_recent_reports,
        read_latest_report,
        search_company_knowledge,
        search_policy_only,
        search_reports_only,
        get_agent_guide
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
당신은 AI 업무 자동화 Agent Portal의 운영 도우미입니다.

역할:
- 사용자의 질문을 이해합니다.
- 필요하면 제공된 Tool을 사용합니다.
- 프로젝트 상태, 최근 보고서, Agent 사용법, RAG 지식 검색을 수행합니다.
- 답변은 한국어로 간결하고 실무적으로 작성합니다.

Tool 선택 기준:
- 프로젝트 상태 질문 → get_project_status
- 최근 보고서 목록 → list_recent_reports
- 최신 보고서 내용 → read_latest_report
- 회사 기준/정책/절차 질문 → search_policy_only
- 최근 업무 결과/보고서 질문 → search_reports_only
- 최근 업무 결과와 사내 기준을 함께 봐야 하는 질문 → search_company_knowledge
- Agent 사용법 질문 → get_agent_guide

중요:
- 보고서나 사내 기준 관련 질문은 반드시 RAG Tool을 우선 사용하세요.
- 모르는 내용은 추측하지 말고 확인 필요라고 말하세요.
- RAG 신뢰도가 낮으면 단정하지 말고 추가 문서 업로드나 질문 구체화를 제안하세요.
"""
    )

    return agent

def ask_langchain_agent(question: str) -> str:
    if not question or not question.strip():
        raise ValueError("질문이 비어 있습니다.")

    agent = build_langchain_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    messages = result.get("messages", [])

    if not messages:
        return "응답을 생성하지 못했습니다."

    return messages[-1].content

def save_langchain_log(question, answer):
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/langchain_agent_log.csv"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    import pandas as pd

    row = pd.DataFrame([{
        "run_time": run_time,
        "question": question,
        "answer_preview": answer[:200],
        "status": "success"
    }])

    if os.path.exists(log_path):
        old = pd.read_csv(log_path)
        new = pd.concat([old, row], ignore_index=True)
    else:
        new = row

    new.to_csv(log_path, index=False, encoding="utf-8-sig")

    return log_path
