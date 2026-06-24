import streamlit as st
import pandas as pd
import os
from pathlib import Path
import runpy

st.set_page_config(
    page_title="AI 업무 자동화 Agent Portal",
    layout="wide"
)

APP_MAP = {
    "VOC Agent": "ui/app.py",
    "PDF Agent": "ui/pdf_app.py",
    "Meeting Agent": "ui/meeting_app.py",
    "Image Agent": "ui/image_app.py",
    "RAG Agent": "ui/rag_app.py",
    "LangChain Agent": "ui/langchain_app.py",
    "LangGraph Agent": "ui/langgraph_app.py",
}

def load_log(agent_name, path):
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["agent"] = agent_name
    return df

def render_home():
    st.title("AI 업무 자동화 Agent Portal")
    st.write("업무 Agent, 지식 Agent, 운영 Agent, 워크플로 Agent를 한 곳에서 실행합니다.")

    st.divider()

    st.subheader("AI 업무 자동화 Agent Portal 구조")

    st.code(
"""Agent Portal
├─ 업무 Agent
│  ├─ VOC Agent
│  ├─ PDF Agent
│  ├─ Meeting Agent
│  └─ Image Agent
│
├─ 지식 Agent
│  └─ RAG Agent
│
├─ 운영 Agent
│  └─ LangChain Agent
│
└─ 워크플로 Agent
   └─ LangGraph Agent""",
        language="text"
    )

    st.divider()

    st.subheader("Agent 설명")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 업무 Agent")
        st.markdown("""
        - **VOC Agent**: VOC 엑셀 분석, 심각도, 담당부서, 액션 추천
        - **PDF Agent**: PyMuPDF 기반 PDF 전처리, 문서별/통합 요약
        - **Meeting Agent**: 회의록 요약, 결정사항, 액션아이템, 리스크 정리
        - **Image Agent**: 이미지와 고객 VOC를 함께 분석
        """)

    with col2:
        st.markdown("### 지식 / 운영 / 워크플로 Agent")
        st.markdown("""
        - **RAG Agent**: reports + data/wiki 근거 기반 검색
        - **LangChain Agent**: 필요한 Tool을 선택해 실행하는 운영 Agent
        - **LangGraph Agent**: 조건분기 기반 업무 워크플로 Agent
        """)

    st.divider()

    st.subheader("통합 실행 현황")

    logs = [
        load_log("VOC Agent", "logs/voc_analysis_log.csv"),
        load_log("PDF Agent", "logs/pdf_summary_log.csv"),
        load_log("Meeting Agent", "logs/meeting_agent_log.csv"),
        load_log("Image Agent", "logs/image_analysis_log.csv"),
        load_log("RAG Agent", "logs/rag_agent_log.csv"),
        load_log("LangChain Agent", "logs/langchain_agent_log.csv"),
        load_log("LangGraph Agent", "logs/langgraph_agent_log.csv"),
    ]

    all_logs = pd.concat(logs, ignore_index=True)

    if len(all_logs) == 0:
        st.info("아직 실행 로그가 없습니다.")
    else:
        if "run_time" in all_logs.columns:
            all_logs["run_time"] = pd.to_datetime(all_logs["run_time"], errors="coerce")
            all_logs = all_logs.sort_values("run_time", ascending=False)

        total_runs = len(all_logs)

        if "status" in all_logs.columns:
            success_count = len(all_logs[all_logs["status"] == "success"])
            failed_count = len(all_logs[all_logs["status"] == "failed"])
        else:
            success_count = total_runs
            failed_count = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 실행 횟수", total_runs)
        c2.metric("성공", success_count)
        c3.metric("실패", failed_count)
        c4.metric("Agent 수", all_logs["agent"].nunique())

        agent_counts = all_logs["agent"].value_counts().reset_index()
        agent_counts.columns = ["agent", "count"]

        st.subheader("Agent별 실행 횟수")
        st.dataframe(agent_counts, use_container_width=True)

        st.subheader("최근 전체 실행 로그")
        st.dataframe(all_logs.head(10), use_container_width=True)

    st.divider()

    st.subheader("최근 생성 보고서")

    report_dir = Path("reports")

    if report_dir.exists():
        report_files = sorted(
            [p for p in report_dir.iterdir() if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if report_files:
            report_rows = []

            for path in report_files[:20]:
                report_rows.append({
                    "file_name": path.name,
                    "path": str(path),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "modified_time": pd.to_datetime(path.stat().st_mtime, unit="s")
                })

            st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
        else:
            st.info("reports 폴더에 파일이 없습니다.")
    else:
        st.info("reports 폴더가 없습니다.")

    st.divider()

    st.subheader("현재 프로젝트 상태")

    st.markdown("""
    - 업무 Agent 4종 완료
    - RAG 기반 지식 검색 완료
    - LangChain Tool 운영 Agent 완료
    - LangGraph 조건분기 워크플로 V2 완료
    - 다음 목표: LangGraph 분기별 담당부서/우선순위 고도화 또는 Portal AX Console 고도화
    """)

st.sidebar.title("Agent Portal")

st.sidebar.markdown("### 업무 Agent")
if st.sidebar.button("VOC Agent", use_container_width=True):
    st.session_state["selected_menu"] = "VOC Agent"
if st.sidebar.button("PDF Agent", use_container_width=True):
    st.session_state["selected_menu"] = "PDF Agent"
if st.sidebar.button("Meeting Agent", use_container_width=True):
    st.session_state["selected_menu"] = "Meeting Agent"
if st.sidebar.button("Image Agent", use_container_width=True):
    st.session_state["selected_menu"] = "Image Agent"

st.sidebar.divider()

st.sidebar.markdown("### 지식 Agent")
if st.sidebar.button("RAG Agent", use_container_width=True):
    st.session_state["selected_menu"] = "RAG Agent"

st.sidebar.divider()

st.sidebar.markdown("### 운영 Agent")
if st.sidebar.button("LangChain Agent", use_container_width=True):
    st.session_state["selected_menu"] = "LangChain Agent"

st.sidebar.divider()

st.sidebar.markdown("### 워크플로 Agent")
if st.sidebar.button("LangGraph Agent", use_container_width=True):
    st.session_state["selected_menu"] = "LangGraph Agent"

st.sidebar.divider()

if st.sidebar.button("Portal Home", use_container_width=True):
    st.session_state["selected_menu"] = "Portal Home"

if "selected_menu" not in st.session_state:
    st.session_state["selected_menu"] = "Portal Home"

menu = st.session_state["selected_menu"]

if menu == "Portal Home":
    render_home()
else:
    app_path = APP_MAP.get(menu)

    if app_path is None:
        st.error("선택한 Agent를 찾을 수 없습니다.")
    else:
        runpy.run_path(app_path, run_name="__main__")

