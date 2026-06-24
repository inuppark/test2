import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rag_agent import (
    build_rag_index,
    load_rag_index,
    answer_with_rag,
    save_rag_log
)

st.title("RAG 지식 검색 Agent")
st.write("reports와 data/wiki 문서를 검색해서 근거 기반 답변을 생성합니다.")

st.info("V5: 검색 근거 원문 보기와 답변 신뢰도 표시를 추가했습니다.")

WIKI_DIR = Path("data/wiki")
WIKI_DIR.mkdir(parents=True, exist_ok=True)

SCOPE_LABEL_TO_VALUE = {
    "전체 검색: 보고서 + 사내위키": "전체",
    "업무 보고서만 검색": "보고서 전체",
    "사내 기준 문서만 검색": "사내위키",
    "VOC 보고서만 검색": "VOC 보고서",
    "PDF 보고서만 검색": "PDF 보고서",
    "회의록 보고서만 검색": "회의록 보고서",
    "이미지 보고서만 검색": "이미지 보고서"
}

SCOPE_GUIDE = {
    "전체 검색: 보고서 + 사내위키": "최근 업무 결과와 회사 기준 문서를 함께 검색합니다. 현재 이슈와 처리 기준을 같이 볼 때 사용합니다.",
    "업무 보고서만 검색": "Agent들이 생성한 reports 폴더의 업무 결과 보고서만 검색합니다.",
    "사내 기준 문서만 검색": "data/wiki 폴더의 정책, 기준, 매뉴얼 문서만 검색합니다.",
    "VOC 보고서만 검색": "VOC 분석 보고서만 검색합니다.",
    "PDF 보고서만 검색": "PDF 요약 보고서만 검색합니다.",
    "회의록 보고서만 검색": "회의록 정리 보고서만 검색합니다.",
    "이미지 보고서만 검색": "이미지 분석 보고서만 검색합니다."
}

st.subheader("사내위키 문서 업로드")

uploaded_wiki_files = st.file_uploader(
    "TXT 또는 MD 문서를 업로드하세요",
    type=["txt", "md"],
    accept_multiple_files=True
)

def save_uploaded_wiki_files(files):
    saved_files = []

    for file in files:
        safe_name = file.name.replace(" ", "_")
        save_path = WIKI_DIR / safe_name

        content = file.read()

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8-sig")

        save_path.write_text(text, encoding="utf-8")

        saved_files.append({
            "file_name": safe_name,
            "path": str(save_path),
            "size_kb": round(len(content) / 1024, 1)
        })

    return saved_files

if uploaded_wiki_files:
    if st.button("사내위키 문서 저장"):
        try:
            saved_files = save_uploaded_wiki_files(uploaded_wiki_files)

            st.success("사내위키 문서 저장 완료")
            st.dataframe(pd.DataFrame(saved_files), use_container_width=True)

            st.warning("새 문서를 검색하려면 아래의 'RAG 인덱스 만들기 / 업데이트' 버튼을 다시 눌러야 합니다.")

        except Exception as e:
            st.error("사내위키 문서 저장 중 오류가 발생했습니다.")
            st.write(str(e))

st.divider()

st.subheader("현재 사내위키 문서 목록")

wiki_files = sorted([
    p for p in WIKI_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in [".txt", ".md"]
])

if wiki_files:
    wiki_rows = []

    for path in wiki_files:
        wiki_rows.append({
            "file_name": path.name,
            "path": str(path),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "modified_time": pd.to_datetime(path.stat().st_mtime, unit="s")
        })

    st.dataframe(pd.DataFrame(wiki_rows), use_container_width=True)
else:
    st.info("아직 data/wiki 폴더에 문서가 없습니다.")

st.divider()

st.subheader("RAG 인덱스 관리")

col1, col2 = st.columns(2)

with col1:
    if st.button("RAG 인덱스 만들기 / 업데이트"):
        try:
            with st.spinner("보고서와 사내위키 문서를 읽고 임베딩 인덱스를 생성 중입니다..."):
                result = build_rag_index()

            st.success("RAG 인덱스 생성 완료")
            st.write(result)

        except Exception as e:
            st.error("RAG 인덱스 생성 중 오류가 발생했습니다.")
            st.write(str(e))

with col2:
    try:
        index_data = load_rag_index()
        st.success("현재 RAG 인덱스 있음")
        st.json({
            "created_at": index_data.get("created_at"),
            "source_count": index_data.get("source_count"),
            "chunk_count": index_data.get("chunk_count"),
            "embedding_model": index_data.get("embedding_model")
        })
    except Exception:
        st.warning("아직 RAG 인덱스가 없습니다.")

st.divider()

st.subheader("RAG 질문하기")

scope_label = st.selectbox(
    "검색 범위 선택",
    list(SCOPE_LABEL_TO_VALUE.keys())
)

search_scope = SCOPE_LABEL_TO_VALUE[scope_label]

st.caption(SCOPE_GUIDE[scope_label])

sample_questions = {
    "전체": [
        "최근 보고서와 사내 기준을 함께 봤을 때 가장 우선 처리할 업무는 뭐야?",
        "현재 고객 불만 처리 기준과 최근 이슈를 연결해서 설명해줘"
    ],
    "보고서 전체": [
        "최근 보고서들을 기준으로 가장 우선 처리할 업무는 뭐야?",
        "최근 보고서들에서 반복되는 핵심 이슈는 뭐야?"
    ],
    "사내위키": [
        "제품 하자 문의가 들어오면 어떤 절차로 처리해야 해?",
        "배송 지연 문의는 어느 부서가 담당해야 해?"
    ],
    "VOC 보고서": [
        "최근 VOC에서 가장 중요한 문제는 뭐야?",
        "VOC 기준으로 가장 시급한 개선 영역은 뭐야?"
    ],
    "PDF 보고서": [
        "PDF 보고서에서 실무 적용 포인트를 요약해줘",
        "PDF 문서 기준으로 확인 필요 사항은 뭐야?"
    ],
    "회의록 보고서": [
        "회의록에서 액션아이템이 뭐였어?",
        "회의록 기준으로 결정사항과 리스크를 정리해줘"
    ],
    "이미지 보고서": [
        "최근 이미지 분석 결과에서 제품 하자 의심 내용이 있었어?",
        "이미지 분석 기준으로 고객에게 추가 확인할 사항은 뭐야?"
    ]
}

default_question = sample_questions[search_scope][0]

question = st.text_area(
    "질문 입력",
    value=default_question,
    height=120
)

st.write("예시 질문:")
for q in sample_questions[search_scope]:
    st.code(q)

top_k = st.slider("검색할 근거 조각 수", min_value=1, max_value=10, value=5)

if st.button("RAG Agent 실행"):
    try:
        with st.spinner("선택한 범위에서 관련 문서를 검색하고 답변을 생성 중입니다..."):
            answer, search_results, confidence = answer_with_rag(
                question=question,
                top_k=top_k,
                search_scope=search_scope
            )

            log_path = save_rag_log(
                question=question,
                answer=answer,
                search_results=search_results,
                search_scope=search_scope,
                confidence=confidence
            )

        st.subheader("답변 신뢰도")

        c1, c2 = st.columns(2)
        c1.metric("신뢰도", confidence["level"])
        c2.metric("최고 유사도", confidence["top_score"])

        if confidence["level"] == "낮음":
            st.warning(confidence["reason"])
        elif confidence["level"] == "보통":
            st.info(confidence["reason"])
        else:
            st.success(confidence["reason"])

        st.subheader("답변")
        st.markdown(answer)

        st.subheader("검색된 근거 요약")
        result_df = pd.DataFrame([
            {
                "rank": idx + 1,
                "score": round(item["score"], 4),
                "source_group": item["source_group"],
                "report_type": item["report_type"],
                "source_file": item["source_file"],
                "chunk_id": item["chunk_id"],
                "text_preview": item["text"][:300]
            }
            for idx, item in enumerate(search_results)
        ])

        st.dataframe(result_df, use_container_width=True)

        st.subheader("검색 근거 원문 보기")

        for idx, item in enumerate(search_results, start=1):
            title = f"근거 {idx} | {item['source_file']} | score={round(item['score'], 4)}"
            with st.expander(title, expanded=False):
                st.write(f"출처 그룹: {item['source_group']}")
                st.write(f"문서 유형: {item['report_type']}")
                st.write(f"파일 경로: {item['source_path']}")
                st.write(f"청크 번호: {item['chunk_id']}")
                st.text_area(
                    f"근거 {idx} 원문",
                    value=item["text"],
                    height=250
                )

        st.success(f"실행 로그 저장 완료: {log_path}")

        if os.path.exists("logs/rag_agent_log.csv"):
            st.subheader("최근 RAG Agent 로그")
            log_df = pd.read_csv("logs/rag_agent_log.csv")
            st.dataframe(log_df.tail(10), use_container_width=True)

    except Exception as e:
        st.error("RAG Agent 실행 중 오류가 발생했습니다.")
        st.write(str(e))
else:
    st.write("사내위키 문서를 추가하거나 인덱스를 만든 뒤 질문을 입력하세요.")
