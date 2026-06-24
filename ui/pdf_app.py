import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.pdf_agent import (
    extract_text_from_pdf,
    summarize_pdf_text,
    create_integrated_report,
    save_text_report
)

st.title("PDF 문서 요약 AI Agent")
st.write("PDF 파일을 업로드하면 PyMuPDF로 텍스트를 전처리하고 문서별 요약 및 통합 요약 보고서를 생성합니다.")

st.info("PDF 처리 엔진: PyMuPDF. 텍스트 PDF를 우선 지원하며, 스캔 이미지 PDF는 추출 실패 페이지로 표시됩니다.")

uploaded_files = st.file_uploader(
    "PDF 파일 업로드",
    type=["pdf"],
    accept_multiple_files=True
)

def save_pdf_log(log_rows):
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/pdf_summary_log.csv"
    log_df = pd.DataFrame(log_rows)

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        updated_log = pd.concat([existing_log, log_df], ignore_index=True)
    else:
        updated_log = log_df

    updated_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    return log_path

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("테스트 비용과 속도를 위해 PDF는 앞 5개만 분석합니다.")
        uploaded_files = uploaded_files[:5]

    st.subheader("업로드된 PDF 목록")

    upload_preview = pd.DataFrame([
        {
            "번호": index + 1,
            "파일명": file.name,
            "크기(KB)": round(file.size / 1024, 1)
        }
        for index, file in enumerate(uploaded_files)
    ])

    st.dataframe(upload_preview)

    if st.button("PDF 요약하기"):
        document_results = []
        extraction_rows = []
        log_rows = []
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for file in uploaded_files:
            try:
                with st.spinner(f"{file.name} 텍스트 추출 중입니다..."):
                    extracted = extract_text_from_pdf(file)

                extraction_rows.append({
                    "파일명": file.name,
                    "처리엔진": "PyMuPDF",
                    "전체 페이지": extracted["total_pages"],
                    "추출 성공 페이지": extracted["success_pages"],
                    "추출 실패 페이지": extracted["failed_pages"],
                    "추출 문자 수": extracted["text_length"]
                })

                st.subheader(f"추출 텍스트 미리보기: {file.name}")
                st.text_area(
                    f"{file.name} 텍스트",
                    value=extracted["text"][:3000],
                    height=250
                )

                page_status_df = pd.DataFrame(extracted["page_status"])
                st.subheader(f"페이지별 추출 현황: {file.name}")
                st.dataframe(page_status_df)

                with st.spinner(f"{file.name} 요약 중입니다..."):
                    summary_result = summarize_pdf_text(
                        file.name,
                        extracted["text"],
                        extracted["total_pages"],
                        extracted["success_pages"],
                        extracted["failed_pages"]
                    )

                report = summary_result["report"]
                report_path = save_text_report(report, "pdf_summary_report")

                document_results.append({
                    "file_name": file.name,
                    "report": report,
                    "report_path": report_path,
                    "total_pages": extracted["total_pages"],
                    "success_pages": extracted["success_pages"],
                    "failed_pages": extracted["failed_pages"],
                    "original_length": summary_result["original_length"],
                    "used_length": summary_result["used_length"],
                    "is_truncated": summary_result["is_truncated"]
                })

                log_rows.append({
                    "run_time": run_time,
                    "uploaded_file": file.name,
                    "status": "success",
                    "message": "PDF 요약 성공",
                    "engine": "PyMuPDF",
                    "total_pages": extracted["total_pages"],
                    "success_pages": extracted["success_pages"],
                    "failed_pages": extracted["failed_pages"],
                    "original_text_length": summary_result["original_length"],
                    "used_text_length": summary_result["used_length"],
                    "is_truncated": summary_result["is_truncated"],
                    "report_path": report_path
                })

            except Exception as e:
                extraction_rows.append({
                    "파일명": file.name,
                    "처리엔진": "PyMuPDF",
                    "전체 페이지": 0,
                    "추출 성공 페이지": 0,
                    "추출 실패 페이지": 0,
                    "추출 문자 수": 0
                })

                log_rows.append({
                    "run_time": run_time,
                    "uploaded_file": file.name,
                    "status": "failed",
                    "message": str(e),
                    "engine": "PyMuPDF",
                    "total_pages": 0,
                    "success_pages": 0,
                    "failed_pages": 0,
                    "original_text_length": 0,
                    "used_text_length": 0,
                    "is_truncated": False,
                    "report_path": ""
                })

                st.error(f"{file.name} 처리 중 오류가 발생했습니다.")
                st.write(str(e))

        log_path = save_pdf_log(log_rows)

        if extraction_rows:
            st.subheader("문서별 추출 요약")
            extraction_df = pd.DataFrame(extraction_rows)
            st.dataframe(extraction_df)

        if document_results:
            st.subheader("문서별 요약 보고서")

            for index, item in enumerate(document_results, start=1):
                st.markdown(f"## 문서 {index}: {item['file_name']}")
                st.markdown(item["report"])
                st.success(f"문서별 보고서 저장 완료: {item['report_path']}")

                st.download_button(
                    label=f"{item['file_name']} 요약 보고서 TXT 다운로드",
                    data=item["report"].encode("utf-8-sig"),
                    file_name=f"pdf_summary_{index}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt",
                    mime="text/plain"
                )

            if len(document_results) >= 2:
                with st.spinner("여러 PDF의 통합 요약 보고서를 생성 중입니다..."):
                    integrated_report = create_integrated_report(document_results)
                    integrated_path = save_text_report(integrated_report, "pdf_integrated_summary_report")

                st.subheader("통합 PDF 문서 요약 보고서")
                st.markdown(integrated_report)
                st.success(f"통합 보고서 저장 완료: {integrated_path}")

                st.download_button(
                    label="통합 요약 보고서 TXT 다운로드",
                    data=integrated_report.encode("utf-8-sig"),
                    file_name=f"pdf_integrated_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt",
                    mime="text/plain"
                )

        st.success(f"실행 로그 저장 완료: {log_path}")

        if os.path.exists("logs/pdf_summary_log.csv"):
            st.subheader("최근 PDF 요약 로그")
            log_df = pd.read_csv("logs/pdf_summary_log.csv")
            st.dataframe(log_df.tail(10))

else:
    st.write("먼저 PDF 파일을 업로드해주세요.")
