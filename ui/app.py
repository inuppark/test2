import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.voc_agent import analyze_voc_text, save_report

st.title("VOC 분석 AI Agent")
st.write("엑셀 파일을 업로드하면 고객 문의를 분석하고 VOC 보고서를 생성합니다.")

st.info("VOC 컬럼이 없어도 '고객문의', '문의내용', '문의', '상담내용' 컬럼이 있으면 자동으로 인식합니다.")

uploaded_file = st.file_uploader("VOC 엑셀 파일 업로드", type=["xlsx"])

POSSIBLE_VOC_COLUMNS = [
    "VOC",
    "고객문의",
    "문의내용",
    "문의",
    "고객 문의",
    "상담내용",
    "고객의견",
    "리뷰",
    "불만내용"
]

def find_voc_column(df):
    for col in POSSIBLE_VOC_COLUMNS:
        if col in df.columns:
            return col
    return None

def save_error_log(uploaded_file_name, error_message):
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/voc_error_log.csv"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    error_row = pd.DataFrame([{
        "run_time": run_time,
        "uploaded_file": uploaded_file_name,
        "status": "failed",
        "error_message": str(error_message)
    }])

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        updated_log = pd.concat([existing_log, error_row], ignore_index=True)
    else:
        updated_log = error_row

    updated_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    return log_path

def save_outputs_and_log(
    uploaded_file_name,
    report,
    classified_df,
    category_summary,
    severity_summary,
    department_summary,
    report_path
):
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    category_path = f"reports/voc_category_summary_{now}.csv"
    severity_path = f"reports/voc_severity_summary_{now}.csv"
    department_path = f"reports/voc_department_summary_{now}.csv"
    classified_path = f"reports/voc_classified_{now}.csv"
    log_path = "logs/voc_analysis_log.csv"

    category_summary.to_csv(category_path, index=False, encoding="utf-8-sig")
    severity_summary.to_csv(severity_path, index=False, encoding="utf-8-sig")
    department_summary.to_csv(department_path, index=False, encoding="utf-8-sig")
    classified_df.to_csv(classified_path, index=False, encoding="utf-8-sig")

    total_count = int(category_summary["count"].sum())

    if len(category_summary) > 0:
        top_row = category_summary.sort_values("count", ascending=False).iloc[0]
        top_category = str(top_row["category"])
        top_count = int(top_row["count"])
    else:
        top_category = "없음"
        top_count = 0

    high_risk_count = int(
        classified_df[classified_df["severity"].isin(["높음", "긴급"])].shape[0]
    )

    log_row = pd.DataFrame([{
        "run_time": run_time,
        "uploaded_file": uploaded_file_name,
        "status": "success",
        "total_voc_count": total_count,
        "top_category": top_category,
        "top_category_count": top_count,
        "high_risk_count": high_risk_count,
        "report_path": report_path,
        "category_path": category_path,
        "severity_path": severity_path,
        "department_path": department_path,
        "classified_path": classified_path
    }])

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        updated_log = pd.concat([existing_log, log_row], ignore_index=True)
    else:
        updated_log = log_row

    updated_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    return category_path, severity_path, department_path, classified_path, log_path

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("업로드된 데이터 미리보기")
        st.dataframe(df)

        voc_column = find_voc_column(df)

        if voc_column is None:
            st.error("VOC로 사용할 수 있는 컬럼을 찾지 못했습니다.")
            st.write("가능한 컬럼명:", POSSIBLE_VOC_COLUMNS)
            save_error_log(uploaded_file.name, "VOC 컬럼 없음")
        else:
            st.success(f"VOC 컬럼으로 '{voc_column}' 컬럼을 사용합니다.")

            voc_list = df[voc_column].dropna().astype(str).tolist()
            voc_list = [item.strip() for item in voc_list if item.strip()]

            if len(voc_list) == 0:
                st.error("분석할 VOC 데이터가 없습니다. 빈 행만 있는지 확인해주세요.")
                save_error_log(uploaded_file.name, "VOC 데이터 없음")
            else:
                original_count = len(voc_list)

                if len(voc_list) > 100:
                    st.warning(f"VOC가 {len(voc_list)}건입니다. 테스트 비용과 속도를 위해 앞 100건만 분석합니다.")
                    voc_list = voc_list[:100]

                voc_text = "\n".join(voc_list)

                st.subheader(f"분석 대상 VOC 총 {len(voc_list)}건")
                if original_count != len(voc_list):
                    st.caption(f"원본 {original_count}건 중 앞 {len(voc_list)}건만 분석합니다.")

                preview_df = pd.DataFrame({
                    "번호": range(1, len(voc_list) + 1),
                    "VOC": voc_list
                })
                st.dataframe(preview_df)

                if st.button("VOC 분석하기"):
                    try:
                        with st.spinner("GPT가 VOC를 분류하고, 심각도와 담당부서 및 후속 액션을 추천 중입니다..."):
                            (
                                report,
                                classified_df,
                                category_summary,
                                severity_summary,
                                department_summary
                            ) = analyze_voc_text(voc_text)

                            report_path = save_report(report)

                            (
                                category_path,
                                severity_path,
                                department_path,
                                classified_path,
                                log_path
                            ) = save_outputs_and_log(
                                uploaded_file.name,
                                report,
                                classified_df,
                                category_summary,
                                severity_summary,
                                department_summary,
                                report_path
                            )

                        st.subheader("카테고리별 정확한 집계")
                        st.dataframe(category_summary)

                        st.subheader("심각도별 집계")
                        st.dataframe(severity_summary)

                        st.subheader("담당 부서별 집계")
                        st.dataframe(department_summary)

                        st.subheader("VOC별 분석 결과")
                        st.dataframe(classified_df)

                        high_risk_df = classified_df[classified_df["severity"].isin(["높음", "긴급"])]

                        if len(high_risk_df) > 0:
                            st.subheader("고위험 VOC")
                            st.warning(f"높음/긴급 VOC가 {len(high_risk_df)}건 있습니다.")
                            st.dataframe(high_risk_df)
                        else:
                            st.success("높음/긴급 VOC가 없습니다.")

                        st.subheader("VOC 분석 보고서")
                        st.markdown(report)

                        st.success(f"보고서 저장 완료: {report_path}")
                        st.success(f"카테고리 집계 저장 완료: {category_path}")
                        st.success(f"심각도 집계 저장 완료: {severity_path}")
                        st.success(f"담당부서 집계 저장 완료: {department_path}")
                        st.success(f"VOC별 분석 결과 저장 완료: {classified_path}")
                        st.success(f"실행 로그 저장 완료: {log_path}")

                        now = datetime.now().strftime("%Y-%m-%d_%H-%M")

                        report_txt = report.encode("utf-8-sig")
                        category_csv = category_summary.to_csv(index=False).encode("utf-8-sig")
                        severity_csv = severity_summary.to_csv(index=False).encode("utf-8-sig")
                        department_csv = department_summary.to_csv(index=False).encode("utf-8-sig")
                        classified_csv = classified_df.to_csv(index=False).encode("utf-8-sig")

                        st.subheader("다운로드")

                        st.download_button(
                            label="보고서 TXT 다운로드",
                            data=report_txt,
                            file_name=f"voc_report_{now}.txt",
                            mime="text/plain"
                        )

                        st.download_button(
                            label="카테고리 집계 CSV 다운로드",
                            data=category_csv,
                            file_name=f"voc_category_summary_{now}.csv",
                            mime="text/csv"
                        )

                        st.download_button(
                            label="심각도 집계 CSV 다운로드",
                            data=severity_csv,
                            file_name=f"voc_severity_summary_{now}.csv",
                            mime="text/csv"
                        )

                        st.download_button(
                            label="담당부서 집계 CSV 다운로드",
                            data=department_csv,
                            file_name=f"voc_department_summary_{now}.csv",
                            mime="text/csv"
                        )

                        st.download_button(
                            label="VOC별 분석 결과 CSV 다운로드",
                            data=classified_csv,
                            file_name=f"voc_classified_{now}.csv",
                            mime="text/csv"
                        )

                        if os.path.exists("logs/voc_analysis_log.csv"):
                            st.subheader("최근 실행 로그")
                            log_df = pd.read_csv("logs/voc_analysis_log.csv")
                            st.dataframe(log_df.tail(10))

                    except Exception as e:
                        error_log_path = save_error_log(uploaded_file.name, e)
                        st.error("분석 중 오류가 발생했습니다.")
                        st.write(str(e))
                        st.warning(f"에러 로그 저장 완료: {error_log_path}")

    except Exception as e:
        error_log_path = save_error_log(uploaded_file.name, e)
        st.error("엑셀 파일을 읽는 중 오류가 발생했습니다.")
        st.write(str(e))
        st.warning(f"에러 로그 저장 완료: {error_log_path}")

else:
    st.write("먼저 VOC 엑셀 파일을 업로드해주세요.")
