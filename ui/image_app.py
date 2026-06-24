import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.image_agent import analyze_image, save_image_report

st.title("이미지 분석 AI Agent")
st.write("이미지와 고객 문의/VOC를 함께 분석해 실무 보고서를 생성합니다.")

uploaded_file = st.file_uploader(
    "이미지 파일 업로드",
    type=["png", "jpg", "jpeg"]
)

analysis_type = st.selectbox(
    "분석 목적 선택",
    [
        "일반 이미지 분석",
        "제품 이미지 분석",
        "상세페이지/광고 소재 분석",
        "VOC/불량 사진 분석",
        "문서/책 페이지 이미지 분석",
        "현장/매장/물류 이미지 분석"
    ]
)

context_text = st.text_area(
    "고객 문의/VOC 또는 상황 설명 입력",
    height=150,
    placeholder="예: 고객이 손잡이에 스크래치가 있다고 주장함 / 배송 중 박스가 찢어져 도착했다고 함 / 상세페이지용 제품 사진으로 쓸 수 있는지 검토"
)

def guess_mime_type(file_name):
    lower_name = file_name.lower()

    if lower_name.endswith(".png"):
        return "image/png"

    if lower_name.endswith(".jpg") or lower_name.endswith(".jpeg"):
        return "image/jpeg"

    return "image/jpeg"

def save_image_log(uploaded_file_name, status, message, analysis_type, context_text, report_path=""):
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/image_analysis_log.csv"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_row = pd.DataFrame([{
        "run_time": run_time,
        "uploaded_file": uploaded_file_name,
        "status": status,
        "message": message,
        "analysis_type": analysis_type,
        "context_length": len(context_text),
        "report_path": report_path
    }])

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        updated_log = pd.concat([existing_log, log_row], ignore_index=True)
    else:
        updated_log = log_row

    updated_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    return log_path

if uploaded_file is not None:
    st.subheader("업로드된 이미지")

    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, use_container_width=True)

    image_bytes = uploaded_file.getvalue()
    mime_type = guess_mime_type(uploaded_file.name)

    st.caption(f"파일 형식: {mime_type}")

    if st.button("이미지 분석하기"):
        try:
            with st.spinner("GPT가 이미지와 상황 설명을 함께 분석 중입니다..."):
                report = analyze_image(
                    image_bytes=image_bytes,
                    file_name=uploaded_file.name,
                    analysis_type=analysis_type,
                    context_text=context_text,
                    mime_type=mime_type
                )

                report_path = save_image_report(report)
                log_path = save_image_log(
                    uploaded_file.name,
                    "success",
                    "이미지 분석 성공",
                    analysis_type,
                    context_text,
                    report_path
                )

            st.subheader("이미지 분석 보고서")
            st.markdown(report)

            st.success(f"보고서 저장 완료: {report_path}")
            st.success(f"실행 로그 저장 완료: {log_path}")

            now = datetime.now().strftime("%Y-%m-%d_%H-%M")

            st.subheader("다운로드")

            st.download_button(
                label="이미지 분석 보고서 TXT 다운로드",
                data=report.encode("utf-8-sig"),
                file_name=f"image_analysis_report_{now}.txt",
                mime="text/plain"
            )

            if os.path.exists("logs/image_analysis_log.csv"):
                st.subheader("최근 이미지 분석 로그")
                log_df = pd.read_csv("logs/image_analysis_log.csv")
                st.dataframe(log_df.tail(10))

        except Exception as e:
            log_path = save_image_log(
                uploaded_file.name,
                "failed",
                str(e),
                analysis_type,
                context_text
            )

            st.error("이미지 분석 중 오류가 발생했습니다.")
            st.write(str(e))
            st.warning(f"에러 로그 저장 완료: {log_path}")

else:
    st.write("먼저 이미지를 업로드해주세요.")
