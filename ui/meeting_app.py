import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.meeting_agent import analyze_meeting_text, save_meeting_outputs

st.title("회의록 정리 AI Agent")
st.write("회의 내용을 입력하면 요약, 결정사항, 액션아이템, 리스크를 정리합니다.")

sample_text = """2026년 6월 16일 VOC 개선 회의

참석자: 물류팀 김팀장, 고객서비스팀 이매니저, 품질관리팀 박매니저, 마케팅팀 최매니저

회의 내용:
최근 VOC 분석 결과 배송 지연과 제품 품질 관련 문의가 가장 많이 발생했다.
배송 관련 문의는 배송 출발 알림 누락, 배송 예정일 지연, 다른 옵션 배송 문제가 포함되었다.
품질 관련 문의는 제품 박스 파손, 색상 차이, 배터리 소모 문제 등이 있었다.

김팀장은 물류 파트너와 배송 프로세스를 다시 점검하겠다고 말했다.
이매니저는 교환/환불 절차 안내 문구를 이번 주 안에 정리하겠다고 말했다.
박매니저는 포장재 품질 기준과 출고 전 검수 기준을 재점검하겠다고 말했다.
최매니저는 사용 설명서와 설치 가이드 영상을 보강하는 방안을 검토하겠다고 말했다.

결정사항:
배송 출발 알림 시스템을 우선 점검한다.
교환/환불 안내 문구를 고객센터 FAQ에 추가한다.
포장재 보강안을 다음 회의에서 검토한다.

우려사항:
앱 개발팀 일정이 밀려 있어 알림 시스템 개선 일정이 늦어질 수 있다.
포장재 변경 시 원가 상승 가능성이 있다.
"""

uploaded_file = st.file_uploader("회의록 TXT 파일 업로드", type=["txt"])

if uploaded_file is not None:
    meeting_text = uploaded_file.read().decode("utf-8")
else:
    meeting_text = st.text_area("회의록 입력", value=sample_text, height=350)

def save_meeting_log(status, message, report_path="", actions_count=0):
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/meeting_agent_log.csv"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_row = pd.DataFrame([{
        "run_time": run_time,
        "status": status,
        "message": message,
        "actions_count": actions_count,
        "report_path": report_path
    }])

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        updated_log = pd.concat([existing_log, log_row], ignore_index=True)
    else:
        updated_log = log_row

    updated_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    return log_path

if st.button("회의록 정리하기"):
    try:
        with st.spinner("GPT가 회의록을 정리 중입니다..."):
            (
                report,
                summary_df,
                decisions_df,
                action_df,
                risks_df,
                follow_up_df
            ) = analyze_meeting_text(meeting_text)

            (
                report_path,
                decisions_path,
                actions_path,
                risks_path,
                follow_up_path
            ) = save_meeting_outputs(
                report,
                decisions_df,
                action_df,
                risks_df,
                follow_up_df
            )

            log_path = save_meeting_log(
                "success",
                "회의록 정리 성공",
                report_path,
                len(action_df)
            )

        st.subheader("회의 요약")
        st.dataframe(summary_df)

        st.subheader("결정사항")
        st.dataframe(decisions_df)

        st.subheader("액션아이템")
        st.dataframe(action_df)

        st.subheader("리스크")
        st.dataframe(risks_df)

        st.subheader("확인 필요 사항")
        st.dataframe(follow_up_df)

        st.subheader("회의록 정리 보고서")
        st.markdown(report)

        st.success(f"보고서 저장 완료: {report_path}")
        st.success(f"결정사항 저장 완료: {decisions_path}")
        st.success(f"액션아이템 저장 완료: {actions_path}")
        st.success(f"리스크 저장 완료: {risks_path}")
        st.success(f"확인사항 저장 완료: {follow_up_path}")
        st.success(f"실행 로그 저장 완료: {log_path}")

        now = datetime.now().strftime("%Y-%m-%d_%H-%M")

        st.subheader("다운로드")

        st.download_button(
            label="회의록 보고서 TXT 다운로드",
            data=report.encode("utf-8-sig"),
            file_name=f"meeting_report_{now}.txt",
            mime="text/plain"
        )

        st.download_button(
            label="액션아이템 CSV 다운로드",
            data=action_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"meeting_actions_{now}.csv",
            mime="text/csv"
        )

        st.download_button(
            label="결정사항 CSV 다운로드",
            data=decisions_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"meeting_decisions_{now}.csv",
            mime="text/csv"
        )

        if os.path.exists("logs/meeting_agent_log.csv"):
            st.subheader("최근 회의록 정리 로그")
            log_df = pd.read_csv("logs/meeting_agent_log.csv")
            st.dataframe(log_df.tail(10))

    except Exception as e:
        log_path = save_meeting_log("failed", str(e))
        st.error("회의록 정리 중 오류가 발생했습니다.")
        st.write(str(e))
        st.warning(f"에러 로그 저장 완료: {log_path}")
else:
    st.write("회의록을 입력하거나 TXT 파일을 업로드한 뒤 버튼을 눌러주세요.")
