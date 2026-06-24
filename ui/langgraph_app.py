import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.langgraph_agent import (
    run_langgraph_workflow,
    save_langgraph_log,
    load_approval_queue,
    update_approval_status,
    regenerate_rejected_case,
    load_regeneration_log
)

st.title("LangGraph Workflow Agent")
st.write("고객 이슈를 입력하면 분류 → 조건분기 → RAG 검색 → 심각도 판단 → 승인 판단 → 조치안 생성 → 보고서 작성 흐름으로 처리합니다.")

st.info("V6: 반려 건은 반려 의견을 반영해 수정 고객 안내문과 수정 조치안을 재작성할 수 있습니다.")

tab1, tab2, tab3, tab4 = st.tabs([
    "워크플로 실행",
    "승인 대기함",
    "후속 처리 로그",
    "반려 재작성"
])

with tab1:
    sample_issues = {
        "배송/물류 예시": "고객이 배송받은 제품이 주문한 옵션과 다르고, 포장도 찢어진 상태였다고 문의했습니다. 교환 절차와 담당 부서를 확인해 주세요.",
        "제품품질 예시": "고객이 제품 사용 첫날부터 배터리가 너무 빨리 닳고 손잡이에 스크래치가 있다고 문의했습니다.",
        "교환/환불 예시": "고객이 제품이 마음에 들지 않아 환불을 요청했고, 교환 절차가 너무 복잡하다고 불만을 제기했습니다.",
        "사용안내 예시": "고객이 설명서가 너무 어려워서 제품 초기화 방법을 모르겠다고 문의했습니다.",
        "고객서비스 예시": "고객이 상담 연결까지 너무 오래 기다렸고 AS 진행 상황 안내를 받지 못했다고 문의했습니다.",
        "긴급/안전 예시": "고객이 제품 사용 중 배터리 과열로 화상 위험을 느꼈고, 같은 문제가 여러 고객에게 반복될 수 있다고 주장했습니다."
    }

    selected_sample = st.selectbox(
        "테스트 예시 선택",
        list(sample_issues.keys())
    )

    user_issue = st.text_area(
        "고객 이슈 입력",
        value=sample_issues[selected_sample],
        height=180
    )

    if st.button("LangGraph Workflow 실행"):
        try:
            with st.spinner("LangGraph가 승인 판단을 포함해 워크플로를 실행 중입니다..."):
                result = run_langgraph_workflow(user_issue)
                report_path, log_path, approval_queue_path = save_langgraph_log(result)

            st.subheader("워크플로 실행 결과")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("이슈 분류", result.get("issue_type"))
            col2.metric("적용 흐름", result.get("route_name"))
            col3.metric("심각도", result.get("severity"))
            col4.metric("승인 필요", result.get("approval_required"))

            col5, col6, col7 = st.columns(3)
            col5.metric("승인 상태", result.get("approval_status"))
            col6.metric("RAG 신뢰도", result.get("confidence_level"))
            col7.metric("최고 유사도", result.get("top_score"))

            st.subheader("승인 정보")
            st.markdown(f"""
            - 승인 ID: **{result.get("approval_id")}**
            - 승인 필요 여부: **{result.get("approval_required")}**
            - 승인 상태: **{result.get("approval_status")}**
            - 승인 판단 이유: {result.get("approval_reason")}
            """)

            if approval_queue_path:
                st.warning(f"승인 필요 건으로 등록되었습니다: {approval_queue_path}")
            else:
                st.success("승인 없이 표준 처리 가능한 건입니다.")

            st.subheader("판단 이유")
            st.markdown(f"""
            **심각도 판단 이유**  
            {result.get("severity_reason")}

            **승인 필요 여부 판단 이유**  
            {result.get("approval_reason")}
            """)

            st.subheader("조건분기 실행 단계")
            st.markdown(f"""
            1. 이슈 분류 완료: **{result.get("issue_type")}**  
            2. 조건분기 완료: **{result.get("route_name")}**  
            3. RAG 기준 검색 완료  
            4. 심각도 판단 완료: **{result.get("severity")}**  
            5. 승인 필요 여부 판단 완료: **{result.get("approval_required")}**  
            6. 승인 대기함 등록 여부 확인  
            7. 실무 조치안 생성 완료  
            8. 최종 보고서 생성 완료  
            """)

            st.subheader("근거 문서")
            sources = result.get("sources", [])

            if sources:
                st.dataframe(pd.DataFrame(sources), use_container_width=True)
            else:
                st.info("검색된 근거 문서가 없습니다.")

            st.subheader("최종 보고서")
            st.markdown(result.get("final_report"))

            st.success(f"보고서 저장 완료: {report_path}")
            st.success(f"실행 로그 저장 완료: {log_path}")

            st.download_button(
                label="LangGraph 처리 보고서 TXT 다운로드",
                data=result.get("final_report", "").encode("utf-8-sig"),
                file_name="langgraph_issue_report.txt",
                mime="text/plain"
            )

            if os.path.exists("logs/langgraph_agent_log.csv"):
                st.subheader("최근 LangGraph 로그")
                log_df = pd.read_csv("logs/langgraph_agent_log.csv")
                st.dataframe(log_df.tail(10), use_container_width=True)

        except Exception as e:
            st.error("LangGraph Workflow 실행 중 오류가 발생했습니다.")
            st.write(str(e))
    else:
        st.write("고객 이슈를 입력한 뒤 실행 버튼을 눌러주세요.")

with tab2:
    st.subheader("승인 대기함")

    approval_df = load_approval_queue()

    if approval_df.empty:
        st.info("아직 승인 대기 건이 없습니다.")
    else:
        st.dataframe(approval_df.sort_values("created_at", ascending=False), use_container_width=True)

        pending_df = approval_df[approval_df["approval_status"] == "대기"]

        if pending_df.empty:
            st.success("현재 승인 대기 상태인 건이 없습니다.")
        else:
            approval_id = st.selectbox(
                "처리할 승인 ID 선택",
                pending_df["approval_id"].tolist()
            )

            selected_row = pending_df[pending_df["approval_id"] == approval_id].iloc[0]

            st.markdown("### 선택한 승인 건")
            st.json({
                "approval_id": selected_row.get("approval_id", ""),
                "issue_type": selected_row.get("issue_type", ""),
                "route_name": selected_row.get("route_name", ""),
                "severity": selected_row.get("severity", ""),
                "user_issue": selected_row.get("user_issue", ""),
                "report_path": selected_row.get("report_path", "")
            })

            reviewer = st.text_input("검토자 이름", value="CS 운영 담당자")

            new_status = st.radio(
                "승인 처리",
                ["승인", "반려"],
                horizontal=True
            )

            default_comment = "내용 확인 후 처리합니다." if new_status == "승인" else "고객 안내 문구와 근거 문서를 보강한 뒤 재검토가 필요합니다."

            comment = st.text_area(
                "검토 의견",
                value=default_comment,
                height=120
            )

            if st.button("승인 상태 업데이트"):
                try:
                    update_result = update_approval_status(
                        approval_id=approval_id,
                        new_status=new_status,
                        reviewer=reviewer,
                        comment=comment
                    )

                    st.success(f"승인 상태 업데이트 완료: {update_result['approval_queue_path']}")

                    followup = update_result.get("followup_result", {})

                    st.subheader("후속 처리 결과")
                    st.write({
                        "followup_status": followup.get("followup_status"),
                        "followup_report_path": followup.get("followup_report_path"),
                        "followup_log_path": followup.get("followup_log_path")
                    })

                    st.markdown(followup.get("followup_report", ""))

                    st.rerun()

                except Exception as e:
                    st.error("승인 상태 업데이트 중 오류가 발생했습니다.")
                    st.write(str(e))

with tab3:
    st.subheader("후속 처리 로그")

    followup_log_path = "logs/langgraph_followup_log.csv"

    if not os.path.exists(followup_log_path):
        st.info("아직 후속 처리 로그가 없습니다.")
    else:
        followup_df = pd.read_csv(followup_log_path, dtype=str, keep_default_na=False)
        st.dataframe(followup_df.sort_values("run_time", ascending=False), use_container_width=True)

        report_paths = followup_df["followup_report_path"].dropna().tolist()

        if report_paths:
            selected_report = st.selectbox(
                "후속 처리 보고서 선택",
                report_paths
            )

            if selected_report and os.path.exists(selected_report):
                with open(selected_report, "r", encoding="utf-8") as file:
                    report_text = file.read()

                st.subheader("후속 처리 보고서")
                st.markdown(report_text)

                st.download_button(
                    label="후속 처리 보고서 TXT 다운로드",
                    data=report_text.encode("utf-8-sig"),
                    file_name=os.path.basename(selected_report),
                    mime="text/plain"
                )
            else:
                st.warning("선택한 후속 처리 보고서 파일을 찾을 수 없습니다.")

with tab4:
    st.subheader("반려 건 재작성")

    approval_df = load_approval_queue()

    if approval_df.empty:
        st.info("승인/반려 데이터가 없습니다.")
    else:
        rejected_df = approval_df[approval_df["approval_status"] == "반려"]

        if rejected_df.empty:
            st.success("현재 반려 상태인 건이 없습니다.")
        else:
            st.write("반려 상태인 건 목록")
            st.dataframe(rejected_df.sort_values("created_at", ascending=False), use_container_width=True)

            rejected_id = st.selectbox(
                "재작성할 반려 승인 ID 선택",
                rejected_df["approval_id"].tolist()
            )

            selected_row = rejected_df[rejected_df["approval_id"] == rejected_id].iloc[0]

            st.markdown("### 선택한 반려 건")
            st.json({
                "approval_id": selected_row.get("approval_id", ""),
                "issue_type": selected_row.get("issue_type", ""),
                "route_name": selected_row.get("route_name", ""),
                "severity": selected_row.get("severity", ""),
                "user_issue": selected_row.get("user_issue", ""),
                "review_comment": selected_row.get("review_comment", ""),
                "report_path": selected_row.get("report_path", "")
            })

            if st.button("반려 의견 반영해서 재작성"):
                try:
                    with st.spinner("반려 의견을 반영해 수정 안내문과 조치안을 재작성 중입니다..."):
                        regen = regenerate_rejected_case(rejected_id)

                    st.success(f"재작성 보고서 저장 완료: {regen['regenerated_report_path']}")

                    st.subheader("재작성 보고서")
                    st.markdown(regen["regenerated_report"])

                    st.download_button(
                        label="재작성 보고서 TXT 다운로드",
                        data=regen["regenerated_report"].encode("utf-8-sig"),
                        file_name="langgraph_regenerated_report.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error("반려 건 재작성 중 오류가 발생했습니다.")
                    st.write(str(e))

    st.divider()

    st.subheader("재작성 로그")

    regen_log = load_regeneration_log()

    if regen_log.empty:
        st.info("아직 재작성 로그가 없습니다.")
    else:
        st.dataframe(regen_log.sort_values("run_time", ascending=False), use_container_width=True)

        regen_paths = regen_log["regenerated_report_path"].dropna().tolist()

        if regen_paths:
            selected_regen_report = st.selectbox(
                "재작성 보고서 선택",
                regen_paths
            )

            if selected_regen_report and os.path.exists(selected_regen_report):
                with open(selected_regen_report, "r", encoding="utf-8") as file:
                    regen_text = file.read()

                st.subheader("저장된 재작성 보고서")
                st.markdown(regen_text)

                st.download_button(
                    label="저장된 재작성 보고서 TXT 다운로드",
                    data=regen_text.encode("utf-8-sig"),
                    file_name=os.path.basename(selected_regen_report),
                    mime="text/plain"
                )
