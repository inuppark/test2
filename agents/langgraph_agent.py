from typing import TypedDict, List, Dict, Any
from datetime import datetime
import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from agents.rag_agent import answer_with_rag

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

APPROVAL_QUEUE_PATH = "data/approval_queue.csv"

class WorkflowState(TypedDict, total=False):
    user_issue: str
    issue_type: str
    search_scope: str
    route_name: str
    rag_answer: str
    confidence_level: str
    top_score: float
    sources: List[Dict[str, Any]]
    severity: str
    severity_reason: str
    approval_required: str
    approval_reason: str
    approval_status: str
    approval_id: str
    action_plan: str
    final_report: str

def classify_issue_text(text: str) -> str:
    lower = text.lower()

    if any(word in lower for word in ["배송", "오배송", "택배", "출고", "도착", "포장", "파손", "옵션"]):
        return "배송/물류"

    if any(word in lower for word in ["하자", "고장", "불량", "품질", "배터리", "색상", "기능", "스크래치", "손상", "안전"]):
        return "제품품질"

    if any(word in lower for word in ["환불", "교환", "반품", "취소", "보상"]):
        return "교환/환불"

    if any(word in lower for word in ["설명서", "사용법", "설치", "초기화", "가이드", "사이즈"]):
        return "사용안내"

    if any(word in lower for word in ["상담", "고객센터", "문의", "응대", "연결", "대기", "as", "AS"]):
        return "고객서비스"

    return "기타"

def classify_node(state: WorkflowState) -> WorkflowState:
    issue = state["user_issue"]
    issue_type = classify_issue_text(issue)

    return {
        **state,
        "issue_type": issue_type
    }

def route_by_issue_type(state: WorkflowState) -> str:
    issue_type = state.get("issue_type", "기타")

    if issue_type == "배송/물류":
        return "logistics_route"

    if issue_type == "제품품질":
        return "quality_route"

    if issue_type == "교환/환불":
        return "exchange_route"

    if issue_type == "사용안내":
        return "guide_route"

    if issue_type == "고객서비스":
        return "cs_route"

    return "general_route"

def logistics_route_node(state: WorkflowState) -> WorkflowState:
    return {**state, "route_name": "배송/물류 전용 흐름", "search_scope": "전체"}

def quality_route_node(state: WorkflowState) -> WorkflowState:
    return {**state, "route_name": "제품품질 전용 흐름", "search_scope": "전체"}

def exchange_route_node(state: WorkflowState) -> WorkflowState:
    return {**state, "route_name": "교환/환불 전용 흐름", "search_scope": "전체"}

def guide_route_node(state: WorkflowState) -> WorkflowState:
    return {**state, "route_name": "사용안내 전용 흐름", "search_scope": "전체"}

def cs_route_node(state: WorkflowState) -> WorkflowState:
    return {**state, "route_name": "고객서비스 전용 흐름", "search_scope": "전체"}

def general_route_node(state: WorkflowState) -> WorkflowState:
    return {**state, "route_name": "일반 이슈 흐름", "search_scope": "전체"}

def make_rag_question(state: WorkflowState) -> str:
    issue_type = state.get("issue_type", "기타")
    user_issue = state.get("user_issue", "")
    route_name = state.get("route_name", "")

    if issue_type == "배송/물류":
        focus = "배송 지연, 오배송, 포장 훼손, 옵션 오배송, 물류 담당 부서, 고객 안내 절차, 교환 가능성을 중심으로 찾아주세요."
    elif issue_type == "제품품질":
        focus = "제품 하자, 고장, 손상, 색상 차이, 배터리, 기능 문제, 품질관리팀 확인 절차, AS 또는 교환 가능성을 중심으로 찾아주세요."
    elif issue_type == "교환/환불":
        focus = "교환 절차, 환불 절차, 반품 기준, 고객 안내, 고객서비스팀 처리 기준, 승인 필요 여부를 중심으로 찾아주세요."
    elif issue_type == "사용안내":
        focus = "사용법, 설명서, 설치, 초기화, 사이즈 선택, 안내 콘텐츠 개선 기준, FAQ 전환 가능성을 중심으로 찾아주세요."
    elif issue_type == "고객서비스":
        focus = "상담 대기, 고객 응대, AS 접수, 진행상황 안내, 고객서비스팀 조치 기준, 고객 불만 완화 방안을 중심으로 찾아주세요."
    else:
        focus = "관련 사내 기준, 최근 보고서, 담당 부서, 고객 안내 절차, 승인 필요 여부를 중심으로 찾아주세요."

    return f"""
다음 고객 이슈에 대해 회사 기준 문서와 최근 보고서를 함께 참고해서 처리 기준을 찾아주세요.

[워크플로 흐름]
{route_name}

[이슈 유형]
{issue_type}

[중점 검색 기준]
{focus}

[고객 이슈]
{user_issue}
"""

def rag_search_node(state: WorkflowState) -> WorkflowState:
    question = make_rag_question(state)

    answer, search_results, confidence = answer_with_rag(
        question=question,
        top_k=5,
        search_scope=state.get("search_scope", "전체")
    )

    sources = []

    for item in search_results:
        sources.append({
            "source_group": item.get("source_group"),
            "report_type": item.get("report_type"),
            "source_file": item.get("source_file"),
            "chunk_id": item.get("chunk_id"),
            "score": round(item.get("score", 0), 4)
        })

    return {
        **state,
        "rag_answer": answer,
        "confidence_level": confidence.get("level"),
        "top_score": confidence.get("top_score"),
        "sources": sources
    }

def judge_severity_rule(issue: str, issue_type: str, rag_answer: str) -> Dict[str, str]:
    text = f"{issue}\n{rag_answer}".lower()

    emergency_keywords = [
        "안전", "화재", "폭발", "감전", "부상", "상해", "위험",
        "대량", "집단", "법적", "소송", "언론", "브랜드 리스크"
    ]

    high_keywords = [
        "하자", "고장", "불량", "파손", "손상", "배터리", "기능",
        "오배송", "옵션", "포장", "환불", "교환", "as",
        "진행 상황 안내", "반복", "고객 불만"
    ]

    normal_keywords = [
        "설명서", "사용법", "설치", "초기화", "가이드", "사이즈",
        "안내", "문의"
    ]

    if any(keyword.lower() in text for keyword in emergency_keywords):
        return {
            "severity": "긴급",
            "reason": "안전, 법적, 대량 클레임 또는 브랜드 리스크 가능성이 포함되어 긴급 대응이 필요합니다."
        }

    if issue_type in ["제품품질", "배송/물류", "교환/환불"] and any(keyword.lower() in text for keyword in high_keywords):
        return {
            "severity": "높음",
            "reason": "제품 하자, 배송/오배송, 교환/환불 등 고객 만족도와 비용에 직접 영향을 주는 이슈입니다."
        }

    if issue_type == "고객서비스":
        return {
            "severity": "높음",
            "reason": "상담 지연 또는 AS 안내 누락은 고객 불만 확대 가능성이 있어 우선 대응이 필요합니다."
        }

    if issue_type == "사용안내" or any(keyword in text for keyword in normal_keywords):
        return {
            "severity": "보통",
            "reason": "사용법 또는 안내 부족 이슈로 즉시 위험도는 낮지만 반복 문의로 확산될 수 있습니다."
        }

    return {
        "severity": "낮음",
        "reason": "현재 내용만으로는 중대한 품질, 배송, 보상 또는 안전 이슈가 확인되지 않습니다."
    }

def judge_approval_rule(issue_type: str, severity: str, issue: str) -> Dict[str, str]:
    text = issue.lower()

    approval_keywords = [
        "환불", "교환", "보상", "파손", "오배송", "하자", "고장",
        "불량", "안전", "법적", "소송", "대량", "브랜드"
    ]

    if severity in ["긴급", "높음"]:
        return {
            "approval_required": "필요",
            "reason": "심각도가 높음 이상이므로 담당자 또는 관리자 승인 후 처리하는 것이 안전합니다."
        }

    if any(keyword in text for keyword in approval_keywords):
        return {
            "approval_required": "필요",
            "reason": "교환, 환불, 보상, 제품 하자 또는 배송 사고 관련 판단이 포함되어 승인 절차가 필요합니다."
        }

    if issue_type in ["사용안내", "기타"] and severity in ["낮음", "보통"]:
        return {
            "approval_required": "불필요",
            "reason": "단순 안내 또는 FAQ성 문의로 판단되어 표준 응대 범위 내에서 처리 가능합니다."
        }

    return {
        "approval_required": "검토 필요",
        "reason": "자동 판단만으로 승인 필요 여부를 확정하기 어려워 담당자 확인이 필요합니다."
    }

def severity_node(state: WorkflowState) -> WorkflowState:
    result = judge_severity_rule(
        issue=state.get("user_issue", ""),
        issue_type=state.get("issue_type", ""),
        rag_answer=state.get("rag_answer", "")
    )

    return {
        **state,
        "severity": result["severity"],
        "severity_reason": result["reason"]
    }

def approval_node(state: WorkflowState) -> WorkflowState:
    result = judge_approval_rule(
        issue_type=state.get("issue_type", ""),
        severity=state.get("severity", ""),
        issue=state.get("user_issue", "")
    )

    approval_status = "대기" if result["approval_required"] == "필요" else "자동처리 가능"

    approval_id = datetime.now().strftime("APV-%Y%m%d-%H%M%S")

    return {
        **state,
        "approval_required": result["approval_required"],
        "approval_reason": result["reason"],
        "approval_status": approval_status,
        "approval_id": approval_id
    }

def action_plan_node(state: WorkflowState) -> WorkflowState:
    prompt = f"""
당신은 제조/판매 회사의 CS 운영 매니저입니다.

아래 고객 이슈와 RAG 검색 결과, 심각도 판단, 승인 필요 여부를 바탕으로 실무 조치안을 작성하세요.

[고객 이슈]
{state["user_issue"]}

[이슈 유형]
{state.get("issue_type")}

[적용된 워크플로]
{state.get("route_name")}

[RAG 검색 답변]
{state.get("rag_answer")}

[검색 신뢰도]
{state.get("confidence_level")} / {state.get("top_score")}

[심각도 판단]
심각도: {state.get("severity")}
판단 이유: {state.get("severity_reason")}

[승인 필요 여부]
승인 필요 여부: {state.get("approval_required")}
승인 상태: {state.get("approval_status")}
판단 이유: {state.get("approval_reason")}

작성 원칙:
- 이슈 유형에 맞는 담당 부서를 명확히 제안하세요.
- 고객에게 바로 안내할 문구를 실무적으로 작성하세요.
- 확인되지 않은 내용은 추가 확인 항목으로 분리하세요.
- 승인 필요 여부가 필요이면, 승인권자 후보와 승인 전 확인 항목을 포함하세요.
- 우선순위는 낮음/보통/높음/긴급 중 하나로 판단하세요.

아래 형식으로 작성하세요.

## 조치안

### 1. 고객에게 바로 안내할 내용

### 2. 내부 확인 항목

### 3. 담당 부서 후보

### 4. 우선순위

### 5. 승인 필요 여부 및 승인 전 확인 항목

### 6. 후속 액션
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        **state,
        "action_plan": response.output_text
    }

def final_report_node(state: WorkflowState) -> WorkflowState:
    source_lines = []

    for item in state.get("sources", []):
        source_lines.append(
            f"- {item.get('source_file')} / {item.get('source_group')} / {item.get('report_type')} / score={item.get('score')}"
        )

    source_text = "\n".join(source_lines)

    report = f"""
# LangGraph 고객 이슈 처리 보고서

## 1. 입력 이슈
{state.get("user_issue")}

## 2. 이슈 분류
- 분류: {state.get("issue_type")}
- 적용 워크플로: {state.get("route_name")}

## 3. RAG 검색 신뢰도
- 신뢰도: {state.get("confidence_level")}
- 최고 유사도: {state.get("top_score")}

## 4. 심각도 판단
- 심각도: {state.get("severity")}
- 판단 이유: {state.get("severity_reason")}

## 5. 승인 필요 여부
- 승인 ID: {state.get("approval_id")}
- 승인 필요 여부: {state.get("approval_required")}
- 승인 상태: {state.get("approval_status")}
- 판단 이유: {state.get("approval_reason")}

## 6. 근거 문서
{source_text}

## 7. RAG 기준 검색 결과
{state.get("rag_answer")}

## 8. 실무 조치안
{state.get("action_plan")}
"""

    return {
        **state,
        "final_report": report
    }

def build_langgraph_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("classify_issue", classify_node)

    graph.add_node("logistics_route", logistics_route_node)
    graph.add_node("quality_route", quality_route_node)
    graph.add_node("exchange_route", exchange_route_node)
    graph.add_node("guide_route", guide_route_node)
    graph.add_node("cs_route", cs_route_node)
    graph.add_node("general_route", general_route_node)

    graph.add_node("rag_search", rag_search_node)
    graph.add_node("judge_severity", severity_node)
    graph.add_node("judge_approval", approval_node)
    graph.add_node("make_action_plan", action_plan_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "classify_issue")

    graph.add_conditional_edges(
        "classify_issue",
        route_by_issue_type,
        {
            "logistics_route": "logistics_route",
            "quality_route": "quality_route",
            "exchange_route": "exchange_route",
            "guide_route": "guide_route",
            "cs_route": "cs_route",
            "general_route": "general_route"
        }
    )

    graph.add_edge("logistics_route", "rag_search")
    graph.add_edge("quality_route", "rag_search")
    graph.add_edge("exchange_route", "rag_search")
    graph.add_edge("guide_route", "rag_search")
    graph.add_edge("cs_route", "rag_search")
    graph.add_edge("general_route", "rag_search")

    graph.add_edge("rag_search", "judge_severity")
    graph.add_edge("judge_severity", "judge_approval")
    graph.add_edge("judge_approval", "make_action_plan")
    graph.add_edge("make_action_plan", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()

def run_langgraph_workflow(user_issue: str) -> WorkflowState:
    if not user_issue or not user_issue.strip():
        raise ValueError("고객 이슈 내용이 비어 있습니다.")

    app = build_langgraph_workflow()

    result = app.invoke({
        "user_issue": user_issue
    })

    return result

def save_approval_queue(result: WorkflowState, report_path: str):
    os.makedirs("data", exist_ok=True)

    if result.get("approval_required") != "필요":
        return None

    row = pd.DataFrame([{
        "approval_id": result.get("approval_id"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "issue_type": result.get("issue_type"),
        "route_name": result.get("route_name"),
        "severity": result.get("severity"),
        "approval_required": result.get("approval_required"),
        "approval_status": result.get("approval_status"),
        "top_score": result.get("top_score"),
        "user_issue": result.get("user_issue"),
        "report_path": report_path,
        "reviewer": "",
        "reviewed_at": "",
        "review_comment": ""
    }])

    if os.path.exists(APPROVAL_QUEUE_PATH):
        old = pd.read_csv(APPROVAL_QUEUE_PATH)
        new = pd.concat([old, row], ignore_index=True)
    else:
        new = row

    new.to_csv(APPROVAL_QUEUE_PATH, index=False, encoding="utf-8-sig")

    return APPROVAL_QUEUE_PATH

def save_langgraph_log(result: WorkflowState):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_path = f"reports/langgraph_issue_report_{now}.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(result.get("final_report", ""))

    approval_queue_path = save_approval_queue(result, report_path)

    log_path = "logs/langgraph_agent_log.csv"

    row = pd.DataFrame([{
        "run_time": run_time,
        "issue_type": result.get("issue_type"),
        "route_name": result.get("route_name"),
        "severity": result.get("severity"),
        "approval_required": result.get("approval_required"),
        "approval_status": result.get("approval_status"),
        "approval_id": result.get("approval_id"),
        "confidence_level": result.get("confidence_level"),
        "top_score": result.get("top_score"),
        "report_path": report_path,
        "approval_queue_path": approval_queue_path,
        "status": "success"
    }])

    if os.path.exists(log_path):
        old = pd.read_csv(log_path)
        new = pd.concat([old, row], ignore_index=True)
    else:
        new = row

    new.to_csv(log_path, index=False, encoding="utf-8-sig")

    return report_path, log_path, approval_queue_path

def update_approval_status(approval_id: str, new_status: str, reviewer: str, comment: str):
    if not os.path.exists(APPROVAL_QUEUE_PATH):
        raise ValueError("?? ??? ??? ????.")

    # ?? ??? ???? ??? reviewer/review_comment ?? ? dtype ?? ??
    df = pd.read_csv(APPROVAL_QUEUE_PATH, dtype=str, keep_default_na=False)

    required_columns = [
        "approval_id",
        "approval_status",
        "reviewer",
        "reviewed_at",
        "review_comment"
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df["approval_id"] = df["approval_id"].astype(str)
    approval_id = str(approval_id)

    if approval_id not in df["approval_id"].values:
        raise ValueError("?? ?? ID? ?? ? ????.")

    mask = df["approval_id"] == approval_id

    # ?? ??? ????? object/string ??
    for col in ["approval_status", "reviewer", "reviewed_at", "review_comment"]:
        df[col] = df[col].astype(str)

    df.loc[mask, "approval_status"] = str(new_status)
    df.loc[mask, "reviewer"] = str(reviewer)
    df.loc[mask, "reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[mask, "review_comment"] = str(comment)

    df.to_csv(APPROVAL_QUEUE_PATH, index=False, encoding="utf-8-sig")

    followup_result = save_followup_result_safe(
        approval_id=approval_id,
        new_status=new_status,
        reviewer=reviewer,
        comment=comment
    )

    return {
        "approval_queue_path": APPROVAL_QUEUE_PATH,
        "followup_result": followup_result
    }


def load_approval_queue():
    if not os.path.exists(APPROVAL_QUEUE_PATH):
        return pd.DataFrame()

    return pd.read_csv(APPROVAL_QUEUE_PATH)


def save_followup_result(approval_id: str, new_status: str, reviewer: str, comment: str):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(APPROVAL_QUEUE_PATH):
        raise ValueError("?? ??? ??? ????.")

    df = pd.read_csv(APPROVAL_QUEUE_PATH, dtype=str, keep_default_na=False)

    if approval_id not in df["approval_id"].astype(str).values:
        raise ValueError("?? ?? ID? ?? ? ????.")

    row = df[df["approval_id"].astype(str) == str(approval_id)].iloc[0]

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    issue_type = row.get("issue_type", "")
    severity = row.get("severity", "")
    user_issue = row.get("user_issue", "")
    report_path = row.get("report_path", "")

    if new_status == "??":
        followup_status = "?? ? ??"
        customer_message = f"""
?????. ?? ?? ?? ??????.

????? ??? ?? ??? ?? ?? ?? ? ?? ??? ??? ???????.
?? ?? ??? ?? ??? ???? ?? ??? ???????.

?? ???? ?? ??? ??? ?? ?? ????????.
"""
        next_action = """
- ?? ??? ?? ?? ??
- ?? ??? ??
- ??/??/AS ? ??? ?? ?? ??
- ?? ?? ? ?? ??
"""
        revision_request = ""

    elif new_status == "??":
        followup_status = "??? ??"
        customer_message = ""
        next_action = """
- ?? ??? ???
- ??? ?? ?? ?? ??
- ??? ?? ?? ??
- ??? ??? ???
"""
        revision_request = f"""
?? ??:
{comment}

?? ?? ??:
- ?? ?? ?? ???
- ?? ?? ? ?? ?? ???
- ?? ?? ??
- ???? ?? ??? ???
"""
    else:
        followup_status = "?? ?? ??"
        customer_message = ""
        next_action = "?? ??? ?????."
        revision_request = comment

    followup_report = f"""
# LangGraph ?? ?? ?? ???

## 1. ?? ??
- ?? ID: {approval_id}
- ?? ??: {new_status}
- ?? ?? ??: {followup_status}
- ???: {reviewer}
- ?? ??: {run_time}

## 2. ? ?? ??
- ?? ??: {issue_type}
- ???: {severity}
- ?? ??:
{user_issue}

## 3. ?? ??
{comment}

## 4. ?? ???
{customer_message if customer_message else "?? ??? ?? ???? ???? ?????."}

## 5. ?? ??
{next_action}

## 6. ??? ??
{revision_request if revision_request else "?? ??"}

## 7. ? ??? ??
{report_path}
"""

    followup_report_path = f"reports/langgraph_followup_{approval_id}_{now}.txt"

    with open(followup_report_path, "w", encoding="utf-8") as file:
        file.write(followup_report)

    followup_log_path = "logs/langgraph_followup_log.csv"

    log_row = pd.DataFrame([{
        "run_time": run_time,
        "approval_id": approval_id,
        "approval_result": new_status,
        "followup_status": followup_status,
        "reviewer": reviewer,
        "issue_type": issue_type,
        "severity": severity,
        "original_report_path": report_path,
        "followup_report_path": followup_report_path,
        "status": "success"
    }])

    if os.path.exists(followup_log_path):
        old = pd.read_csv(followup_log_path, dtype=str, keep_default_na=False)
        new = pd.concat([old, log_row], ignore_index=True)
    else:
        new = log_row

    new.to_csv(followup_log_path, index=False, encoding="utf-8-sig")

    return {
        "followup_status": followup_status,
        "followup_report": followup_report,
        "followup_report_path": followup_report_path,
        "followup_log_path": followup_log_path
    }


def save_followup_result_safe(approval_id: str, new_status: str, reviewer: str, comment: str):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    def T(codes):
        return "".join(chr(c) for c in codes)

    if not os.path.exists(APPROVAL_QUEUE_PATH):
        raise ValueError("approval queue file does not exist")

    df = pd.read_csv(APPROVAL_QUEUE_PATH, dtype=str, keep_default_na=False)

    if approval_id not in df["approval_id"].astype(str).values:
        raise ValueError("approval id not found")

    row = df[df["approval_id"].astype(str) == str(approval_id)].iloc[0]

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    issue_type = row.get("issue_type", "")
    severity = row.get("severity", "")
    user_issue = row.get("user_issue", "")
    report_path = row.get("report_path", "")

    approved = T([49849, 51064])
    rejected = T([48152, 47140])

    title = "# LangGraph " + T([49849, 51064, 32, 54980, 49549, 32, 52376, 47532, 32, 48372, 44256, 49436])

    if str(new_status) == approved:
        followup_status = T([49849, 51064, 32, 54980, 32, 51652, 54665])
        customer_message = "\n".join([
            T([50504,45397,54616,49464,50836,46,32,47928,51032,32,51452,49888,32,45236,50857,32,54869,51064,54664,49845,45768,45796,46]),
            "",
            T([45236,48512,32,49849,51064,51060,32,50756,47308,46104,50612,32,45812,45817,32,48512,49436,32,54980,49549,32,51312,52824,47484,32,51652,54665,54616,44192,49845,45768,45796,46]),
            T([52628,44032,32,54869,51064,51060,32,54596,50836,54620,32,44221,50864,32,45796,49884,32,50504,45236,46300,47532,44192,49845,45768,45796,46])
        ])
        next_action = "\n".join([
            T([45,32,45812,45817,32,48512,49436,50640,32,52376,47532,32,50836,52397,32,51204,45804]),
            T([45,32,44256,44061,32,50504,45236,47928,32,48156,49569]),
            T([45,32,54596,50836,54620,32,54980,49549,32,51312,52824,32,51652,54665]),
            T([45,32,52376,47532,32,50756,47308,32,54980,32,44208,44284,32,44592,47197])
        ])
        revision_request = T([54644,45817,32,50630,51020])

    elif str(new_status) == rejected:
        followup_status = T([51116,51089,49457,32,54596,50836])
        customer_message = T([48152,47140,32,44148,51004,47196,32,44256,44061,32,50504,45236,47928,51008,32,54869,51221,46104,51648,32,50506,50520,49845,45768,45796,46])
        next_action = "\n".join([
            T([45,32,44592,51316,32,51312,52824,50504,32,51116,44160,53664]),
            T([45,32,48512,51313,54620,32,44540,44144,32,47928,49436,32,52628,44032,32,54869,51064]),
            T([45,32,45812,45817,51088,32,44160,53664,32,51032,44204,32,48152,50689]),
            T([45,32,49688,51221,46108,32,51312,52824,50504,32,51116,49373,49457])
        ])
        revision_request = "\n".join([
            T([48152,47140,32,51032,44204,58]),
            str(comment),
            "",
            T([49688,51221,32,54596,50836,32,54637,47785,58]),
            T([45,32,44256,44061,32,50504,45236,32,47928,44396,32,51116,44160,53664]),
            T([45,32,45812,45817,32,48512,49436,32,48143,32,49849,51064,32,44592,51456,32,51116,54869,51064]),
            T([45,32,44540,44144,32,47928,49436,32,48372,44053]),
            T([45,32,51312,52824,50504,32,49892,54665,32,44032,45733,49457,32,51116,44160,53664])
        ])
    else:
        followup_status = T([44160,53664,32,49345,53468,32,48120,51221])
        customer_message = T([52628,44032,32,44160,53664,44032,32,54596,50836,54633,45768,45796,46])
        next_action = customer_message
        revision_request = str(comment)

    lines = [
        title,
        "",
        "## 1. " + T([49849,51064,32,51221,48372]),
        "- " + T([49849,51064,32,73,68]) + ": " + str(approval_id),
        "- " + T([49849,51064,32,44208,44284]) + ": " + str(new_status),
        "- " + T([54980,49549,32,52376,47532,32,49345,53468]) + ": " + str(followup_status),
        "- " + T([44160,53664,51088]) + ": " + str(reviewer),
        "- " + T([44160,53664,32,49884,44036]) + ": " + str(run_time),
        "",
        "## 2. " + T([50896,32,51060,49800,32,51221,48372]),
        "- " + T([51060,49800,32,50976,54805]) + ": " + str(issue_type),
        "- " + T([49900,44033,46020]) + ": " + str(severity),
        "- " + T([44256,44061,32,51060,49800]) + ":",
        str(user_issue),
        "",
        "## 3. " + T([44160,53664,32,51032,44204]),
        str(comment),
        "",
        "## 4. " + T([44256,44061,32,50504,45236,47928]),
        str(customer_message),
        "",
        "## 5. " + T([54980,49549,32,50529,49496]),
        str(next_action),
        "",
        "## 6. " + T([51116,44160,53664,32,50836,52397]),
        str(revision_request),
        "",
        "## 7. " + T([50896,32,48372,44256,49436,32,44221,47196]),
        str(report_path),
        ""
    ]

    followup_report = "\n".join(lines)

    followup_report_path = f"reports/langgraph_followup_{approval_id}_{now}.txt"

    with open(followup_report_path, "w", encoding="utf-8") as file:
        file.write(followup_report)

    followup_log_path = "logs/langgraph_followup_log.csv"

    log_row = pd.DataFrame([{
        "run_time": run_time,
        "approval_id": str(approval_id),
        "approval_result": str(new_status),
        "followup_status": str(followup_status),
        "reviewer": str(reviewer),
        "issue_type": str(issue_type),
        "severity": str(severity),
        "original_report_path": str(report_path),
        "followup_report_path": str(followup_report_path),
        "status": "success"
    }])

    if os.path.exists(followup_log_path):
        old = pd.read_csv(followup_log_path, dtype=str, keep_default_na=False)
        new = pd.concat([old, log_row], ignore_index=True)
    else:
        new = log_row

    new.to_csv(followup_log_path, index=False, encoding="utf-8-sig")

    return {
        "followup_status": followup_status,
        "followup_report": followup_report,
        "followup_report_path": followup_report_path,
        "followup_log_path": followup_log_path
    }


def regenerate_rejected_case(approval_id: str):
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    def T(codes):
        return "".join(chr(c) for c in codes)

    if not os.path.exists(APPROVAL_QUEUE_PATH):
        raise ValueError("approval queue file does not exist")

    df = pd.read_csv(APPROVAL_QUEUE_PATH, dtype=str, keep_default_na=False)

    if approval_id not in df["approval_id"].astype(str).values:
        raise ValueError("approval id not found")

    row = df[df["approval_id"].astype(str) == str(approval_id)].iloc[0]

    approval_status = row.get("approval_status", "")
    review_comment = row.get("review_comment", "")
    user_issue = row.get("user_issue", "")
    issue_type = row.get("issue_type", "")
    severity = row.get("severity", "")
    route_name = row.get("route_name", "")
    report_path = row.get("report_path", "")

    rejected = T([48152, 47140])

    if str(approval_status) != rejected:
        raise ValueError("only rejected approval cases can be regenerated")

    original_report_text = ""

    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as file:
            original_report_text = file.read()

    prompt = f"""
You are a CS operations manager for a manufacturing and sales company.

The following customer issue was rejected during approval review.
Reflect the rejection comment and rewrite the customer message and internal action plan.

Write the final answer in Korean.

[Approval ID]
{approval_id}

[Customer Issue]
{user_issue}

[Issue Type]
{issue_type}

[Severity]
{severity}

[Workflow Route]
{route_name}

[Rejection Comment]
{review_comment}

[Original Report]
{original_report_text[:8000]}

Rules:
1. Reflect the rejection comment.
2. Do not over-confirm exchange/refund/compensation before internal review.
3. Separate confirmed guidance from items requiring additional verification.
4. Include customer message, internal action plan, responsible departments, and next actions.
5. Decide whether re-approval is required.

Use this Korean report format:

# LangGraph ?? ? ??? ???

## 1. ??? ??
- ?? ID:
- ?? ??:
- ???:

## 2. ?? ?? ??

## 3. ?? ?? ???

## 4. ?? ?? ???

## 5. ?? ?? ? ?? ??

## 6. ?? ??? ??

## 7. ?? ?? ??
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    regenerated_report = response.output_text

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    regenerated_report_path = f"reports/langgraph_regenerated_{approval_id}_{now}.txt"

    with open(regenerated_report_path, "w", encoding="utf-8") as file:
        file.write(regenerated_report)

    log_path = "logs/langgraph_regeneration_log.csv"

    log_row = pd.DataFrame([{
        "run_time": run_time,
        "approval_id": str(approval_id),
        "issue_type": str(issue_type),
        "severity": str(severity),
        "approval_status": str(approval_status),
        "review_comment": str(review_comment),
        "original_report_path": str(report_path),
        "regenerated_report_path": str(regenerated_report_path),
        "status": "success"
    }])

    if os.path.exists(log_path):
        old = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        new = pd.concat([old, log_row], ignore_index=True)
    else:
        new = log_row

    new.to_csv(log_path, index=False, encoding="utf-8-sig")

    return {
        "regenerated_report": regenerated_report,
        "regenerated_report_path": regenerated_report_path,
        "log_path": log_path
    }



def load_regeneration_log():
    log_path = "logs/langgraph_regeneration_log.csv"

    if not os.path.exists(log_path):
        return pd.DataFrame()

    return pd.read_csv(log_path, dtype=str, keep_default_na=False)
