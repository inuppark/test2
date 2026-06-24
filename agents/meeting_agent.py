from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import json
import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_json_object(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"JSON 객체를 찾을 수 없습니다:\n{text}")

    return text[start:end + 1]

def analyze_meeting_text(meeting_text):
    if not meeting_text or not meeting_text.strip():
        raise ValueError("분석할 회의록 내용이 없습니다.")

    prompt = f"""
당신은 제조/판매 회사의 회의록 정리 AI 서기입니다.

아래 회의록을 분석해서 회의 요약, 결정사항, 액션아이템, 리스크, 확인 필요 사항을 정리하세요.

중요 규칙:
1. 문서에 없는 내용은 만들지 마세요.
2. 담당자가 명확하지 않으면 "담당자 미정"이라고 쓰세요.
3. 마감일이 없으면 "미정"이라고 쓰세요.
4. JSON 객체만 출력하세요.
5. 마크다운, 설명문, 코드블록은 출력하지 마세요.

출력 JSON 형식:
{{
  "meeting_title": "회의 제목",
  "summary": ["요약1", "요약2"],
  "decisions": [
    {{"decision": "결정사항", "reason": "결정 이유 또는 문서에서 확인되지 않음"}}
  ],
  "action_items": [
    {{"task": "할 일", "owner": "담당자", "due_date": "마감일", "priority": "낮음/보통/높음"}}
  ],
  "risks": [
    {{"risk": "리스크", "impact": "영향", "owner": "담당부서 또는 담당자 미정"}}
  ],
  "follow_up_questions": ["확인 필요 사항1", "확인 필요 사항2"],
  "executive_summary": "경영진 보고용 한 줄 요약"
}}

[회의록]
{meeting_text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = response.output_text.strip()
    json_text = extract_json_object(text)
    data = json.loads(json_text)

    summary_df = pd.DataFrame({
        "summary": data.get("summary", [])
    })

    decisions_df = pd.DataFrame(data.get("decisions", []))
    action_df = pd.DataFrame(data.get("action_items", []))
    risks_df = pd.DataFrame(data.get("risks", []))
    follow_up_df = pd.DataFrame({
        "question": data.get("follow_up_questions", [])
    })

    report = build_meeting_report(data)

    return report, summary_df, decisions_df, action_df, risks_df, follow_up_df

def build_meeting_report(data):
    lines = []

    lines.append("# 회의록 정리 보고서")
    lines.append("")
    lines.append("## 1. 회의 개요")
    lines.append(f"- 회의 제목: {data.get('meeting_title', '문서에서 확인되지 않음')}")
    lines.append("")

    lines.append("## 2. 핵심 요약")
    for item in data.get("summary", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 3. 결정사항")
    decisions = data.get("decisions", [])
    if decisions:
        for item in decisions:
            lines.append(f"- 결정: {item.get('decision', '')}")
            lines.append(f"  - 이유: {item.get('reason', '문서에서 확인되지 않음')}")
    else:
        lines.append("- 문서에서 확인되지 않음")
    lines.append("")

    lines.append("## 4. 액션아이템")
    actions = data.get("action_items", [])
    if actions:
        for item in actions:
            lines.append(f"- 할 일: {item.get('task', '')}")
            lines.append(f"  - 담당자: {item.get('owner', '담당자 미정')}")
            lines.append(f"  - 마감일: {item.get('due_date', '미정')}")
            lines.append(f"  - 우선순위: {item.get('priority', '보통')}")
    else:
        lines.append("- 문서에서 확인되지 않음")
    lines.append("")

    lines.append("## 5. 리스크")
    risks = data.get("risks", [])
    if risks:
        for item in risks:
            lines.append(f"- 리스크: {item.get('risk', '')}")
            lines.append(f"  - 영향: {item.get('impact', '')}")
            lines.append(f"  - 담당: {item.get('owner', '담당자 미정')}")
    else:
        lines.append("- 문서에서 확인되지 않음")
    lines.append("")

    lines.append("## 6. 확인 필요 사항")
    questions = data.get("follow_up_questions", [])
    if questions:
        for item in questions:
            lines.append(f"- {item}")
    else:
        lines.append("- 문서에서 확인되지 않음")
    lines.append("")

    lines.append("## 7. 경영진 보고용 한 줄 요약")
    lines.append(data.get("executive_summary", "문서에서 확인되지 않음"))

    return "\n".join(lines)

def save_meeting_outputs(report, decisions_df, action_df, risks_df, follow_up_df):
    os.makedirs("reports", exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M")

    report_path = f"reports/meeting_report_{now}.txt"
    decisions_path = f"reports/meeting_decisions_{now}.csv"
    actions_path = f"reports/meeting_actions_{now}.csv"
    risks_path = f"reports/meeting_risks_{now}.csv"
    follow_up_path = f"reports/meeting_follow_up_{now}.csv"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    decisions_df.to_csv(decisions_path, index=False, encoding="utf-8-sig")
    action_df.to_csv(actions_path, index=False, encoding="utf-8-sig")
    risks_df.to_csv(risks_path, index=False, encoding="utf-8-sig")
    follow_up_df.to_csv(follow_up_path, index=False, encoding="utf-8-sig")

    return report_path, decisions_path, actions_path, risks_path, follow_up_path
