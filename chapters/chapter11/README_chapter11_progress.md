# Chapter 11 진행 기록

## 11-1 ReAct Agent Basic

### 목적

책의 11.1 "자율적 에이전트와 ReAct" 개념을 앳플리 업무 기준으로 실습한다.

ReAct(Reasoning + Acting) 흐름을 Python 코드 수준에서 명확히 드러내는 콘솔 실습이다.
MCP 서버나 외부 API 없이 기존 `utils/rag_utils.py`와 규칙 기반 분류를 재사용한다.

---

### 생성 파일

| 파일 | 설명 |
|---|---|
| `chapters/chapter11/01_atflee_react_agent_basic.py` | ReAct 흐름 콘솔 실습 |
| `chapters/chapter11/README_chapter11_progress.md` | 이 문서 |

---

### 실행 명령어

```bash
python chapters/chapter11/01_atflee_react_agent_basic.py
```

사전 조건:
- `.env`에 `ANTHROPIC_API_KEY` 설정
- `data/wiki/*.md` 파일 존재

---

### ReAct 흐름

```
[User]         사용자 문의 입력
    ↓
[Thought]      의도 유형·심각도 판단 + 필요 도구 설명
    ↓
[Action Plan]  실행할 도구 목록 결정
    ↓
[Action 1]     classify_customer_intent  (키워드 기반 의도 분류)
[Action 2]     search_atflee_wiki        (data/wiki 키워드 검색)
[Action 3]     generate_final_answer     (Claude 최종 답변 생성)
    ↓
[Observation]  도구 결과 요약
    ↓
[Final Answer] Claude 답변 출력
    ↓
[Safety Check] evaluate_safety           (위험 표현·개인정보 요청 점검)
```

---

### 도구 목록

| 도구 | 역할 |
|---|---|
| `classify_customer_intent` | 의도(앱연결/배송/교환/AS/제품품질/고객센터/기타), 심각도, 담당자 필요 여부 |
| `search_atflee_wiki` | `data/wiki` 키워드 검색, TOP 3 문서 + snippet 반환 |
| `evaluate_safety` | 답변에 단정 표현·개인정보 요청·참고 문서 누락 여부 점검 |

---

### 테스트 질문 4개

| 질문 | 예상 intent | 위키 검색 |
|---|---|---|
| 체중계 앱 연결 안 됨 | 앱 연결 | atflee_app_guide |
| 제품 불량 교환 | 제품 품질 or 교환/반품 | product_quality_policy, customer_service_policy |
| AS 접수 확인 | AS | customer_service_policy |
| 고객센터 전화번호 | 고객센터 문의 | atflee_contact_guide |

---

### Chapter 8 Tool Use Agent와의 차이

| 항목 | Chapter 8 Tool Use | Chapter 11 ReAct |
|---|---|---|
| 도구 실행 주체 | Claude API `tool_use` 블록이 직접 선택 | Python 코드가 규칙으로 선택 |
| 흐름 가시성 | 도구 선택 근거가 API 내부에 있음 | Thought/Action/Observation이 콘솔에 명시적 출력 |
| 확장 방향 | Anthropic SDK tool 스키마 기반 | MCP 서버 / 외부 API 연결로 확장 가능 |

---

### 다음 단계: 11-2

Claude `tool_use` API를 이용해 Python 규칙 없이 Claude 스스로 도구를 선택하는 에이전트 구현.

- Claude에게 도구 스키마(`tools=[]`)를 전달
- Claude가 `tool_use` 블록으로 도구를 선택
- Python이 해당 도구를 실행하고 `tool_result`로 결과를 돌려줌
- Claude가 최종 답변 생성

이 구조가 진정한 자율 ReAct 에이전트이며, MCP 서버 연동 전 단계이다.
