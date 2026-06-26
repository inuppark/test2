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

MCP 서버 구조 실습 → `02_atflee_mcp_server_basic.py` 참고

---

## 11-2 MCP Server Basic

### 목적

앳플리 도구들을 MCP(Model Context Protocol) 서버 형태로 외부에 노출한다.
11-1에서 Python 코드 안에서 직접 실행했던 도구들을 표준 MCP tool로 감싼다.
Claude Desktop 또는 MCP 클라이언트가 연결해 앳플리 도구를 표준 프로토콜로 호출할 수 있다.

---

### 생성 파일

| 파일 | 설명 |
|---|---|
| `chapters/chapter11/02_atflee_mcp_server_basic.py` | FastMCP 기반 앳플리 MCP 서버 v0 |

---

### 주요 도구 (MCP tool)

| 도구 | 설명 |
|---|---|
| `get_atflee_status` | 서버 상태, 위키 문서 수, 사용 가능 도구 목록 반환 |
| `list_atflee_tools` | 등록된 MCP 도구 이름과 설명 목록 반환 |
| `search_atflee_wiki` | `data/wiki` 문서를 키워드 스코어링으로 TOP K 검색 |
| `classify_atflee_voc` | 고객 문의를 의도/심각도/담당팀 기준으로 규칙 분류 |

---

### 실행 명령어

```bash
# MCP 서버 실행 (Ctrl+C로 종료)
python chapters/chapter11/02_atflee_mcp_server_basic.py

# 문법 검증만
python -m py_compile chapters/chapter11/02_atflee_mcp_server_basic.py
```

사전 조건:
- `pip install fastmcp` 완료 (requirements.txt에 추가됨)
- `data/wiki/*.md` 파일 존재

---

### 11-1 ReAct vs 11-2 MCP Server 차이

| 항목 | 11-1 ReAct (콘솔 실습) | 11-2 MCP Server |
|---|---|---|
| 도구 실행 방식 | Python 코드가 직접 함수 호출 | MCP 클라이언트가 표준 프로토콜로 호출 |
| 외부 연결 | 없음 (단독 실행) | Claude Desktop, MCP 클라이언트 연결 가능 |
| 확장성 | 파일 내 함수 추가로만 확장 | `@mcp.tool` 데코레이터로 어디서나 도구 추가 가능 |

---

### 다음 단계: 11-3

MCP 클라이언트 테스트 → `03_atflee_mcp_client_test.py` 참고

---

## 11-3 MCP Client Test

### 목적

11-2에서 만든 MCP 서버에 Python MCP 클라이언트로 연결해
등록된 도구 목록을 확인하고 실제 tool call을 실행한다.

Claude API, ANTHROPIC_API_KEY, UPSTAGE_API_KEY를 사용하지 않는다.
MCP 서버와 MCP 클라이언트 연결 구조 자체를 이해하는 단계다.

---

### 생성 파일

| 파일 | 설명 |
|---|---|
| `chapters/chapter11/03_atflee_mcp_client_test.py` | FastMCP Client로 MCP 서버 tool call 테스트 |

---

### 호출한 도구 (4개)

| 도구 | 입력 | 결과 요약 |
|---|---|---|
| `get_atflee_status` | `{}` | status=ready, wiki_doc_count=8 |
| `list_atflee_tools` | `{}` | 4개 도구 목록 반환 |
| `search_atflee_wiki` | `{"question": "체중계 앱 연결..."}` | atflee_app_guide.md TOP 1 |
| `classify_atflee_voc` | `{"customer_message": "제품 불량 교환"}` | 교환/반품 / 높음 / CS/물류팀 |

---

### 실행 명령어

```bash
# 클라이언트 실행 (서버 자동 실행 후 연결)
python chapters/chapter11/03_atflee_mcp_client_test.py

# 문법 검증만
python -m py_compile chapters/chapter11/03_atflee_mcp_client_test.py
```

사전 조건:
- `pip install fastmcp` 완료
- `02_atflee_mcp_server_basic.py` 존재
- `data/wiki/*.md` 파일 존재
- 별도 터미널에서 서버를 수동 실행할 필요 없음 (Client(SERVER_PATH)가 stdio로 자동 실행)

---

### 연결 방식

```python
from fastmcp import Client

async with Client(SERVER_PATH) as client:
    tools = await client.list_tools()
    result = await client.call_tool("search_atflee_wiki", {"question": "...", "top_k": 3})
```

`Client(SERVER_PATH)`에 파일 경로를 넘기면 FastMCP가 해당 스크립트를 stdio 서브프로세스로
자동 실행하고 JSON-RPC over stdio로 통신한다.

---

### 11-2 MCP Server vs 11-3 MCP Client 차이

| 항목 | 11-2 MCP Server | 11-3 MCP Client |
|---|---|---|
| 역할 | 도구를 등록하고 연결 대기 | 서버 도구를 원격 호출하는 소비자 |
| 실행 흐름 | `mcp.run()`으로 서버 대기 | `async with Client(path):`로 서버 자동 실행 후 연결 |
| 코드 위치 | `@mcp.tool` 데코레이터 서버 파일 | `await client.call_tool(...)` 클라이언트 파일 |

---

### 다음 단계: 11-4

Claude `tool_use` API와 MCP 서버를 연결하는 에이전트:
- Claude에게 MCP 서버 도구 스키마를 `tools=[]`로 전달
- Claude가 `tool_use` 블록으로 도구를 선택
- Python이 MCP 클라이언트로 해당 도구를 실행하고 `tool_result`로 반환
- Claude가 최종 답변 생성
