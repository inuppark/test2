# Claude Desktop 앳플리 MCP 연결 결과

## 요약

Claude Desktop에서 `atflee-mcp-server-v0` 연결이 성공했다.
`claude_desktop_config.json`의 `mcpServers`에 서버를 등록한 뒤
Claude Desktop 채팅창에서 앳플리 MCP 도구를 직접 호출할 수 있다.

---

## 연결된 도구

| 도구 | 설명 |
|---|---|
| `get_atflee_status` | 서버 상태 및 위키 문서 수 확인 |
| `list_atflee_tools` | 사용 가능한 도구 목록 반환 |
| `search_atflee_wiki` | `data/wiki` 문서 키워드 검색 |
| `classify_atflee_voc` | 고객 문의 의도/심각도/담당팀 분류 |

---

## 테스트 질문과 결과

### 1. 도구 목록 확인

질문:
> 앳플리에서 사용할 수 있는 MCP 도구 목록 알려줘.

결과:
- `list_atflee_tools` 호출 성공
- 4개 도구 목록 확인

---

### 2. 서버 상태 확인

질문:
> 앳플리 MCP 서버 상태 확인해줘.

결과:
- `get_atflee_status` 호출 성공
- `status=ready`
- `wiki_doc_count=8` 확인

---

### 3. 위키 검색

질문:
> 체중계가 앱이랑 연결이 안 돼요. 관련 문서 찾아줘.

결과:
- `search_atflee_wiki` 호출 성공
- `atflee_app_guide.md` 기반 앱 연결 체크리스트 답변 생성

---

### 4. VOC 분류

질문:
> 제품이 불량 같고 교환하고 싶어요. VOC 분류해줘.

결과:
- `classify_atflee_voc` 호출 성공
- `intent=교환/반품`
- `severity=높음`
- `owner_team=CS/물류팀`
- `needs_human_review=True`

---

## 의미

이번 단계에서 앳플리 도구는 더 이상 Python 콘솔 안에서만 실행되는 함수가 아니라,
Claude Desktop에서 직접 호출 가능한 MCP 도구가 되었다.

11-1(ReAct 콘솔) → 11-2(MCP 서버) → 11-3(MCP 클라이언트) → 11-4(Claude tool_use)
→ 11-5(Desktop config) → **11-6(Desktop 연결 검증 완료)**

---

## 현재 한계

- MCP 도구는 기본 위키 검색과 간단 VOC 분류 중심이다.
- AX Console의 하이브리드 RAG, CS 답변 초안, 답변 안전성 평가 기능은
  아직 MCP 도구로 승격되지 않았다.
- Claude Desktop에서 AX Console UI를 직접 사용하는 것은 아니며,
  MCP 서버에 노출된 도구만 호출한다.

---

## 다음 확장 후보

아래 기능들을 MCP tool로 승격하면 Claude Desktop에서도 사용 가능해진다.

| 후보 도구 | 현재 위치 |
|---|---|
| `generate_cs_reply` | `apps/ax_console_v0.py` CS 답변 초안 탭 |
| `hybrid_rag_answer` | `utils/hybrid_rag_utils.py` |
| `evaluate_answer_quality` | `chapters/chapter11/01_atflee_react_agent_basic.py` |
| `analyze_voc` | `apps/ax_console_v0.py` VOC 분석 탭 |
| `search_hybrid_atflee_knowledge` | `utils/hybrid_rag_utils.py` |
