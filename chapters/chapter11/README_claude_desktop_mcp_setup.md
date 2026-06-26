# 앳플리 MCP 서버 Claude Desktop 연결 가이드

## 목적

앳플리 MCP 서버 v0를 Claude Desktop에서 사용할 수 있게 연결한다.

---

## 현재 서버

- 서버명: `atflee-mcp-server-v0`
- 서버 파일: `chapters/chapter11/02_atflee_mcp_server_basic.py`
- 제공 도구:
  - `get_atflee_status` — 서버 상태 확인
  - `list_atflee_tools` — 도구 목록 확인
  - `search_atflee_wiki` — 앳플리 위키 문서 검색
  - `classify_atflee_voc` — 고객 문의 VOC 분류

---

## 환경 정보

| 항목 | 경로 |
|---|---|
| Python 실행 파일 | `E:\ipark\.venv\Scripts\python.exe` |
| MCP 서버 파일 | `E:\ipark\chapters\chapter11\02_atflee_mcp_server_basic.py` |
| 설정 예시 파일 | `E:\ipark\chapters\chapter11\claude_desktop_config_atflee_example.json` |

---

## 설정 파일 예시 생성

```bash
python chapters/chapter11/05_atflee_claude_desktop_config_example.py
```

생성 파일:
- `chapters/chapter11/claude_desktop_config_atflee_example.json`

---

## Claude Desktop에 반영하는 방법

### 1단계 — Claude Desktop config 파일 위치 확인

Windows 기준:

```
%APPDATA%\Claude\claude_desktop_config.json
```

PowerShell에서 열기:

```powershell
notepad "$env:APPDATA\Claude\claude_desktop_config.json"
```

파일이 없으면 새로 만든다.

### 2단계 — mcpServers 내용 병합

`claude_desktop_config_atflee_example.json`의 내용을 복사해
기존 `claude_desktop_config.json`의 `mcpServers` 항목에 병합한다.

기존 config가 없는 경우 아래 내용 전체를 붙여넣는다:

```json
{
  "mcpServers": {
    "atflee-mcp-server-v0": {
      "command": "E:\\ipark\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\ipark\\chapters\\chapter11\\02_atflee_mcp_server_basic.py"
      ]
    }
  }
}
```

기존 `mcpServers`가 이미 있는 경우 `atflee-mcp-server-v0` 항목만 추가한다.

### 3단계 — Claude Desktop 재시작

Claude Desktop을 **완전히 종료** 후 다시 실행한다.
(트레이 아이콘에서 종료해야 완전히 닫힌다.)

### 4단계 — 연결 확인

새 대화를 열고 채팅창 하단의 도구 아이콘에서
`atflee-mcp-server-v0` 서버와 4개 도구가 보이는지 확인한다.

---

## 테스트 질문

Claude Desktop에서 아래 질문을 직접 입력한다.

```
앳플리 MCP 서버 상태 확인해줘.
```

```
앳플리에서 사용할 수 있는 MCP 도구 목록 알려줘.
```

```
체중계가 앱이랑 연결이 안 돼요. 관련 문서 찾아줘.
```

```
제품이 불량 같고 교환하고 싶어요. VOC 분류해줘.
```

---

## 주의사항

- 실제 주문 상태, 배송 상태, AS 접수 상태를 단정하지 않는다.
- 주문번호, 연락처, 주소 등 개인정보는 Claude Desktop 채팅창에 입력하지 않는다.
- 경로는 반드시 절대 경로를 사용한다.
- Python 가상환경 경로(`.venv`)가 바뀌면 config도 다시 생성해야 한다.
- 서버 실행이 안 되면 먼저 아래 명령어로 단독 실행을 확인한다.

```bash
python chapters/chapter11/02_atflee_mcp_server_basic.py
```

---

## 문제 해결

| 증상 | 확인 사항 |
|---|---|
| 도구 목록에 서버가 없음 | Claude Desktop 완전 종료 후 재시작 |
| 서버가 연결 오류 | Python 경로, 서버 파일 경로 절대경로 확인 |
| 도구 호출 오류 | `data/wiki/*.md` 파일 존재 여부 확인 |
| 경로 오류 | config JSON에서 역슬래시가 이스케이프(`\\`)되어 있는지 확인 |
