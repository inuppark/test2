"""
Chapter 11 선택 실습 11-5: Claude Desktop MCP 설정 예시 생성

앳플리 MCP 서버 v0(02_atflee_mcp_server_basic.py)를
Claude Desktop에서 연결하기 위한 mcpServers 설정 예시 JSON을 생성한다.

이 스크립트는 실제 Claude Desktop 설정 파일을 수정하지 않는다.
생성된 JSON을 확인한 뒤 Claude Desktop config에 수동으로 반영한다.

실행:
  python chapters/chapter11/05_atflee_claude_desktop_config_example.py
"""

import os
import sys
import json
from pathlib import Path

CURRENT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
SERVER_PATH  = PROJECT_ROOT / "chapters" / "chapter11" / "02_atflee_mcp_server_basic.py"
PYTHON_EXE   = Path(sys.executable).resolve()

CONFIG_OUTPUT = CURRENT_DIR / "claude_desktop_config_atflee_example.json"
README_OUTPUT = CURRENT_DIR / "README_claude_desktop_mcp_setup.md"


# ==============================
# 콘솔 출력 helper
# ==============================
def _p(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


# ==============================
# 경로 검증
# ==============================
def check_paths() -> bool:
    _p("=" * 60)
    _p("[Chapter 11-5] Claude Desktop MCP 설정 예시 생성")
    _p("=" * 60)

    _p("\n[Python]")
    _p(str(PYTHON_EXE))

    _p("\n[Project Root]")
    _p(str(PROJECT_ROOT))

    _p("\n[MCP Server]")
    _p(str(SERVER_PATH))
    server_exists = SERVER_PATH.exists()
    _p(f"exists: {server_exists}")

    if not server_exists:
        _p("\n[오류] MCP 서버 파일이 존재하지 않습니다.")
        _p("02_atflee_mcp_server_basic.py 파일을 먼저 확인하세요.")
        return False

    return True


# ==============================
# Claude Desktop config 예시 생성
# ==============================
def build_config() -> dict:
    """
    Claude Desktop mcpServers 설정 예시를 dict로 반환한다.
    Windows 경로는 json.dump가 역슬래시를 자동 이스케이프 처리한다.
    API Key 환경변수는 이번 서버에 불필요하므로 포함하지 않는다.
    """
    return {
        "mcpServers": {
            "atflee-mcp-server-v0": {
                "command": str(PYTHON_EXE),
                "args":    [str(SERVER_PATH)],
            }
        }
    }


def save_config(config: dict) -> None:
    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    _p(f"\n[Config Example Saved]")
    _p(str(CONFIG_OUTPUT))

    _p("\n[생성된 JSON 내용]")
    _p(json.dumps(config, ensure_ascii=False, indent=2))


# ==============================
# README 생성
# ==============================
def save_readme() -> None:
    content = f"""# 앳플리 MCP 서버 Claude Desktop 연결 가이드

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
| Python 실행 파일 | `{PYTHON_EXE}` |
| MCP 서버 파일 | `{SERVER_PATH}` |
| 설정 예시 파일 | `{CONFIG_OUTPUT}` |

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
%APPDATA%\\Claude\\claude_desktop_config.json
```

PowerShell에서 열기:

```powershell
notepad "$env:APPDATA\\Claude\\claude_desktop_config.json"
```

파일이 없으면 새로 만든다.

### 2단계 — mcpServers 내용 병합

`claude_desktop_config_atflee_example.json`의 내용을 복사해
기존 `claude_desktop_config.json`의 `mcpServers` 항목에 병합한다.

기존 config가 없는 경우 아래 내용 전체를 붙여넣는다:

```json
{{
  "mcpServers": {{
    "atflee-mcp-server-v0": {{
      "command": "{str(PYTHON_EXE).replace(chr(92), chr(92)*2)}",
      "args": [
        "{str(SERVER_PATH).replace(chr(92), chr(92)*2)}"
      ]
    }}
  }}
}}
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
| 경로 오류 | config JSON에서 역슬래시가 이스케이프(`\\\\`)되어 있는지 확인 |
"""

    with open(README_OUTPUT, "w", encoding="utf-8") as f:
        f.write(content)

    _p(f"\n[README Saved]")
    _p(str(README_OUTPUT))


# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    ok = check_paths()
    if not ok:
        sys.exit(1)

    config = build_config()
    save_config(config)
    save_readme()

    _p("\n[주의]")
    _p("이 스크립트는 실제 Claude Desktop 설정 파일을 수정하지 않습니다.")
    _p("생성된 JSON 내용을 확인한 뒤 수동으로 Claude Desktop 설정에 복사하세요.")
    _p("")
    _p("Claude Desktop config 파일 위치 (Windows):")
    _p("  %APPDATA%\\Claude\\claude_desktop_config.json")
    _p("")
    _p("[실습 완료]")
    _p("다음 단계: 11-6 - 실제 Claude Desktop 연결 테스트 또는 AX Console MCP 탭 추가")
