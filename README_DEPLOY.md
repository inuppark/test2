# 앳플리 AX Console v0 배포 준비

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run apps/ax_console_v0.py
```

## 로컬 환경변수

`.env` 파일에 아래 값을 설정한다.

```
ANTHROPIC_API_KEY=your_key_here
```

실제 키는 GitHub에 올리지 않는다.

## Streamlit Cloud 배포 시 Secrets 설정

Streamlit Cloud 앱 설정의 **Secrets**에 아래 값을 등록한다.

```
ANTHROPIC_API_KEY="실제 Anthropic API Key"
```

## 보안 체크

배포 전 아래 명령어를 실행한다.

```bash
python security_check.py
```

## GitHub에 올리면 안 되는 파일

- `.env`
- `.venv/`
- `venv/`
- `__pycache__/`
- `logs/`
- `.streamlit/secrets.toml`
