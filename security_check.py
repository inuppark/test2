import os

SCAN_EXTENSIONS = {".py", ".txt", ".md", ".toml", ".yaml", ".yml", ".json"}

SKIP_DIRS = {".venv", "venv", "__pycache__", ".git"}
SKIP_FILES = {".env", ".env.example"}

SENSITIVE_KEYWORDS = [
    "sk-ant-",
    "ANTHROPIC_API_KEY=",
    "api_key=",
    "secret_key",
    "password",
]


def mask_line(line):
    if len(line) <= 20:
        return "***"
    return line[:10] + "..." + line[-10:]


def scan_file(filepath):
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, start=1):
                for keyword in SENSITIVE_KEYWORDS:
                    if keyword.lower() in line.lower():
                        findings.append((line_num, keyword, mask_line(line.strip())))
    except Exception:
        pass
    return findings


def run_check(root="."):
    print("=" * 50)
    print("보안 점검 시작")
    print("=" * 50)

    found_any = False

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            if filename in SKIP_FILES:
                continue

            _, ext = os.path.splitext(filename)
            if ext not in SCAN_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, filename)
            findings = scan_file(filepath)

            if findings:
                found_any = True
                print(f"\n[주의] {filepath}")
                for line_num, keyword, masked in findings:
                    print(f"  줄 {line_num}: '{keyword}' 감지 → {masked}")

    print("\n" + "=" * 50)
    if found_any:
        print("민감 정보가 의심되는 항목이 발견되었습니다. 위 파일을 확인하세요.")
    else:
        print("민감 정보 없음. 안전합니다.")
    print("=" * 50)


if __name__ == "__main__":
    run_check()
