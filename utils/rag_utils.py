"""
utils/rag_utils.py
앳플리 봇 RAG 검색 공통 모듈.

apps/atflee_bot.py, apps/ax_console_v0.py,
chapters/chapter07/02_rag_answer_with_claude.py 에서 공통으로 사용한다.
"""

import os

# ==============================
# 상수
# ==============================

WIKI_DIR = "data/wiki"
TOP_K = 3

# ==============================
# 불용어 목록
# 점수 계산에서 제외할 일반 단어들.
# 이런 단어들은 여러 문서에 고루 등장해 점수를 왜곡한다.
# ==============================

STOPWORDS = {
    "어떻게", "해야", "하나요", "해요", "할때", "할", "때",
    "무엇", "뭐", "좀", "그리고", "근데", "관련", "정보",
    "확인", "문의", "안내", "알려줘", "해주세요", "됩니다",
    "있는", "없는", "하면", "해서", "이랑", "랑", "가", "이",
    "은", "는", "을", "를", "에", "의", "도"
}

# 한국어 조사/어미: 토큰 끝에서 제거해 어근을 추출한다.
# 길이가 긴 것을 먼저 시도해야 부분 제거 오류를 막을 수 있다.
KOREAN_SUFFIXES = [
    "이랑", "에서", "에게", "부터", "까지", "으로", "한테", "에는",
    "이고", "이야", "이다", "됐는지", "됐어", "됐나",
    "랑", "은", "는", "이", "가", "을", "를", "에", "의", "도",
    "과", "와", "로",
]

# ==============================
# 도메인 동의어 사전
# 질문에서 특정 키워드를 발견하면, 연관 키워드를 함께 검색에 포함시킨다.
# 예: "AS" → "수리", "고장", "접수" 등도 함께 검색
# ==============================

DOMAIN_SYNONYMS = {
    "as":       ["as", "에이에스", "수리", "고장", "접수", "센터", "교환", "불량"],
    "에이에스": ["as", "에이에스", "수리", "고장", "접수", "센터", "교환", "불량"],
    "수리":     ["as", "에이에스", "수리", "고장", "접수", "센터"],
    "고장":     ["as", "에이에스", "수리", "고장", "불량", "교환"],
    "배송":     ["배송", "택배", "송장", "출고", "지연", "도착"],
    "택배":     ["배송", "택배", "송장", "출고", "지연", "도착"],
    "환불":     ["환불", "반품", "취소", "카드", "승인취소", "결제취소"],
    "반품":     ["환불", "반품", "교환", "취소", "회수"],
    "교환":     ["교환", "반품", "불량", "as", "고장"],
    "앱":       ["앱", "어플", "애플리케이션", "연동", "연결", "블루투스"],
    "어플":     ["앱", "어플", "애플리케이션", "연동", "연결", "블루투스"],
    "연결":     ["연결", "연동", "블루투스", "권한", "페어링"],
    "연동":     ["연결", "연동", "블루투스", "권한", "페어링"],
    "체중계":   ["체중계", "스마트체중계", "인바디", "체성분", "측정"],
    "t9":       ["t9", "체중계", "스마트체중계", "듀얼주파수"],
    "문의":     ["문의", "고객센터", "1:1", "전화", "대표번호", "이메일"],
}


# ==============================
# 1. 문서 로드
# ==============================

def load_wiki_documents(wiki_dir=WIKI_DIR):
    """
    wiki_dir 안의 .md 파일을 모두 읽어
    [{"file_name": 파일명, "content": 내용}, ...] 형태로 반환한다.
    파일명은 정렬해서 읽으므로 순서가 일정하다.
    폴더가 없으면 빈 리스트를 반환한다.
    """
    documents = []

    if not os.path.exists(wiki_dir):
        return documents

    for file_name in sorted(os.listdir(wiki_dir)):
        if not file_name.endswith(".md"):
            continue

        file_path = os.path.join(wiki_dir, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(
                    {
                        "file_name": file_name,
                        "content": file.read()
                    }
                )
        except Exception:
            pass

    return documents


# ==============================
# 2. 토큰화
# ==============================

def _strip_korean_suffix(token):
    """
    토큰 끝의 한국어 조사/어미를 제거해 어근에 가까운 형태로 반환한다.
    예: "배송은" → "배송", "앱이랑" → "앱", "체중계가" → "체중계"
    결과가 너무 짧아지면(1글자 이하) 원본을 반환한다.
    """
    for suffix in KOREAN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: len(token) - len(suffix)]
    return token


def tokenize(text):
    """
    텍스트를 단어 단위 토큰 리스트로 변환한다.

    처리 순서:
    1. 소문자 변환
    2. 한글/영어/숫자/공백 이외 문자를 공백으로 변환
    3. 공백 기준 분리
    4. 조사/어미 제거 (배송은→배송, 앱이랑→앱)
    5. 2글자 미만 제거
    6. 불용어 제거
    """
    text = text.lower()

    cleaned_chars = []
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")

    tokens = "".join(cleaned_chars).split()

    # 조사/어미를 제거해 어근에 가깝게 만든다.
    tokens = [_strip_korean_suffix(t) for t in tokens]

    tokens = [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]

    return tokens


# ==============================
# 3. 동의어 확장
# ==============================

def expand_tokens(tokens):
    """
    DOMAIN_SYNONYMS를 사용해 토큰을 확장한다.
    원본 토큰을 유지하면서 동의어를 추가하고 중복을 제거한다.

    예: ["as"] → {"as", "에이에스", "수리", "고장", "접수", "센터", "교환", "불량"}
    """
    expanded = set(tokens)

    for token in tokens:
        if token in DOMAIN_SYNONYMS:
            expanded.update(DOMAIN_SYNONYMS[token])

    return expanded


# ==============================
# 4. 문서 점수 계산
# ==============================

def score_document(question, document_content):
    """
    질문과 문서 사이의 관련도 점수를 계산한다.

    점수 구성:
    - 기본 점수: 질문 확장 토큰과 문서 토큰의 교집합 크기
    - 헤더 가중치: 문서의 제목/헤더(# 또는 ##로 시작하는 줄)에
      질문 토큰이 포함되면 겹친 개수 * 2점을 추가로 준다.
      헤더에 있다는 것은 해당 문서의 핵심 주제일 가능성이 높기 때문이다.
    """
    # 질문 토큰화 및 동의어 확장
    question_tokens = tokenize(question)
    expanded_question_tokens = expand_tokens(question_tokens)

    # 문서 전체 토큰화 (set으로 중복 제거)
    document_token_set = set(tokenize(document_content))

    # 기본 점수: 교집합 크기
    base_score = len(expanded_question_tokens.intersection(document_token_set))

    # 헤더 가중치: #, ## 로 시작하는 줄만 추출
    header_lines = [
        line.lstrip("#").strip()
        for line in document_content.splitlines()
        if line.startswith("#")
    ]
    header_text = " ".join(header_lines)
    header_token_set = set(tokenize(header_text))

    header_bonus = len(expanded_question_tokens.intersection(header_token_set)) * 2

    return base_score + header_bonus


# ==============================
# 5. 관련 스니펫 추출
# ==============================

def extract_relevant_snippet(question, document_content, max_chars=300):
    """
    질문과 가장 관련 있어 보이는 문서 일부를 추출한다.

    처리 순서:
    1. 질문을 토큰화하고 동의어 확장한다.
    2. 문서를 줄 단위로 나눈다 (빈 줄 제외).
    3. 각 줄과 확장 토큰의 교집합 크기로 점수를 계산한다.
    4. 점수가 가장 높은 줄 앞뒤 1줄을 합쳐 스니펫을 만든다.
    5. max_chars를 초과하면 잘라내고 "..."를 붙인다.
    6. 모든 점수가 0이면 문서 앞부분 max_chars를 반환한다.
    """
    expanded_tokens = expand_tokens(tokenize(question))

    # 빈 줄은 제외하고 줄 단위로 분리
    lines = [line for line in document_content.splitlines() if line.strip()]

    if not lines:
        snippet = document_content[:max_chars]
        return snippet + "..." if len(document_content) > max_chars else snippet

    # 각 줄에 점수 부여
    line_scores = [
        len(expanded_tokens.intersection(set(tokenize(line))))
        for line in lines
    ]

    best_score = max(line_scores)

    # 모든 점수가 0이면 문서 앞부분 반환
    if best_score == 0:
        snippet = document_content[:max_chars]
        return snippet + "..." if len(document_content) > max_chars else snippet

    best_idx = line_scores.index(best_score)

    # 최고 점수 줄 앞뒤 1줄씩 포함
    start = max(0, best_idx - 1)
    end = min(len(lines), best_idx + 2)
    snippet = "\n".join(lines[start:end])

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."

    return snippet


# ==============================
# 6. 위키 검색
# ==============================

def search_wiki(question, top_k=TOP_K, wiki_dir=WIKI_DIR):
    """
    질문과 관련도가 높은 문서를 top_k개 반환한다.

    정렬 기준:
    - 점수 내림차순
    - 점수가 같으면 파일명 오름차순 (결과 일관성 확보)

    점수가 0인 문서는 제외한다.
    단, 모든 문서 점수가 0이면 파일명 기준 상위 top_k를 fallback으로 반환한다.

    반환 형태:
    [{"file_name": ..., "content": ..., "score": ..., "snippet": ...}, ...]
    """
    documents = load_wiki_documents(wiki_dir)

    scored = []
    for doc in documents:
        score = score_document(question, doc["content"])
        snippet = extract_relevant_snippet(question, doc["content"])
        scored.append(
            {
                "file_name": doc["file_name"],
                "content": doc["content"],
                "score": score,
                "snippet": snippet,
            }
        )

    # 점수 내림차순, 동점이면 파일명 오름차순
    scored.sort(key=lambda x: (-x["score"], x["file_name"]))

    # 점수가 0인 문서는 제외
    non_zero = [d for d in scored if d["score"] > 0]

    if non_zero:
        return non_zero[:top_k]

    # fallback: 모든 점수가 0이면 파일명 기준 상위 top_k 반환
    return scored[:top_k]


# ==============================
# 7. RAG Context 조합
# ==============================

def build_rag_context(search_results):
    """
    검색된 문서들을 하나의 문자열로 합친다.
    [관련 스니펫]을 먼저 제시해 Claude가 핵심 근거를 빠르게 파악하도록 한다.
    그 다음 [문서 전체]를 붙여 전체 맥락도 활용할 수 있게 한다.
    """
    context_parts = []

    for result in search_results:
        snippet = result.get("snippet", "")
        if snippet:
            block = (
                f"[문서명: {result['file_name']}]\n"
                f"[관련 스니펫]\n{snippet}\n\n"
                f"[문서 전체]\n{result['content']}"
            )
        else:
            block = f"[문서명: {result['file_name']}]\n{result['content']}"

        context_parts.append(block)

    return "\n\n---\n\n".join(context_parts)


# ==============================
# 8. RAG 답변 품질 평가
# ==============================

# 실제 주문/배송/AS 상태를 확정하는 위험 표현
_ORDER_STATUS_RISK = [
    "접수되었습니다",
    "접수 완료되었습니다",
    "배송 중입니다",
    "출고되었습니다",
    "환불 완료되었습니다",
    "교환 처리되었습니다",
    "as가 접수되었습니다",
]

# 가격/재고/이벤트를 단정하는 위험 표현
_PRICE_STOCK_RISK = [
    "현재 가격은",
    "재고가 있습니다",
    "품절입니다",
    "이벤트 진행 중입니다",
    "프로모션 중입니다",
]

# 개인정보 입력을 직접 요구하는 위험 표현
# "입력하지 마세요" 같은 안전 안내와 구분하기 위해 "해주세요" 형태만 체크한다.
_PRIVACY_RISK = [
    "주문번호를 입력해주세요",
    "연락처를 입력해주세요",
    "주소를 입력해주세요",
    "개인정보를 입력해주세요",
]


def evaluate_rag_answer(answer: str, source_files: list) -> dict:
    """
    RAG 답변의 기본 품질을 점검한다.

    점검 항목:
    1. 답변에 "참고 문서" 섹션이 있는지
    2. source_files 중 하나 이상이 답변에 포함되어 있는지
    3. 주문/배송/AS 상태를 확정하는 위험 표현이 있는지
    4. 가격/재고/이벤트를 단정하는 위험 표현이 있는지
    5. 개인정보 입력을 직접 요구하는 위험 표현이 있는지
    6. 답변 길이가 100자 이상인지

    반환 형태:
    {
        "score": 0~100,
        "status": "좋음" | "주의" | "위험",
        "checks": [{"name": str, "passed": bool, "message": str}, ...]
    }
    """
    answer_lower = answer.lower()
    checks = []
    score = 100

    # 1. 참고 문서 섹션 여부
    has_ref = "참고 문서" in answer
    checks.append({
        "name": "참고 문서 표시",
        "passed": has_ref,
        "message": "참고 문서 섹션이 포함되어 있습니다." if has_ref
                   else "참고 문서 섹션이 없습니다. (-20점)",
    })
    if not has_ref:
        score -= 20

    # 2. 출처 파일명 포함 여부
    has_source = any(f.lower() in answer_lower for f in source_files)
    checks.append({
        "name": "출처 파일명 포함",
        "passed": has_source,
        "message": "출처 파일명이 답변에 포함되어 있습니다." if has_source
                   else "출처 파일명이 답변에 없습니다. (-20점)",
    })
    if not has_source:
        score -= 20

    # 3. 주문/배송/AS 상태 단정 위험 표현
    order_hits = [p for p in _ORDER_STATUS_RISK if p in answer_lower]
    order_safe = len(order_hits) == 0
    checks.append({
        "name": "주문/배송/AS 상태 단정",
        "passed": order_safe,
        "message": "주문·배송·AS 상태를 단정하는 표현이 없습니다." if order_safe
                   else f"위험 표현 발견: {', '.join(order_hits)} (-15점)",
    })
    if not order_safe:
        score -= 15

    # 4. 가격/재고 단정 위험 표현
    price_hits = [p for p in _PRICE_STOCK_RISK if p in answer_lower]
    price_safe = len(price_hits) == 0
    checks.append({
        "name": "가격/재고 단정",
        "passed": price_safe,
        "message": "가격·재고를 단정하는 표현이 없습니다." if price_safe
                   else f"위험 표현 발견: {', '.join(price_hits)} (-15점)",
    })
    if not price_safe:
        score -= 15

    # 5. 개인정보 입력 직접 요구
    privacy_hits = [p for p in _PRIVACY_RISK if p in answer_lower]
    privacy_safe = len(privacy_hits) == 0
    checks.append({
        "name": "개인정보 유도",
        "passed": privacy_safe,
        "message": "개인정보 입력을 요구하는 표현이 없습니다." if privacy_safe
                   else f"위험 표현 발견: {', '.join(privacy_hits)} (-15점)",
    })
    if not privacy_safe:
        score -= 15

    # 6. 답변 충분성 (100자 이상)
    long_enough = len(answer) >= 100
    checks.append({
        "name": "답변 충분성",
        "passed": long_enough,
        "message": f"답변 길이가 충분합니다. ({len(answer)}자)" if long_enough
                   else f"답변이 너무 짧습니다. ({len(answer)}자, -10점)",
    })
    if not long_enough:
        score -= 10

    score = max(0, score)

    if score >= 80:
        status = "좋음"
    elif score >= 60:
        status = "주의"
    else:
        status = "위험"

    return {"score": score, "status": status, "checks": checks}


# ==============================
# 9. 파일명 목록 반환
# ==============================

def get_source_file_names(search_results):
    """검색된 문서의 파일명 리스트를 반환한다."""
    return [result["file_name"] for result in search_results]


# ==============================
# 10. 콘솔 출력용 포맷
# ==============================

def format_search_results_for_display(search_results):
    """
    콘솔/Streamlit expander 표시용 문자열 리스트를 반환한다.
    파일명, 점수, 스니펫을 포함한다.

    반환 예:
    [
        "- atflee_app_guide.md / 점수: 7",
        "  관련 내용: 블루투스 연결이 안 될 때는...",
        ...
    ]
    """
    lines = []
    for result in search_results:
        lines.append(f"- {result['file_name']} / 점수: {result['score']}")
        snippet = result.get("snippet", "")
        if snippet:
            # 줄바꿈을 공백으로 바꿔 한 줄로 출력한다.
            one_line = snippet.replace("\n", " ")
            lines.append(f"  관련 내용: {one_line}")
    return lines
