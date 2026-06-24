import os
import re
from dotenv import load_dotenv
from anthropic import Anthropic

# .env 파일에서 환경변수를 불러온다.
load_dotenv()

# ==============================
# 설정값
# ==============================

WIKI_DIR = "data/wiki"
TOP_K = 3
model_name = "claude-sonnet-4-5"

# ==============================
# API Key 확인
# ==============================

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("[오류] ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
    print(".env 파일에 ANTHROPIC_API_KEY=sk-... 형식으로 등록해주세요.")
    exit(1)

client = Anthropic(api_key=api_key)

# ==============================
# System Prompt (Chapter 4.3 원칙 반영)
# ==============================

system_prompt = """
# Role
너는 앳플리 위키 기반 RAG 답변 봇이다.

# Goal
사용자 질문에 대해 검색된 앳플리 위키 문서를 근거로 정확하고 친절하게 답변한다.

# Context
너에게 제공되는 <rag_context>는 data/wiki 문서 중 사용자 질문과 관련도가 높은 문서만 검색해서 가져온 것이다.
너는 이 문서 내용을 우선 근거로 사용한다.

# Rules
- <rag_context>에 있는 정보만 확정적으로 말한다.
- <rag_context>에 없는 내용은 추측하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
- 정책, 보증, 교환/환불, AS 조건은 확실하지 않으면 "정확한 확인이 필요합니다"라고 말한다.
- 개인정보, 주문번호, 연락처 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
- 고객이 바로 해볼 수 있는 다음 행동을 안내한다.
- 답변은 초보자도 이해할 수 있게 쉽게 작성한다.

# Process
1. 사용자 질문의 의도를 파악한다.
2. <rag_context>에서 관련 근거를 찾는다.
3. 확실한 정보와 확인이 필요한 정보를 구분한다.
4. 사용자가 바로 할 수 있는 행동을 안내한다.
5. 마지막에 참고한 문서명을 표시한다.

# Output Format
아래 형식으로 답변한다.

1. 간단한 답변
2. 근거가 되는 앳플리 위키 정보
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 문서
"""


# ==============================
# 1. 문서 로드 함수
# ==============================

def load_wiki_documents():
    """
    data/wiki 폴더 안의 .md 파일을 모두 읽어
    [{"file_name": 파일명, "content": 내용}, ...] 형태로 반환한다.
    """
    if not os.path.exists(WIKI_DIR):
        print(f"[오류] '{WIKI_DIR}' 폴더를 찾을 수 없습니다.")
        print("프로젝트 루트 디렉토리에서 실행하고 있는지 확인해주세요.")
        return []

    documents = []

    for file_name in sorted(os.listdir(WIKI_DIR)):
        if not file_name.endswith(".md"):
            continue

        file_path = os.path.join(WIKI_DIR, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            documents.append({
                "file_name": file_name,
                "content": content
            })

        except Exception as e:
            print(f"[경고] '{file_name}' 파일을 읽는 중 오류가 발생했습니다: {e}")

    return documents


# ==============================
# 2. 텍스트 토큰화 함수
# ==============================

def tokenize(text):
    """
    텍스트를 단어 단위 토큰 리스트로 변환한다.
    한글, 영어, 숫자가 섞여도 기본적으로 처리할 수 있게 한다.
    """
    # 소문자로 변환 (영어 대소문자 통일)
    text = text.lower()

    # 한글/영어/숫자/공백 이외의 문자를 공백으로 변환
    text = re.sub(r"[^\w\s가-힣]", " ", text)

    # 공백 기준으로 단어를 나눈다.
    tokens = text.split()

    # 1글자 이하 토큰은 노이즈가 많으므로 제거한다.
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


# ==============================
# 3. 문서 관련도 점수 계산 함수
# ==============================

def score_document(question, document_content):
    """
    질문과 문서 사이의 키워드 겹침 점수를 계산한다.
    set을 사용하므로 같은 단어가 문서에 여러 번 나와도 1번으로 계산한다.
    """
    question_tokens = tokenize(question)
    document_token_set = set(tokenize(document_content))

    score = 0
    for token in question_tokens:
        if token in document_token_set:
            score += 1

    return score


# ==============================
# 4. 위키 검색 함수
# ==============================

def search_wiki(question, top_k=TOP_K):
    """
    질문과 관련 있는 문서를 키워드 기반으로 검색해 반환한다.
    점수가 같으면 파일명 알파벳 순서로 정렬해 결과가 일정하게 나오게 한다.
    """
    documents = load_wiki_documents()

    if not documents:
        print("[안내] data/wiki 폴더에 .md 문서가 없습니다.")
        return []

    scored_documents = []

    for doc in documents:
        score = score_document(question, doc["content"])
        scored_documents.append({
            "file_name": doc["file_name"],
            "content": doc["content"],
            "score": score
        })

    # 점수 내림차순, 점수가 같으면 파일명 오름차순으로 정렬
    scored_documents.sort(key=lambda x: (-x["score"], x["file_name"]))

    return scored_documents[:top_k]


# ==============================
# 5. RAG Context 조합 함수
# ==============================

def build_rag_context(search_results):
    """
    검색된 문서들을 하나의 문자열로 합친다.
    각 문서 앞에 출처 파일명을 붙여 Claude가 출처를 파악할 수 있게 한다.
    """
    context_parts = []

    for doc in search_results:
        # 각 문서 앞에 파일명 출처 표시를 붙인다.
        part = f"[문서명: {doc['file_name']}]\n{doc['content']}"
        context_parts.append(part)

    # 문서 사이는 구분선으로 나눈다.
    return "\n\n---\n\n".join(context_parts)


# ==============================
# 6. Claude RAG 답변 함수
# ==============================

def ask_claude_with_rag(question, rag_context, source_files):
    """
    검색된 문서를 Context로 Claude에게 전달하고 답변을 받는다.
    XML 태그로 rag_context와 user_question을 구분한다.
    source_files는 참고 문서명 목록이다.
    """
    # XML 태그로 Context와 질문을 구분해 Claude가 명확히 인식하게 한다.
    user_prompt = f"""
<rag_context>
{rag_context}
</rag_context>

<user_question>
{question}
</user_question>

참고: 위 답변의 5번 항목 '참고 문서'에는 아래 파일명을 반드시 포함해줘.
{", ".join(source_files)}
"""

    response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.content[0].text


# ==============================
# 검색 결과 출력 함수
# ==============================

def print_search_results(results):
    """
    검색된 문서 목록과 점수를 출력한다.
    """
    print("\n[검색된 문서]")
    print("-" * 40)

    for rank, doc in enumerate(results, start=1):
        print(f"  {rank}위. {doc['file_name']}  (점수: {doc['score']})")

    print("-" * 40)


# ==============================
# 실습용 메인 실행
# ==============================

# 테스트할 질문을 이 변수에 넣고 실행한다.
question = "앳플리 체중계가 앱이랑 연결이 안 될 때 어떻게 해야 해?"

print("=" * 60)
print(f"질문: {question}")
print("=" * 60)

# 1단계: 질문과 관련 있는 문서를 검색한다.
search_results = search_wiki(question, top_k=TOP_K)

if not search_results:
    print("[안내] 검색된 문서가 없어 답변을 생성할 수 없습니다.")
    exit(0)

# 검색된 문서 목록 출력
print_search_results(search_results)

# 2단계: 검색된 문서만 하나의 Context로 합친다.
rag_context = build_rag_context(search_results)

# 참고 문서명 목록 추출
source_files = [doc["file_name"] for doc in search_results]

# 3단계: Claude에게 검색된 문서를 넘기고 답변을 받는다.
print("\n[Claude 답변 생성 중...]\n")

answer = ask_claude_with_rag(question, rag_context, source_files)

print(answer)
print("\n" + "=" * 60)
