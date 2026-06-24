import os
import re

# ==============================
# 설정값
# ==============================

# data/wiki 폴더 경로 (프로젝트 루트 기준)
WIKI_DIR = "data/wiki"

# 검색 결과로 가져올 최대 문서 수
TOP_K = 3

# 문서 미리보기 글자 수
PREVIEW_LENGTH = 300


# ==============================
# 1. 문서 로드 함수
# ==============================

def load_wiki_documents():
    """
    data/wiki 폴더 안의 .md 파일을 모두 읽어
    [{"file_name": 파일명, "content": 내용}, ...] 형태로 반환한다.
    """
    # 폴더가 없으면 안내 메시지 출력 후 빈 리스트 반환
    if not os.path.exists(WIKI_DIR):
        print(f"[오류] '{WIKI_DIR}' 폴더를 찾을 수 없습니다.")
        print("프로젝트 루트 디렉토리에서 실행하고 있는지 확인해주세요.")
        return []

    documents = []

    # 폴더 안의 파일을 순서대로 읽는다.
    for file_name in sorted(os.listdir(WIKI_DIR)):
        # .md 파일만 처리한다.
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

    처리 순서:
    1. 소문자로 변환 (영어 대소문자 통일)
    2. 줄바꿈, 특수문자를 공백으로 변환
    3. 공백 기준으로 분리
    4. 너무 짧은 토큰(1글자 이하) 제거
    """
    # 소문자로 변환
    text = text.lower()

    # 한글/영어/숫자/공백을 제외한 문자는 공백으로 변환
    # 마크다운 기호(#, *, -, [, ], (, ))도 공백으로 처리된다.
    text = re.sub(r"[^\w\s가-힣]", " ", text)

    # 공백 기준으로 단어를 나눈다.
    tokens = text.split()

    # 너무 짧은 토큰은 노이즈가 많으므로 제거한다.
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


# ==============================
# 3. 문서 관련도 점수 계산 함수
# ==============================

def score_document(question, document_content):
    """
    질문과 문서 사이의 키워드 겹침 점수를 계산한다.

    방식:
    - 질문을 토큰으로 나눈다.
    - 문서를 토큰 집합(set)으로 만든다.
    - 질문 토큰 중 문서에 포함된 것의 수를 점수로 반환한다.
    - set을 사용하므로 같은 단어가 문서에 여러 번 나와도 1번으로 계산한다.
    """
    # 질문을 토큰으로 분리
    question_tokens = tokenize(question)

    # 문서를 토큰 집합으로 변환 (중복 제거, 검색 속도 향상)
    document_token_set = set(tokenize(document_content))

    # 질문 토큰 중 문서에 포함된 것 개수를 센다.
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

    반환 형식:
    [
        {
            "file_name": 파일명,
            "content": 전체 내용,
            "score": 관련도 점수
        },
        ...
    ]
    """
    # 문서를 모두 불러온다.
    documents = load_wiki_documents()

    # 문서가 하나도 없으면 안내 메시지 출력
    if not documents:
        print("[안내] data/wiki 폴더에 .md 문서가 없습니다.")
        return []

    # 각 문서에 점수를 계산해 추가한다.
    scored_documents = []

    for doc in documents:
        score = score_document(question, doc["content"])
        scored_documents.append({
            "file_name": doc["file_name"],
            "content": doc["content"],
            "score": score
        })

    # 점수가 높은 순으로 정렬한다.
    # 점수가 같으면 파일명 알파벳 순서를 유지한다.
    scored_documents.sort(key=lambda x: x["score"], reverse=True)

    # 상위 top_k개만 반환한다.
    return scored_documents[:top_k]


# ==============================
# 5. 검색 결과 출력 함수
# ==============================

def print_search_results(question, results):
    """
    검색 결과를 사람이 읽기 쉽게 출력한다.
    """
    print("=" * 60)
    print(f"질문: {question}")
    print("=" * 60)

    if not results:
        print("[안내] 검색 결과가 없습니다.")
        return

    for rank, doc in enumerate(results, start=1):
        print(f"\n[{rank}위] {doc['file_name']}  (점수: {doc['score']})")
        print("-" * 40)

        # 미리보기: 앞 PREVIEW_LENGTH 글자만 출력
        preview = doc["content"][:PREVIEW_LENGTH].replace("\n", " ")
        print(f"미리보기: {preview}...")

    print("\n" + "=" * 60)


# ==============================
# 실습용 메인 실행
# ==============================

# 테스트할 질문을 이 변수에 넣고 실행한다.
question = "앳플리 체중계가 앱이랑 연결이 안 될 때 어떻게 해야 해?"

# 키워드 기반 문서 검색 실행
results = search_wiki(question, top_k=TOP_K)

# 결과 출력
print_search_results(question, results)
