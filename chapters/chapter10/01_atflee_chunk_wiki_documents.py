"""
Chapter 10-1: 앳플리 위키 문서 청킹 실습

목적:
  임베딩을 만들기 전 단계로, data/wiki Markdown 문서를 작은 청크 단위로 나누고
  각 청크에 메타데이터를 붙여 data/rag/atflee_wiki_chunks.json에 저장한다.

배경:
  Chapter 7에서는 문서 전체를 키워드 기반으로 검색했다.
  Chapter 10에서는 문서를 임베딩 벡터로 변환해 의미 기반 검색을 할 예정이다.
  임베딩은 보통 문서 전체가 아니라 300~800자 정도의 청크 단위로 만든다.
  이번 10-1은 임베딩 전 단계인 문서 청킹 작업이다.
"""

import os
import json
from collections import Counter
from datetime import datetime

# 프로젝트 루트 경로를 계산한다.
# 이 파일(chapters/chapter10/) 기준으로 두 단계 위가 프로젝트 루트다.
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# =========================
# 경로 상수
# =========================

# 원본 Markdown 문서 폴더
WIKI_DIR = os.path.join(PROJECT_ROOT, "data", "wiki")

# 청크 결과물 저장 폴더와 파일
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "rag")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "atflee_wiki_chunks.json")

# =========================
# 청킹 파라미터
# =========================

# 청크로 인정하는 최소 글자 수: 이보다 짧으면 인접 문단과 합친다.
MIN_CHARS = 80

# 하나의 청크 최대 글자 수: 초과하면 잘라서 여러 청크로 만든다.
MAX_CHARS = 800

# 문단을 합칠 때 목표 길이: 이 길이 이하면 버퍼에 계속 모은다.
TARGET_CHARS = 500


# =========================
# 함수 1: Markdown 파일 읽기
# =========================

def read_markdown_files():
    """
    data/wiki 폴더의 .md 파일을 읽어온다.

    반환 형태:
    [
        {"file_name": "atflee_app_guide.md", "content": "..."},
        ...
    ]
    """
    documents = []

    if not os.path.exists(WIKI_DIR):
        print(f"data/wiki 폴더를 찾을 수 없습니다: {WIKI_DIR}")
        return documents

    # 파일명 오름차순으로 정렬해 일관된 순서를 보장한다.
    for file_name in sorted(os.listdir(WIKI_DIR)):
        if not file_name.endswith(".md"):
            continue

        file_path = os.path.join(WIKI_DIR, file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        documents.append(
            {
                "file_name": file_name,
                "content":   content
            }
        )

    return documents


# =========================
# 함수 2: 문단 분리
# =========================

def split_into_paragraphs(text):
    """
    Markdown 문서를 문단 단위로 나눈다.
    빈 줄(\n\n) 기준으로 나누고, 앞뒤 공백을 제거한다.
    빈 문단은 제거한다.
    """
    # Markdown 문서에서 빈 줄(연속된 \n\n)은 문단 경계를 나타낸다.
    raw_paragraphs = text.split("\n\n")

    paragraphs = []

    for paragraph in raw_paragraphs:
        cleaned = paragraph.strip()

        # 공백만 있거나 아무 내용도 없는 문단은 제외한다.
        if not cleaned:
            continue

        paragraphs.append(cleaned)

    return paragraphs


# =========================
# 함수 3: 긴 텍스트 분할
# =========================

def split_long_text(text, max_chars=MAX_CHARS):
    """
    max_chars보다 긴 텍스트를 여러 조각으로 나눈다.
    가능하면 줄 단위 경계를 유지하고, 불가피할 때만 글자 수 기준으로 자른다.

    예: 1,200자 텍스트 → 800자 + 400자 두 청크
    """
    # 이미 충분히 짧으면 그대로 반환한다.
    if len(text) <= max_chars:
        return [text]

    chunks = []
    lines  = text.split("\n")
    current = ""

    for line in lines:
        # 현재 버퍼에 이 줄을 더해도 max_chars 이하이면 추가한다.
        if len(current) + len(line) + 1 <= max_chars:
            current += ("\n" if current else "") + line
        else:
            # 버퍼가 있으면 저장하고 새 버퍼를 시작한다.
            if current:
                chunks.append(current)
            current = line

    # 마지막 버퍼를 저장한다.
    if current:
        chunks.append(current)

    # 줄 단위로 잘라도 max_chars를 초과하는 경우 글자 수 기준으로 한 번 더 자른다.
    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            for start in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[start:start + max_chars])

    return final_chunks


# =========================
# 함수 4: 문서 한 개 청킹
# =========================

def build_chunks_for_document(file_name, content):
    """
    문서 하나를 청크 여러 개로 나눈다.

    처리 순서:
    1. 문단 단위로 분리한다.
    2. 긴 문단은 MAX_CHARS 기준으로 먼저 자른다.
    3. 너무 짧은 조각은 버퍼에 모아 TARGET_CHARS 이하로 합친다.
    4. MIN_CHARS 미만 조각은 청크로 만들지 않는다.
    5. 각 청크에 메타데이터를 붙인다.

    반환 형태:
    [
        {
            "chunk_id":    "atflee_app_guide__chunk_001",
            "source_file": "atflee_app_guide.md",
            "chunk_index": 1,
            "text":        "...",
            "char_count":  350
        },
        ...
    ]
    """
    paragraphs = split_into_paragraphs(content)

    chunks = []
    buffer = ""  # 짧은 문단들을 일시적으로 모아두는 버퍼

    for paragraph in paragraphs:
        # 긴 문단은 청킹하기 전에 먼저 나눈다.
        pieces = split_long_text(paragraph, MAX_CHARS)

        for piece in pieces:
            if not buffer:
                # 버퍼가 비어 있으면 현재 조각을 버퍼에 넣는다.
                buffer = piece
            elif len(buffer) + len(piece) + 2 <= TARGET_CHARS:
                # 버퍼 + 현재 조각이 목표 길이 이하이면 합친다.
                buffer = buffer + "\n\n" + piece
            else:
                # 목표 길이를 초과하면 버퍼를 청크로 확정하고 새 버퍼를 시작한다.
                if len(buffer) >= MIN_CHARS:
                    chunks.append(buffer)
                buffer = piece

    # 마지막 버퍼를 처리한다.
    if buffer and len(buffer) >= MIN_CHARS:
        chunks.append(buffer)

    # 합치는 과정에서 MAX_CHARS를 초과한 청크가 생기면 한 번 더 나눈다.
    final_chunks = []

    for chunk in chunks:
        final_chunks.extend(split_long_text(chunk, MAX_CHARS))

    # 청크에 메타데이터를 붙인다.
    result = []

    for index, chunk_text in enumerate(final_chunks, start=1):
        # chunk_id: 파일명(확장자 제거) + 청크 번호 (예: atflee_app_guide__chunk_001)
        chunk_id = f"{file_name.replace('.md', '')}__chunk_{index:03d}"

        result.append(
            {
                "chunk_id":    chunk_id,
                "source_file": file_name,
                "chunk_index": index,
                "text":        chunk_text,
                "char_count":  len(chunk_text)
            }
        )

    return result


# =========================
# 함수 5: 전체 문서 청킹
# =========================

def build_all_chunks():
    """
    data/wiki의 모든 Markdown 문서를 읽어 청크 목록을 만든다.

    반환: 모든 문서의 청크를 합친 리스트
    """
    documents = read_markdown_files()

    if not documents:
        return []

    all_chunks = []

    for document in documents:
        file_chunks = build_chunks_for_document(
            document["file_name"],
            document["content"]
        )
        all_chunks.extend(file_chunks)

    return all_chunks


# =========================
# 함수 6: JSON 저장
# =========================

def save_chunks(chunks):
    """
    청크 목록을 data/rag/atflee_wiki_chunks.json에 저장한다.

    저장 형태:
    {
        "created_at": "2026-06-25 ...",
        "project": "atflee",
        "description": "...",
        "chunk_count": 42,
        "chunks": [...]
    }
    """
    # 출력 폴더가 없으면 만든다.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    payload = {
        "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project":     "atflee",
        "description": "Atflee wiki chunks for embedding and vector search practice",
        "chunk_count": len(chunks),
        "chunks":      chunks
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return OUTPUT_PATH


# =========================
# 함수 7: 요약 출력
# =========================

def print_summary(chunks):
    """
    생성된 청크 요약을 콘솔에 출력한다.
    전체 청크 수, 파일별 청크 수, 첫 번째 청크 미리 보기를 보여준다.
    """
    print("\n[청크 생성 요약]")
    print(f"  전체 청크 수: {len(chunks)}")

    # Counter로 파일별 청크 수를 빠르게 집계한다.
    counter = Counter(chunk["source_file"] for chunk in chunks)

    print("\n[파일별 청크 수]")
    for file_name, count in counter.items():
        print(f"  - {file_name}: {count}개")

    # 글자 수 통계
    char_counts = [chunk["char_count"] for chunk in chunks]
    print("\n[청크 글자 수 통계]")
    print(f"  최소: {min(char_counts)}자")
    print(f"  최대: {max(char_counts)}자")
    print(f"  평균: {sum(char_counts) // len(char_counts)}자")

    if chunks:
        print("\n[첫 번째 청크 예시]")
        first = chunks[0]
        print(f"  chunk_id:    {first['chunk_id']}")
        print(f"  source_file: {first['source_file']}")
        print(f"  char_count:  {first['char_count']}")
        print("  text preview:")
        # 미리 보기는 앞 200자까지만 출력한다.
        preview = first["text"][:200].replace("\n", " ")
        print(f"    {preview}...")


# =========================
# 실행부
# =========================

if __name__ == "__main__":
    print("[앳플리 위키 문서 청킹 시작]")
    print(f"  wiki 폴더:  {WIKI_DIR}")
    print(f"  저장 경로:  {OUTPUT_PATH}")
    print(f"  MIN_CHARS:  {MIN_CHARS}자")
    print(f"  MAX_CHARS:  {MAX_CHARS}자")
    print(f"  TARGET_CHARS: {TARGET_CHARS}자")

    # 1단계: 모든 문서를 청크로 나눈다.
    chunks = build_all_chunks()

    if not chunks:
        print("\n생성된 청크가 없습니다. data/wiki 문서를 확인하세요.")
    else:
        # 2단계: 청크를 JSON으로 저장한다.
        saved_path = save_chunks(chunks)
        print_summary(chunks)

        print("\n[저장 완료]")
        print(f"  {saved_path}")

        print("\n[다음 단계]")
        print("  Chapter 10-2에서는 이 청크들을 임베딩 벡터로 변환하고 저장합니다.")
        print("  Claude의 Embeddings API 또는 sentence-transformers를 사용할 수 있습니다.")
