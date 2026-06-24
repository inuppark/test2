from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import os
import json
import math
import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_PATH = Path("data/rag_index.json")

SOURCE_CONFIGS = [
    {"group": "보고서", "path": Path("reports")},
    {"group": "사내위키", "path": Path("data/wiki")}
]

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

def detect_report_type(file_name, source_group):
    lower_name = file_name.lower()

    if source_group == "사내위키":
        return "사내위키"

    if "voc_report" in lower_name:
        return "VOC 보고서"

    if "pdf_summary_report" in lower_name or "pdf_integrated_summary_report" in lower_name:
        return "PDF 보고서"

    if "meeting_report" in lower_name:
        return "회의록 보고서"

    if "image_analysis_report" in lower_name:
        return "이미지 보고서"

    return "기타"

def read_text_file(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start >= len(text):
            break

    return chunks

def embed_text(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding

def build_rag_index():
    INDEX_PATH.parent.mkdir(exist_ok=True)

    all_files = []

    for config in SOURCE_CONFIGS:
        source_dir = config["path"]
        source_group = config["group"]

        source_dir.mkdir(parents=True, exist_ok=True)

        files = [
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in [".txt", ".md"]
        ]

        for file_path in files:
            all_files.append({
                "path": file_path,
                "source_group": source_group
            })

    if not all_files:
        raise ValueError("인덱싱할 txt/md 문서가 없습니다. reports 또는 data/wiki 폴더를 확인하세요.")

    documents = []

    for item in all_files:
        file_path = item["path"]
        source_group = item["source_group"]

        text = read_text_file(file_path)

        if not text.strip():
            continue

        report_type = detect_report_type(file_path.name, source_group)
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            embedding = embed_text(chunk)

            documents.append({
                "source_group": source_group,
                "source_file": file_path.name,
                "source_path": str(file_path),
                "report_type": report_type,
                "chunk_id": idx + 1,
                "text": chunk,
                "embedding": embedding
            })

    if not documents:
        raise ValueError("인덱싱할 문서 내용이 없습니다.")

    index_data = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL,
        "source_count": len(all_files),
        "chunk_count": len(documents),
        "documents": documents
    }

    INDEX_PATH.write_text(
        json.dumps(index_data, ensure_ascii=False),
        encoding="utf-8"
    )

    return {
        "index_path": str(INDEX_PATH),
        "source_count": len(all_files),
        "chunk_count": len(documents),
        "created_at": index_data["created_at"]
    }

def load_rag_index():
    if not INDEX_PATH.exists():
        raise ValueError("RAG 인덱스가 없습니다. 먼저 인덱스를 생성하세요.")

    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))

def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)

def filter_documents(documents, search_scope):
    if search_scope == "전체":
        return documents

    if search_scope == "보고서 전체":
        return [item for item in documents if item.get("source_group") == "보고서"]

    if search_scope == "사내위키":
        return [item for item in documents if item.get("source_group") == "사내위키"]

    return [
        item for item in documents
        if item.get("report_type") == search_scope
    ]

def calculate_confidence(search_results):
    if not search_results:
        return {
            "level": "낮음",
            "top_score": 0,
            "reason": "검색된 근거가 없습니다."
        }

    top_score = search_results[0]["score"]

    if top_score >= 0.45:
        return {
            "level": "높음",
            "top_score": round(top_score, 4),
            "reason": "질문과 관련성이 높은 근거가 검색되었습니다."
        }

    if top_score >= 0.30:
        return {
            "level": "보통",
            "top_score": round(top_score, 4),
            "reason": "관련 근거가 있으나 일부 내용은 추가 확인이 필요합니다."
        }

    return {
        "level": "낮음",
        "top_score": round(top_score, 4),
        "reason": "검색 점수가 낮아 답변 신뢰도가 낮습니다. 질문을 더 구체화하거나 문서를 추가해야 합니다."
    }

def search_rag(question, top_k=5, search_scope="전체"):
    index_data = load_rag_index()
    documents = filter_documents(index_data["documents"], search_scope)

    if not documents:
        raise ValueError(f"'{search_scope}' 범위에 검색할 문서가 없습니다. 인덱스를 다시 생성하거나 다른 검색 범위를 선택하세요.")

    question_embedding = embed_text(question)

    scored = []

    for item in documents:
        score = cosine_similarity(question_embedding, item["embedding"])

        scored.append({
            "score": score,
            "source_group": item.get("source_group", ""),
            "source_file": item["source_file"],
            "source_path": item["source_path"],
            "report_type": item.get("report_type", "기타"),
            "chunk_id": item["chunk_id"],
            "text": item["text"]
        })

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def answer_with_rag(question, top_k=5, search_scope="전체"):
    search_results = search_rag(
        question=question,
        top_k=top_k,
        search_scope=search_scope
    )

    confidence = calculate_confidence(search_results)

    context = ""

    for idx, item in enumerate(search_results, start=1):
        context += f"""

[근거 {idx}]
출처 그룹: {item["source_group"]}
문서 유형: {item["report_type"]}
파일명: {item["source_file"]}
chunk: {item["chunk_id"]}
유사도: {round(item["score"], 4)}
내용:
{item["text"]}
"""

    prompt = f"""
당신은 AI 업무 자동화 Agent Portal의 RAG 기반 지식 검색 Agent입니다.

아래 검색된 문서 내용을 근거로 사용자의 질문에 답변하세요.

[검색 범위]
{search_scope}

[검색 신뢰도]
신뢰도: {confidence["level"]}
최고 유사도: {confidence["top_score"]}
신뢰도 판단 이유: {confidence["reason"]}

중요 규칙:
1. 반드시 제공된 근거 안에서만 답하세요.
2. 근거에 없는 내용은 "현재 인덱스된 문서에서는 확인되지 않습니다"라고 말하세요.
3. 답변에는 어떤 문서를 근거로 했는지 파일명을 표시하세요.
4. 검색 범위 밖의 내용은 추측하지 마세요.
5. 보고서 결과와 사내위키 기준이 함께 있으면, 보고서는 현재 상황, 사내위키는 처리 기준으로 구분하세요.
6. 신뢰도가 낮음이면 단정하지 말고 추가 문서 업로드나 질문 구체화를 제안하세요.

[사용자 질문]
{question}

[검색된 근거]
{context}

아래 형식으로 답변하세요.

## 답변

## 근거 문서

## 신뢰도 및 한계

## 확인 필요 사항
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt
    )

    return response.output_text, search_results, confidence

def save_rag_log(question, answer, search_results, search_scope, confidence=None):
    os.makedirs("logs", exist_ok=True)

    log_path = "logs/rag_agent_log.csv"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    source_files = ", ".join(sorted(set([item["source_file"] for item in search_results])))

    if confidence is None:
        confidence = calculate_confidence(search_results)

    row = pd.DataFrame([{
        "run_time": run_time,
        "question": question,
        "search_scope": search_scope,
        "confidence": confidence.get("level"),
        "top_score": confidence.get("top_score"),
        "answer_preview": answer[:200],
        "source_files": source_files,
        "status": "success"
    }])

    if os.path.exists(log_path):
        old = pd.read_csv(log_path)
        new = pd.concat([old, row], ignore_index=True)
    else:
        new = row

    new.to_csv(log_path, index=False, encoding="utf-8-sig")

    return log_path
