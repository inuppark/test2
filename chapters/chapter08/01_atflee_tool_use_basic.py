"""
Chapter 8-1: Claude Tool Use 기본 실습
앳플리 위키 검색 도구를 Claude에게 제공하는 방법을 배운다.

[Chapter 7 RAG vs Chapter 8 Tool Use 차이]

Chapter 7:
  Python 코드 → data/wiki 검색 → Claude에게 결과 전달 → 답변
  (Python이 검색 여부를 결정)

Chapter 8:
  Claude가 질문을 보고 → "검색이 필요하다" 판단 → 도구 요청 →
  Python이 도구 실행 → Claude에게 결과 전달 → 답변
  (Claude가 도구 사용 여부를 스스로 결정)
"""

import os
import sys
import json

# ==============================
# 프로젝트 루트 경로 설정
# ==============================
# 이 파일 위치: chapters/chapter08/ → 두 단계 위가 프로젝트 루트다.
# 어느 디렉토리에서 실행해도 utils 패키지를 찾을 수 있게 sys.path에 추가한다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from anthropic import Anthropic

# utils.rag_utils의 검색 함수를 도구 실행에 재사용한다.
from utils.rag_utils import search_wiki, build_rag_context, get_source_file_names


# ==============================
# API Key 설정
# ==============================

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

client = Anthropic(api_key=api_key)
model_name = "claude-sonnet-4-5"


# ==============================
# 도구(Tool) 정의
# ==============================
# Claude에게 "이런 도구를 사용할 수 있다"고 알려주는 스키마다.
# Claude는 이 정보를 보고 언제 어떤 도구를 호출할지 스스로 결정한다.

tools = [
    {
        "name": "search_atflee_wiki",
        "description": "앳플리 제품, 앱, 배송, 환불, AS, 문의 관련 data/wiki 문서를 검색한다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "앳플리 위키에서 검색할 사용자 질문"
                }
            },
            "required": ["question"]
        }
    }
]


# ==============================
# 도구 실행 함수
# ==============================
# Claude가 "search_atflee_wiki 도구를 써달라"고 요청하면
# Python 코드가 이 함수를 실행하고 결과를 Claude에게 돌려준다.

def search_atflee_wiki(question):
    """
    앳플리 위키 문서를 검색하고 결과를 반환한다.
    반환값은 json.dumps()로 직렬화해서 Claude에게 tool_result로 전달한다.
    """
    search_results = search_wiki(question, top_k=3)
    rag_context = build_rag_context(search_results)
    source_files = get_source_file_names(search_results)

    return {
        "source_files": source_files,
        "search_results": [
            {
                "file_name": result["file_name"],
                "score": result["score"],
                "snippet": result.get("snippet", "")
            }
            for result in search_results
        ],
        "rag_context": rag_context
    }


# ==============================
# System Prompt
# ==============================

system_prompt = """
# Role
너는 앳플리 Tool Use 기반 업무 지원 봇이다.

# Goal
사용자의 앳플리 제품, 앱, 배송, 환불, AS, 문의 관련 질문에 답하기 위해
필요하면 search_atflee_wiki 도구를 사용한다.

# Context
search_atflee_wiki 도구는 data/wiki 문서를 검색해
관련 문서, 점수, 스니펫, 문서 내용을 반환한다.
도구 결과의 rag_context 안에 실제 문서 내용이 들어 있다.

# Rules
- 앳플리 관련 질문에는 가능한 한 search_atflee_wiki 도구를 사용한다.
- 도구 결과에 있는 정보만 확정적으로 말한다.
- 도구 결과에 없는 내용은 추측하지 않는다.
- 실제 주문 상태, 배송 상태, AS 접수 상태를 지어내지 않는다.
- 가격, 재고, 품절, 이벤트, 프로모션은 변동될 수 있으므로 단정하지 않는다.
- 정책, 보증, 교환/환불, AS 조건은 확실하지 않으면 "정확한 확인이 필요합니다"라고 말한다.
- 개인정보, 주문번호, 연락처, 주소 등 민감정보는 공개 채팅에 입력하지 않도록 안내한다.
- 답변 마지막에 참고 문서를 표시한다.

# Output Format
아래 형식으로 최종 답변한다.

1. 간단한 답변
2. 도구로 확인한 근거
3. 바로 해볼 수 있는 것
4. 확인이 필요한 것
5. 참고 문서
"""


# ==============================
# 실습용 테스트 질문
# ==============================
# 이 변수를 바꿔서 다양한 질문을 테스트할 수 있다.

user_question = "앳플리 체중계가 앱이랑 연결이 안 돼요. 뭘 확인해야 해요?"

print("=" * 60)
print(f"질문: {user_question}")
print("=" * 60)


# ==============================
# Step 1: 첫 번째 Claude 호출
# ==============================
# Claude에게 질문과 함께 도구 목록을 전달한다.
# Claude는 응답에서 "이 도구가 필요하다"는 신호(tool_use)를 보낼 수 있다.

first_response = client.messages.create(
    model=model_name,
    max_tokens=1000,
    temperature=0.2,
    system=system_prompt,
    tools=tools,
    messages=[
        {"role": "user", "content": user_question}
    ]
)

# Claude가 도구를 요청했는지 확인한다.
# stop_reason이 "tool_use"이면 Claude가 도구 실행을 기다리고 있다는 뜻이다.
tool_use_block = None

for block in first_response.content:
    if block.type == "tool_use":
        tool_use_block = block
        break


# ==============================
# Step 2: 도구 실행
# ==============================

if tool_use_block is None:
    # Claude가 도구 없이 바로 답변한 경우
    print("\n[안내] Claude가 도구를 요청하지 않았습니다.")
    print("\n[Claude 답변]")
    for block in first_response.content:
        if hasattr(block, "text"):
            print(block.text)
else:
    # Claude가 search_atflee_wiki 도구를 요청한 경우
    tool_use_id = tool_use_block.id
    tool_name = tool_use_block.name
    tool_input = tool_use_block.input
    search_question = tool_input["question"]

    print(f"\n[Claude tool_use 요청]")
    print(f"도구명: {tool_name}")
    print(f"입력: {search_question}")

    # Python 코드가 직접 도구를 실행한다.
    tool_result = search_atflee_wiki(search_question)

    print(f"\n[도구 실행 결과]")
    for item in tool_result["search_results"]:
        print(f"  - {item['file_name']} / 점수: {item['score']}")
        if item["snippet"]:
            preview = item["snippet"].replace("\n", " ")[:80]
            print(f"    스니펫: {preview}...")

    # ==============================
    # Step 3: 두 번째 Claude 호출
    # ==============================
    # 첫 번째 응답(assistant) + 도구 실행 결과(tool_result)를 포함해서
    # Claude에게 다시 요청한다.
    # Claude는 도구 결과를 보고 최종 답변을 생성한다.

    final_response = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.2,
        system=system_prompt,
        tools=tools,
        messages=[
            # 사용자 질문
            {"role": "user", "content": user_question},
            # 첫 번째 Claude 응답 (tool_use 요청 포함)
            {"role": "assistant", "content": first_response.content},
            # 도구 실행 결과를 tool_result 형식으로 전달
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(tool_result, ensure_ascii=False, indent=2)
                    }
                ]
            }
        ]
    )

    # ==============================
    # Step 4: 최종 답변 출력
    # ==============================

    print("\n[최종 답변]")
    print("-" * 60)
    for block in final_response.content:
        if hasattr(block, "text"):
            print(block.text)
    print("=" * 60)
