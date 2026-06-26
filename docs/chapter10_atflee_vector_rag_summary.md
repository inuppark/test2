# Chapter 10 정리: 앳플리 벡터 RAG 실습

---

## 1. Chapter 10의 목표

Chapter 10의 목표는 **키워드 기반 RAG를 벡터 기반 RAG로 확장**하는 것이다.

책에서는 도슨트봇을 RAG로 증강하지만, 앳플리 프로젝트에서는 **앳플리 봇**과 **AX Console**을 대상으로 적용했다.

- `data/wiki` 폴더의 앳플리 정책 문서를 더 똑똑하게 검색한다.
- 질문과 유사한 문서 청크를 찾아 Claude에게 전달한다.
- Claude가 근거 기반 답변을 생성하도록 만든다.

Chapter 7에서 만든 키워드 RAG와 병행해, 두 방식을 비교하고 각각의 강점과 한계를 확인하는 것도 이번 챕터의 중요한 실습이다.

---

## 2. 기존 Chapter 7 RAG와 Chapter 10 RAG의 차이

| 비교 항목 | Chapter 7 키워드 RAG | Chapter 10 벡터 RAG |
|---|---|---|
| 검색 방식 | 질문 단어와 문서 단어의 겹침 기반 | 문서 청크를 벡터로 변환 후 코사인 유사도 계산 |
| 입력 데이터 | `data/wiki` 마크다운 문서 전체 | `data/wiki` 문서를 청크로 분리한 JSON |
| 검색 단위 | 파일 단위 | 문단(청크) 단위 |
| 장점 | 구현이 쉽고 빠르며, 명확한 키워드가 있는 질문에 강함 | 의미 기반 검색으로 확장 가능하며, 세밀한 문단 검색 가능 |
| 한계 | 표현이 다르면 관련 문서를 놓칠 수 있음 | 현재 TF-IDF 방식은 진짜 semantic embedding은 아님 |
| 앳플리 적용 위치 | AX Console 키워드 RAG / `utils/rag_utils.py` | AX Console 벡터 RAG 탭 / `utils/vector_rag_utils.py` |

---

## 3. 구현한 파일 목록

### Chapter 10 실습 파일

| 파일 | 역할 |
|---|---|
| `chapters/chapter10/01_atflee_chunk_wiki_documents.py` | `data/wiki` 문서를 문단 단위로 청크 분리하고 JSON으로 저장 |
| `chapters/chapter10/02_atflee_build_tfidf_vector_index.py` | 청크에서 TF-IDF 벡터 인덱스를 생성하고 JSON으로 저장 |
| `chapters/chapter10/03_atflee_tfidf_vector_search.py` | 질문 벡터를 만들고 코사인 유사도로 관련 청크 TOP K 검색 |
| `chapters/chapter10/04_atflee_vector_rag_answer.py` | 벡터 검색 결과를 Context로 Claude에 전달해 답변 생성 |
| `chapters/chapter10/05_atflee_vector_rag_utils_test.py` | `utils/vector_rag_utils.py` 모듈 통합 테스트 |
| `chapters/chapter10/06_atflee_rag_quality_compare_report.py` | 키워드 RAG vs 벡터 RAG hit rate 비교 리포트 생성 |

### 공통 모듈

| 파일 | 역할 |
|---|---|
| `utils/vector_rag_utils.py` | 벡터 인덱스 로드, 청크 검색, RAG 답변 생성 함수 모음 |

### 앱 연결

| 파일 | 역할 |
|---|---|
| `apps/ax_console_v0.py` | AX Console에 "벡터 RAG" 탭을 추가해 UI로 제공 |

### 생성 산출물

| 파일 | 설명 |
|---|---|
| `data/rag/atflee_wiki_chunks.json` | 청크 분리 결과 (Git 포함, Streamlit Cloud 배포용) |
| `data/rag/atflee_tfidf_vector_index.json` | TF-IDF 벡터 인덱스 (Git 포함, Streamlit Cloud 배포용) |
| `reports/chapter10/atflee_rag_quality_compare_*.json` | 검색 품질 비교 리포트 (**Git 미포함**, `.gitignore`에 의해 제외) |

---

## 4. 전체 처리 흐름

```
data/wiki 마크다운 문서
        ↓
   문단/청크 분리
  (01_atflee_chunk_wiki_documents.py)
        ↓
  청크 JSON 저장
  (data/rag/atflee_wiki_chunks.json)
        ↓
 TF-IDF 벡터 인덱스 생성
  (02_atflee_build_tfidf_vector_index.py)
        ↓
  벡터 인덱스 JSON 저장
  (data/rag/atflee_tfidf_vector_index.json)
        ↓
   사용자 질문 입력
  (AX Console 벡터 RAG 탭)
        ↓
   질문 벡터 생성
  (build_query_vector)
        ↓
  코사인 유사도 계산
  (search_similar_chunks)
        ↓
 관련 청크 TOP 3 검색
        ↓
  Claude에게 Context 전달
  (generate_vector_rag_answer)
        ↓
   근거 기반 답변 생성
        ↓
 AX Console 벡터 RAG 탭에서 확인
```

---

## 5. 실습 결과 요약

- **청크 수**: 15개 (`data/wiki` 문서 전체 기준)
- **Vocabulary 크기**: 536개 (TF-IDF 인덱스 기준)
- `data/rag` 인덱스 파일을 `.gitignore` 예외 처리해 Git에 포함했다.
  - Streamlit Cloud에서도 별도 빌드 없이 벡터 RAG 탭이 바로 작동하도록 **배포 안정화** 완료.
- AX Console에 **"벡터 RAG" 탭**을 추가했다.
  - 질문 벡터 토큰 표시, 벡터 검색 TOP 3, Claude 답변, 키워드 vs 벡터 비교까지 한 화면에서 확인 가능.
- Streamlit Cloud 온라인 환경에서 벡터 RAG 답변이 정상 생성되는 것을 확인했다.

---

## 6. 키워드 RAG vs 벡터 RAG 품질 비교 결과

8개 앳플리 테스트 질문으로 hit rate를 측정했다.

| 방식 | TOP 1 hit rate | TOP 3 hit rate |
|---|---|---|
| 키워드 검색 (Chapter 7) | 0.625 (5/8) | 0.875 (7/8) |
| 벡터 검색 (Chapter 10 TF-IDF) | 0.625 (5/8) | 0.875 (7/8) |

### 해석

- 현재 TF-IDF 벡터 검색은 키워드 검색보다 압도적으로 좋지는 않다.
- 두 방식은 전체 hit rate가 같지만, **질문 유형별로 강점이 다르다**.
- "배송 얼마나 걸려?", "T9은 어떤 제품이야?"처럼 **명확한 키워드가 있는 질문**은 키워드 검색이 강하다.
- "AS 접수됐는지 확인해줘."처럼 **행위 중심 질문**은 벡터 검색이 더 나은 파일을 TOP 1에 올렸다.
- "제품이 불량 같고 교환하고 싶어요."처럼 **복합 의도 질문**은 두 방식 모두 TOP 3에서 놓쳤다. 보강이 필요한 유형이다.

---

## 7. 현재 TF-IDF 방식의 한계

### 1. 한글 형태소 분석 미적용

"환불은", "T9은", "연결이"처럼 조사가 붙으면 토큰이 분리되지 않아 검색 품질이 낮아질 수 있다.
`normalize_token` 함수로 일부 보완했지만 완전하지 않다.

### 2. 진짜 의미 이해는 아님

TF-IDF는 단어 빈도와 가중치 기반이다.
"배송 얼마나 걸려?"와 "배송 기간"이 같은 의미라는 것을 모델처럼 이해하지 못한다.
동의어 사전이나 규칙을 수동으로 관리해야 한다.

### 3. 문서 수가 늘어나면 품질 관리 필요

문서가 많아질수록 단어 분포가 복잡해지고, 의도와 맞지 않는 문서가 높은 점수를 받을 수 있다.
인덱스 재생성 기준과 품질 검증 프로세스가 필요해진다.

---

## 8. 다음 고도화 방향

### 1. 하이브리드 검색

키워드 검색 결과와 벡터 검색 결과를 합쳐서 사용한다.
어느 한 방식만 쓰는 것보다 다양한 질문 유형을 안정적으로 커버할 수 있다.

### 2. 한국어 형태소 분석

조사, 어미, 복합명사를 더 잘 처리한다.
Okt, Mecab, Kiwi 등의 오픈소스 형태소 분석기를 나중에 검토할 수 있다.

### 3. 실제 임베딩 API 사용

TF-IDF 대신 Voyage AI, OpenAI Embeddings 등 실제 semantic embedding을 사용하면
의미 기반 검색 품질이 크게 높아질 가능성이 크다.
단, 외부 API 비용과 문서 보안(고객 정보 포함 여부) 고려가 필요하다.

### 4. Reranking

벡터 검색으로 TOP 10 후보를 뽑고, Claude 또는 별도 reranker로 다시 정렬한다.
최종 TOP 3의 정확도를 높일 수 있다.

### 5. AX Console 개선

- 키워드 RAG와 벡터 RAG를 사용자가 선택하는 UI 구조
- 하이브리드 RAG 탭 추가
- 검색 품질 평가 리포트 다운로드 기능 추가

---

## 9. 앳플리 AX 관점에서의 의미

- **앳플리 봇**이 단순 FAQ 챗봇에서 **근거 기반 검색형 봇**으로 발전했다.
- `data/wiki`를 지식베이스로 활용하는 구조가 만들어졌다.
  앞으로 문서를 업데이트하면 인덱스를 재생성하는 것만으로 봇 지식이 갱신된다.
- **AX Console**에 키워드 RAG, Tool Use Agent, Prompt Caching, 벡터 RAG가 통합되었다.
  단일 콘솔에서 앳플리 봇의 다양한 AI 기능을 직접 테스트할 수 있다.
- 향후 고객 문의 자동 분류, CS 답변 보조, 내부 정책 검색 자동화로 확장할 수 있다.

---

## 10. Chapter 10 완료 기준

- [x] `data/wiki` 문서 청크 분리
- [x] TF-IDF 벡터 인덱스 생성
- [x] 질문 벡터 검색 구현
- [x] Claude 벡터 RAG 답변 생성
- [x] `utils/vector_rag_utils.py` 공통 모듈화
- [x] AX Console 벡터 RAG 탭 추가
- [x] Streamlit Cloud 배포 안정화
- [x] 키워드 RAG vs 벡터 RAG 품질 비교 리포트 생성
- [x] Chapter 10 정리 문서 작성
