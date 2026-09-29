# 테디노트의 랭체인을 활용한 RAG 비법노트: 심화편 — 최신 코드 복습 노트

## 이 레포지토리는?

[teddylee777/langchain-kr](https://github.com/teddylee777/langchain-kr)에 있는 책 예제 코드는 작성된 지 시간이 지나서, 지금의 LangChain(1.x)에서는 에러가 나거나 더 이상 권장되지 않는(deprecated) API를 쓰는 부분이 많습니다.
이 레포지토리는 책을 공부하면서 **최신 LangChain 방식으로 직접 고쳐서 실행해 본 코드와 정리 문서**를 책 목차 순서대로 모아둔 복습용 저장소입니다.

- 날짜별 학습 기록은 [hanwha_0902](https://github.com/TechieMoon/hanwha_0902)에 있고, 여기는 같은 내용을 **책 챕터 기준**으로 재정리한 것입니다.
- 폴더는 `chXX/`(챕터), 파일은 `절번호_내용.ipynb` 형식입니다. 예) `ch13/05_parent_document_retriever.ipynb` = CHAPTER 13의 05절
- 한 노트북이 여러 절을 다루면 `01-05_...`처럼 범위로 표시했습니다.
- 같은 이름의 `.md` 파일이 그 노트북의 정리 문서입니다.

## 실행 환경

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

루트에 `.env` 파일을 만들고 `.env.example`을 참고해 API 키를 넣어주세요. (`.env`는 `.gitignore`에 포함되어 있어 커밋되지 않습니다.)
노트북은 각 챕터 폴더 안에서 실행하는 것을 기준으로 작성되어 있습니다. (`data/`, `prompts/` 등 상대 경로 사용)

## 목차

코드/문서가 있는 절은 링크가 걸려 있고, `-`는 아직 정리하지 않은 절입니다.

### PART 01 고급 RAG


**CHAPTER 01 리랭커로 검색된 문서 순위 조정하기** → [`ch01/`](ch01/)

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | 교차 인코더 리랭커 | [01_cross_encoder_reranker.ipynb](ch01/01_cross_encoder_reranker.ipynb) | [01_cross_encoder_reranker.md](ch01/01_cross_encoder_reranker.md) | `langchain.retrievers` → `langchain_classic.retrievers` |
| 02 | Cohere 리랭커 | [02_cohere_reranker.ipynb](ch01/02_cohere_reranker.ipynb) | [02_cohere_reranker.md](ch01/02_cohere_reranker.md) | `langchain_classic.retrievers.contextual_compression` 사용 |
| 03 | Jina 리랭커 | - | - |  |
| 04 | FlashRank 리랭커 | - | - |  |

**CHAPTER 02 RAG 파이프라인**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | 네이버 뉴스 기사 기반 QA 시스템 | - | - |  |
| 02 | RAPTOR | - | - |  |
| 03 | 무료 오픈 모델로 구현하는 문서 기반 QA 시스템 | - | - |  |

**CHAPTER 03 사전에 정의된 체인**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | Stuff 요약 | - | - |  |
| 02 | Map-Reduce 요약 | - | - |  |
| 03 | Map-Refine 요약 | - | - |  |
| 04 | Chain of Density(CoD) 요약 | - | - |  |
| 05 | Clustering-Map-Refine 요약 | - | - |  |
| 06 | SQL 쿼리 생성기 | - | - |  |

### PART 02 LCEL 고급 문법


**CHAPTER 04 LCEL 고급 문법으로 Runnable 활용하기**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | RunnablePassthrough | - | - |  |
| 02 | Runnable 구조 검토하기 | - | - |  |
| 03 | RunnableLambda | - | - |  |
| 04 | RunnableBranch와 RunnableLambda를 이용한 라우팅 | - | - |  |
| 05 | RunnableParallel | - | - |  |
| 06 | config로 LLM이나 프롬프트를 동적으로 변경하기 | - | - |  |
| 07 | @chain 데코레이터로 Runnable 설정하기 | - | - |  |

### PART 03 RAG 평가와 개선


**CHAPTER 05 RAGAS로 답변 평가하기**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | 합성 테스트 데이터셋 생성하기 | - | - |  |
| 02 | RAGAS로 평가하기 | - | - |  |
| 03 | 테스트 데이터셋 번역 및 업로드 관리하기 | - | - |  |

**CHAPTER 06 LangSmith API를 활용한 프롬프트 최적화**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | 평가용 테스트 데이터셋 구축하기 | - | - |  |
| 02 | LLM-as-a-judge로 평가하기 | - | - |  |
| 03 | 사용자 정의 평가자 만들기 | - | - |  |
| 04 | 휴리스틱 방식으로 평가하기 | - | - |  |
| 05 | 실험별 비교 분석하기 | - | - |  |
| 06 | 요약 평가자로 전체 수준 평가하기 | - | - |  |
| 07 | Groundedness 평가자로 할루시네이션 확인하기 | - | - |  |
| 08 | Pairwise 평가로 실험 비교 분석하기 | - | - |  |
| 09 | 반복 평가하기 | - | - |  |
| 10 | 온라인 LLM 평가자를 활용한 평가 자동화하기 | - | - |  |
| 11 | RAGAS를 활용해 RAG 평가하기 | - | - |  |

### PART 04 에이전트


**CHAPTER 07 도구와 툴킷**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | LangChain에서 제공하는 도구 | - | - |  |
| 02 | 사용자 정의 도구 만들기 | - | - |  |

**CHAPTER 08 에이전트 주요 기능**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | LLM에 도구 바인딩하기 | - | - |  |
| 02 | 에이전트와 AgentExecutor 생성하기 | - | - |  |
| 03 | 에이전트의 중간 단계 스트리밍 출력하기 | - | - |  |
| 04 | 이전 대화 내용을 기억하는 에이전트 만들기 | - | - |  |
| 05 | 다양한 LLM을 활용해 에이전트 생성하기 | - | - |  |
| 06 | iter() 함수로 단계별 출력하고 확인하기 | - | - |  |

**CHAPTER 09 에이전트 활용**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | Agentic RAG | - | - |  |
| 02 | CSV, 엑셀 데이터 분석 에이전트 | - | - |  |
| 03 | 파일 관리 업무 자동화 에이전트 | - | - |  |
| 04 | 보고서 작성 업무 자동화 에이전트 | - | - |  |
| 05 | CSV 파일 기반 데이터 분석 에이전트 | - | - |  |

### PART 05 서비스 배포


**CHAPTER 10 Streamlit을 이용한 서비스 배포 실습**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | 배포를 위한 config 설정하기 | - | - |  |
| 02 | 배포를 위한 GitHub와 기타 설정 파일 준비하기 | - | - |  |
| 03 | 배포 및 동작 테스트하기 | - | - |  |
| 04 | 서비스 수정 및 변경 사항 확인하기 | - | - |  |

**CHAPTER 11 LangServe**

| 절 | 내용 | 코드 | 정리 문서 | 비고 |
|---|---|---|---|---|
| 01 | LangServe로 모델을 서빙하기 | - | - |  |
| 02 | NGROK을 활용해서 모델을 외부에 공개하기 | - | - |  |
