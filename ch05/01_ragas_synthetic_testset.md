# RAGAS로 합성 테스트 데이터셋 생성하기

RAG 시스템을 평가하려면 "질문 – 정답 – 정답의 근거 문서"로 된 테스트 데이터셋이 필요하다. 사람이 직접 만들기엔 시간이 많이 들기 때문에, **RAGAS**의 `TestsetGenerator`로 PDF 문서에서 질문과 정답을 LLM이 자동으로 만들게 한다(합성 데이터셋, synthetic dataset). 만든 데이터셋은 CSV로 저장해서 이후 RAG 평가에 사용한다.

- 사용 버전: LangChain 1.4, RAGAS 0.4

## 1. 버전 확인

RAGAS는 버전에 따라 API가 크게 달라서, 먼저 설치된 버전을 확인한다.

```python
import langchain 
import ragas 

# RAGAS는 버전별로 API 차이가 커서 버전부터 확인
print(f"LangChain Version: {langchain.__version__}")
print(f"Ragas Version: {ragas.__version__}")
```

출력:

```txt
LangChain Version: 1.4.2
Ragas Version: 0.4.3
```

## 2. 환경변수 로드 및 LangSmith 프로젝트 설정

```python
from dotenv import load_dotenv

# .env의 OPENAI_API_KEY, LANGSMITH_API_KEY 등을 불러온다.
load_dotenv()
```

출력:

```txt
True
```

```python
import os

# LangSmith에서 이번 실습의 실행 기록을 모아볼 프로젝트 이름
# (LANGSMITH_TRACING, LANGSMITH_API_KEY는 .env에서 읽음)
os.environ["LANGSMITH_PROJECT"] = "CH16-Evaluations"
```

## 3. 문서 전처리

소프트웨어정책연구소(SPRi) AI Brief 2023년 12월호를 `PDFPlumberLoader`로 로드한다. 앞쪽 목차 페이지와 마지막 페이지는 질문을 만들 내용이 없으므로 빼고 19페이지만 사용한다.

```python
from langchain_community.document_loaders import PDFPlumberLoader

# 문서 로더 생성
loader = PDFPlumberLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")

# 문서 로딩 (페이지 1개 = Document 1개)
docs = loader.load()

# 목차(앞 3페이지)와 끝 페이지 제외
docs = docs[3:-1]

# 문서의 페이지수
len(docs)
```

출력:

```txt
19
```

각 페이지의 metadata에는 파일 경로, 페이지 번호, 작성 프로그램 등의 정보가 들어 있다.

```python
# 첫 번째 페이지의 metadata 확인
docs[0].metadata
```

출력:

```txt
{'source': 'data/SPRI_AI_Brief_2023년12월호_F.pdf',
 'file_path': 'data/SPRI_AI_Brief_2023년12월호_F.pdf',
 'page': 3,
 'total_pages': 23,
 'Author': 'dj',
 'Creator': 'Hwp 2018 10.0.0.13462',
 'Producer': 'Hancom PDF 1.3.0.542',
 'CreationDate': "D:20231208132838+09'00'",
 'ModDate': "D:20231208132838+09'00'",
 'PDFVersion': '1.4'}
```

## 4. 데이터셋 생성기 준비

질문·정답을 만들 LLM과 문서를 임베딩할 모델을 준비한다. RAGAS의 `llm_factory`에 OpenAI 클라이언트를 직접 넘긴다.

- `AsyncOpenAI`: RAGAS 내부는 비동기(async)로 동작하므로 async 클라이언트를 써야 LLM 호출이 병렬로 실행된다. 동기 `OpenAI()`를 넘기면 호출이 하나씩 순서대로 실행되어 느리고, 임베딩 단계에서 `Using sync embedding model ...` 경고가 뜬다.
- `wrap_openai`: `llm_factory`는 LangChain을 거치지 않고 OpenAI SDK를 직접 호출하기 때문에, 클라이언트를 `wrap_openai`로 감싸야 LangSmith에 호출 기록이 남는다.

```python
from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

# LangSmith 추적이 되도록 감싼 async OpenAI 클라이언트
openai_client = wrap_openai(AsyncOpenAI())

# 질문과 정답을 생성할 LLM
generator_llm = llm_factory("gpt-4o-mini", client=openai_client)

# 문서를 임베딩할 모델 (문서 간 유사도로 지식 그래프를 만들 때 사용)
generator_embeddings = RagasOpenAIEmbeddings(
    client=openai_client, model="text-embedding-3-small"
)
```

`TestsetGenerator`에 LLM과 임베딩 모델을 넘겨 생성기를 만든다. 생성 과정에서 문서의 제목·요약·주제·개체명을 추출하고 문서 간 관계를 이은 **지식 그래프(knowledge graph)** 를 자동으로 만든 뒤, 그 그래프를 바탕으로 질문을 만든다.

```python
from ragas.testset import TestsetGenerator

# 데이터셋 생성기
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
```

## 5. 질문 유형별 분포 정하기

RAGAS는 질문을 만드는 방식(synthesizer)을 여러 개 제공한다.

| synthesizer | 질문 형태 |
|---|---|
| `SingleHopSpecificQuerySynthesizer` | 청크 하나에서 구체적인 사실을 묻는 쉬운 질문 |
| `MultiHopAbstractQuerySynthesizer` | 여러 청크를 종합·해석해야 답할 수 있는 추상적인 질문 |
| `MultiHopSpecificQuerySynthesizer` | 여러 청크의 구체적인 사실을 엮어야 하는 질문 |

쉬운 질문 40%, 복잡한 질문 60%가 되도록 0.4 / 0.3 / 0.3으로 나눴다.

- `adapt_prompts("korean")`: 질문 생성 프롬프트를 한국어로 바꿔서 한국어 질문이 나오게 한다. 기본값은 few-shot 예시만 번역하는데, 그러면 가끔 영어 질문이 섞여 나와서 `adapt_instruction=True`로 지시문까지 번역했다.
- 노트북에서는 셀에서 바로 `await`를 쓸 수 있어서 비동기 함수인 `adapt_prompts()`를 그대로 호출했다.

```python
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

# 질문 유형별 분포 결정 (쉬운 질문 40%, 복잡한 질문 60%)
# 유형별 개수는 ceil(testset_size * 비율)이라 10 * 0.15 = 1.5처럼 소수가 나오면 올림되어 총 개수가 늘어남
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.4),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.3),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.3),
]

# 각 synthesizer의 프롬프트를 한국어로 바꾼다.
# adapt_instruction=True: few-shot 예시뿐 아니라 지시문도 한국어로 번역
for synthesizer, _ in query_distribution:
    adapted_prompts = await synthesizer.adapt_prompts(
        "korean", llm=generator_llm, adapt_instruction=True
    )
    synthesizer.set_prompts(**adapted_prompts)
```

## 6. 테스트셋 생성

`generate_with_langchain_docs()`에 LangChain `Document` 리스트를 그대로 넘긴다. 출력되는 진행 표시줄을 보면 내부 과정을 알 수 있다.

1. 문서 변환: `HeadlinesExtractor`(제목 추출) → `HeadlineSplitter`(제목 기준 분할, 19페이지 → 44개 노드) → `SummaryExtractor`(요약) → `EmbeddingExtractor`(임베딩) → `ThemesExtractor`(주제) → `NERExtractor`(개체명)
2. 관계 연결: `CosineSimilarityBuilder`, `OverlapScoreBuilder`로 노드 사이 관계를 이어 지식 그래프 완성
3. 질문 생성: 페르소나(질문하는 사람 유형) → 시나리오 → 샘플 10개 생성

```python
# 테스트셋 생성
# testset_size: 생성할 질문의 수, query_distribution: 질문 유형별 분포
# with_debugging_logs=True: 생성 과정 로그 출력
# raise_exceptions=True: 샘플 생성이 실패하면 조용히 빠지지 않고 에러를 냄
testset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=10,
    query_distribution=query_distribution,
    with_debugging_logs=True,
    raise_exceptions=True,
)
```

## 7. 생성된 데이터셋 확인

`to_pandas()`로 DataFrame으로 바꿔서 확인한다. 주요 컬럼은 다음과 같다.

| 컬럼 | 내용 |
|---|---|
| `user_input` | 생성된 질문 |
| `reference_contexts` | 정답의 근거가 된 문서 조각 |
| `reference` | 정답 |
| `persona_name`, `query_style`, `query_length` | 질문을 만든 페르소나와 질문의 말투·길이 |
| `synthesizer_name` | 어떤 synthesizer로 만든 질문인지 |

```python
# 생성된 테스트셋을 DataFrame으로 변환
test_df = testset.to_pandas()
test_df
```

출력:

```txt
                                          user_input  \
0                               미국에서 AI 안전을 위해 뭐하나요?   
1                          E.O. 14110의 주요 목적은 무엇인가요?   
2   이탈리아는 G7의 일원으로서 첨단 AI 시스템의 위험 관리에 어떤 역할을 하고 있나요?   
3                          AI 국제 행동강령의 주요 내용은 무엇인가요?   
4  영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전 보장을 위해 어떤 ...   
5  영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전성을 보장하기 위한 ...   
6  영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전 보장을 위해 어떤 ...   
7  오픈AI의 GPT-4와 GPT-3.5 터보의 성능 비교는 어떻게 되며, 앤스로픽과 ...   
8  2023년 10월 24일에 발표된 연구에 따르면 AI 기술이 근로자의 임금에 미치는...   
9  아마존이 앤스로픽에 대한 투자 계획은 무엇이며, 구글과의 협력은 어떻게 이루어지고 ...   

                                  reference_contexts  \
0  [KEY Contents n 미국 바이든 대통령이 ‘안전하고 신뢰할 수 있는 AI ...   
1  [1.  정책/법제 2.  기업/산업 3.  기술/연구 4.  인력/교육\n미국, ...   
2  [G7, 첨단 AI 시스템의 위험 관리를 위한 국제 행동강령 마련 n 주요 7개국(...   
3  [SPRi AI Brief |\n2023-12월호\n G7, 히로시마 AI 프로세스...   
4  [<1-hop>\n\n1. 정책/법제 2. 기업/산업 3. 기술/연구 4. 인력/교...   
5  [<1-hop>\n\n1. 정책/법제 2. 기업/산업 3. 기술/연구 4. 인력/교...   
6  [<1-hop>\n\n1. 정책/법제 2. 기업/산업 3. 기술/연구 4. 인력/교...   
...
```

```python
# 앞 5개 행만 확인
test_df.head()
```

출력:

```txt
                                          user_input  \
0                               미국에서 AI 안전을 위해 뭐하나요?   
1                          E.O. 14110의 주요 목적은 무엇인가요?   
2   이탈리아는 G7의 일원으로서 첨단 AI 시스템의 위험 관리에 어떤 역할을 하고 있나요?   
3                          AI 국제 행동강령의 주요 내용은 무엇인가요?   
4  영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전 보장을 위해 어떤 ...   

                                  reference_contexts  \
0  [KEY Contents n 미국 바이든 대통령이 ‘안전하고 신뢰할 수 있는 AI ...   
1  [1.  정책/법제 2.  기업/산업 3.  기술/연구 4.  인력/교육\n미국, ...   
2  [G7, 첨단 AI 시스템의 위험 관리를 위한 국제 행동강령 마련 n 주요 7개국(...   
3  [SPRi AI Brief |\n2023-12월호\n G7, 히로시마 AI 프로세스...   
4  [<1-hop>\n\n1. 정책/법제 2. 기업/산업 3. 기술/연구 4. 인력/교...   

                                           reference          persona_name  \
0  미국은 바이든 대통령이 안전하고 신뢰할 수 있는 AI 개발과 사용을 보장하기 위한 ...  AI Safety Researcher   
1  E.O. 14110은 미국 전역의 AI 연구를 촉진하고, 중소기업과 개발자에게 기술...     AI Policy Analyst   
2  이탈리아는 G7의 일원으로, 2023년 10월 30일에 합의된 AI 국제 행동강령을...  AI Safety Researcher   
3  AI 국제 행동강령은 G7이 첨단 AI 시스템을 개발하는 기업을 대상으로 자발적인 ...  AI Safety Researcher   
4  영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전 보장을 위해 국가,...                   NaN   
...
```

## 8. CSV로 저장

만든 데이터셋을 `data/ragas_synthetic_dataset.csv`로 저장해두고, 다음 실습(RAG 평가)에서 불러와 사용한다.

```python
# index=False: DataFrame의 행 번호는 저장하지 않음
test_df.to_csv("data/ragas_synthetic_dataset.csv", index=False)
```

## 생성된 질문 10개

`data/ragas_synthetic_dataset.csv`에 저장된 질문과 질문 유형이다.

| # | 질문 (`user_input`) | 유형 |
|---|---|---|
| 1 | 미국에서 AI 안전을 위해 뭐하나요? | Single-hop Specific |
| 2 | E.O. 14110의 주요 목적은 무엇인가요? | Single-hop Specific |
| 3 | 이탈리아는 G7의 일원으로서 첨단 AI 시스템의 위험 관리에 어떤 역할을 하고 있나요? | Single-hop Specific |
| 4 | AI 국제 행동강령의 주요 내용은 무엇인가요? | Single-hop Specific |
| 5 | 영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전 보장을 위해 어떤 정책과 법제를 강조했나요? | Multi-hop Abstract |
| 6 | 영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전성을 보장하기 위한 어떤 안전 테스트 계획을 포함하고 있나요? | Multi-hop Abstract |
| 7 | 영국 AI 안전성 정상회의에서 발표된 블레츨리 선언은 AI 안전 보장을 위해 어떤 정책과 법제를 강조했나요? | Multi-hop Abstract |
| 8 | 오픈AI의 GPT-4와 GPT-3.5 터보의 성능 비교는 어떻게 되며, 앤스로픽과 구글의 협력은 AI 안전성에 어떤 영향을 미칠까요? | Multi-hop Specific |
| 9 | 2023년 10월 24일에 발표된 연구에 따르면 AI 기술이 근로자의 임금에 미치는 영향은 어떤가요? | Multi-hop Specific |
| 10 | 아마존이 앤스로픽에 대한 투자 계획은 무엇이며, 구글과의 협력은 어떻게 이루어지고 있나요? | Multi-hop Specific |

5번과 7번은 같은 질문이 두 번 생성됐다. LLM이 만드는 합성 데이터라 이런 중복이나 어색한 질문이 섞일 수 있으므로, 평가에 쓰기 전에 한 번 훑어보고 중복을 지우거나 다시 생성하는 게 좋다.

## 정리

| 단계 | 사용한 도구 |
|---|---|
| 문서 로드 | `PDFPlumberLoader` |
| 생성기 LLM·임베딩 | `llm_factory(..., client=wrap_openai(AsyncOpenAI()))`, `ragas.embeddings.OpenAIEmbeddings` |
| 생성기 | `TestsetGenerator(llm=..., embedding_model=...)` |
| 질문 유형 분포 | `query_distribution = [(Synthesizer, 비율), ...]` |
| 한국어 질문 | `synthesizer.adapt_prompts("korean", adapt_instruction=True)` + `set_prompts()` |
| 생성 | `generate_with_langchain_docs(documents, testset_size, query_distribution)` |
| 저장 | `testset.to_pandas().to_csv(...)` |
