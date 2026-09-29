# Cohere Reranker

Cohere의 호스팅 재정렬(Rerank) API를 이용해, FAISS로 1차 검색한 후보 문서들을 질의와의 관련도 기준으로 다시 정렬하는 법을 실습한다. 로컬 Cross Encoder 모델을 직접 돌리는 방식과 달리, API 호출만으로 재정렬을 처리한다.

## 1. 환경 변수 로드 및 LangSmith 프로젝트 설정

`.env`의 API 키를 불러오고, LangSmith에서 이 실습의 추적 로그를 구분해서 볼 수 있도록 프로젝트 이름을 지정한다.

```python
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "CH11-Reranker"
```

## 2. 결과 출력 헬퍼 함수

검색된 문서들을 번호와 함께 보기 좋게 출력하는 함수다. (참고: 원래 `f"Document {i*1}:"`로 되어 있었는데, `i*1`은 그냥 `i`와 같은 값이라 0번부터 번호가 매겨지는 오타였다. 다른 실습들과 동일하게 1번부터 세도록 `i+1`로 고쳤다.)

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

## 3. 1차 검색: FAISS + Cohere 임베딩

Cohere의 다국어 임베딩 모델(`embed-multilingual-v3.0`)로 FAISS retriever를 구성하고, `k=10`으로 넉넉하게 후보를 가져온다.

```python
from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_cohere import CohereEmbeddings 

documents = TextLoader("./data/appendix-keywords.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

texts = text_splitter.split_documents(documents)

retriever = FAISS.from_documents(
    texts, CohereEmbeddings(model="embed-multilingual-v3.0"),
).as_retriever(search_kwargs={"k": 10})

query = "Word2Vec에 대하여 알려줘!"

docs = retriever.invoke(query) 

pretty_print_docs(docs)
```

출력:

```txt
C:\Users\user\AppData\Local\Temp\ipykernel_15992\55041316.py:1: DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained. See https://github.com/langchain-ai/langchain-community/issues/674 for details and migration guidance toward standalone integration packages.
  from langchain_community.document_loaders import TextLoader

Document 1:

Crawling

정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.
예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.
연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진

Word2Vec

정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
-------------------------------
... (일부 생략)
```

## 4. Cohere Rerank로 재정렬

Cohere가 호스팅하는 다국어 재정렬 모델(`rerank-multilingual-v3.0`)을 API로 호출해서, 1차로 가져온 후보 문서들을 질의와의 관련도 기준으로 다시 정렬한다. `CrossEncoderReranker`가 로컬에서 모델을 직접 돌리는 방식이라면, `CohereRerank`는 별도 모델을 내려받지 않고 API 호출만으로 같은 역할(정밀 재정렬)을 한다는 차이가 있다.

```python
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever 
from langchain_cohere import CohereRerank

compressor = CohereRerank(model="rerank-multilingual-v3.0")

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("Word2Vec에 대하여 알려줘!")

pretty_print_docs(compressed_docs)
```

출력:

```txt
Document 1:

Crawling

정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.
예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.
연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진

Word2Vec

정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
----------------------------------------------------------------------------------------------------
Document 2:

정의: LLM은 대규모의 텍스트 데이터로 훈련된 큰 규모의 언어 모델을 의미합니다. 이러한 모델은 다양한 자연어 이해 및 생성 작업에 사용됩니다.
예시: OpenAI의 GPT 시리즈는 대표적인 대규모 언어 모델입니다.
연관키워드: 자연어 처리, 딥러닝, 텍스트 생성

FAISS (Facebook AI Similarity Search)

정의: FAISS는 페이스북에서 개발한 고속 유사성 검색 라이브러리로, 특히 대규모 벡터 집합에서 유사 벡터를 효과적으로 검색할 수 있도록 설계되었습니다.
예시: 수백
... (일부 생략)
```

## 정리

| 항목 | 내용 |
|---|---|
| 1차 검색 | FAISS + `CohereEmbeddings(model="embed-multilingual-v3.0")`, `k=10` |
| 재정렬 | `CohereRerank(model="rerank-multilingual-v3.0")` (Cohere 호스팅 API) |
| 구조 | `ContextualCompressionRetriever(base_compressor=CohereRerank(...), base_retriever=...)` |

앞서 실습한 `CrossEncoderReranker`(로컬 모델 직접 실행)와 비교하면, `CohereRerank`는 모델을 내려받아 실행할 필요 없이 API 호출만으로 동일한 역할(1차 후보 재정렬)을 한다는 점이 다르다. 로컬 GPU/모델 관리 없이 빠르게 쓸 수 있는 대신, API 비용과 네트워크 호출이 발생한다.
