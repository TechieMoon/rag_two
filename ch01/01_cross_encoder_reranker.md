# Cross Encoder Reranker

임베딩 기반 검색(FAISS)으로 후보 문서를 넉넉히 가져온 뒤, Cross Encoder 모델로 질의-문서 쌍의 관련도를 직접 계산해서 다시 정밀하게 순위를 매기는 2단계(retrieve → rerank) 검색 패턴을 실습한다.

## 1. 결과 출력 헬퍼 함수

검색된 문서들을 구분선과 함께 출력하는 함수를 정의한다.

```python
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

## 2. 1차 검색: FAISS + HuggingFace 임베딩

HuggingFace의 `sentence-transformers/msmarco-distilbert-dot-v5` 임베딩 모델로 FAISS retriever를 만들고, `k=10`으로 넉넉하게 후보 문서를 가져온다. 임베딩 기반 검색은 빠르지만, 질의와 문서를 각각 독립적으로 벡터화한 뒤 비교하기 때문에 정교함에는 한계가 있다.

```python
from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter 

documents = TextLoader("./data/appendix-keywords.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

texts = text_splitter.split_documents(documents)

embeddingsModel = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-dot-v5"
)

retriever = FAISS.from_documents(texts, embeddingsModel).as_retriever(
    search_kwargs={"k": 10}
)


query = "Word2Vec에 대하여 알려줄래?"

docs = retriever.invoke(query)

pretty_print_docs(docs)
```

출력:

```txt
Loading weights: 100%|██████████| 100/100 [00:00<?, ?it/s]

Document 1:

Open Source

정의: 오픈 소스는 소스 코드가 공개되어 누구나 자유롭게 사용, 수정, 배포할 수 있는 소프트웨어를 의미합니다. 이는 협업과 혁신을 촉진하는 데 중요한 역할을 합니다.
예시: 리눅스 운영 체제는 대표적인 오픈 소스 프로젝트입니다.
연관키워드: 소프트웨어 개발, 커뮤니티, 기술 협업

Structured Data

정의: 구조화된 데이터는 정해진 형식이나 스키마에 따라 조직된 데이터입니다. 이는 데이터베이스, 스프레드시트 등에서 쉽게 검색하고 분석할 수 있습니다.
예시: 관계형 데이터베이스에 저장된 고객 정보 테이블은 구조화된 데이터의 예입니다.
연관키워드: 데이터베이스, 데이터 분석, 데이터 모델링

Parser
----------------------------------------------------------------------------------------------------
Document 2:

정의: LLM은 대규모의 텍스트 데이터로 훈련된 큰 규모의 언어 모델을 의미합니다. 이러한 모델은 다양한 자연어 이해 및 생성 작업에 사용됩니다.
예시: OpenAI의 GPT 시리즈는 대표적인 대규모 언어 모델입니다.
연관키워드: 자연어 처리, 딥러닝, 텍스트 생성

FAISS (Facebook AI Similarity Search)

정의: FAISS는 페이스북에서 개발한 고속 유사성 검색 라이브러리로, 특히 대규모 벡터 집합에서 유사 벡터를 효과
... (일부 생략)
```

## 3. Cross Encoder로 재정렬(Rerank)

Cross Encoder는 질의와 문서를 각각 따로 임베딩하는 대신, 질의-문서 쌍을 통째로 모델에 입력해서 "이 둘이 얼마나 관련 있는가"를 직접 점수로 계산한다. 그만큼 느리지만 훨씬 정교해서, 1차로 넉넉히 가져온 후보들(k=10)을 2차로 다시 정밀하게 순위를 매기는 재정렬(Reranking) 단계에 적합하다. `CrossEncoderReranker(model=HuggingFaceCrossEncoder("BAAI/bge-reranker-v2-m3"), top_n=3)`으로 10개 후보 중 가장 관련도 높은 3개만 추려낸다.

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder 

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

compressor = CrossEncoderReranker(model=model, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke("Word2Vec에 대하여 알려줄래?")

pretty_print_docs(compressed_docs)
```

출력:

```txt
d:\moon0902\hanwha_0902\ex0923\.venv\Lib\site-packages\huggingface_hub\file_download.py:150: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\user\.cache\huggingface\hub\models--BAAI--bge-reranker-v2-m3. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/
... (일부 생략)
```

## 정리

| 단계 | 방식 | 특징 |
|---|---|---|
| 1차 검색 (Retrieve) | FAISS + `HuggingFaceEmbeddings` | 질의·문서를 각각 독립적으로 임베딩 후 비교. 빠르지만 정교함에 한계, 넉넉한 k로 후보 확보 |
| 2차 재정렬 (Rerank) | `CrossEncoderReranker` + `HuggingFaceCrossEncoder` | 질의-문서 쌍을 함께 모델에 입력해 관련도를 직접 계산. 느리지만 정교함, 적은 후보에만 적용 |

`ContextualCompressionRetriever(base_compressor=CrossEncoderReranker(...), base_retriever=...)` 구조는 앞서 본 `LLMChainExtractor`, `EmbeddingsFilter`와 동일한 `ContextualCompressionRetriever` 틀 위에서, 압축기 대신 재정렬기를 끼워 넣은 것이다. "넉넉히 가져와서(retrieve) 정밀하게 다시 추린다(rerank)"는 것이 핵심 아이디어다.
