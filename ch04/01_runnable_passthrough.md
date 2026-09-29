# RunnablePassthrough 실습

`RunnablePassthrough`는 **입력을 아무것도 바꾸지 않고 그대로 다음 단계로 넘겨주는** Runnable이다. 단독으로는 하는 일이 없어 보이지만, 체인 중간에서 "원래 입력을 유지한 채 다른 값을 함께 만들어야 할 때" 꼭 필요하다. 입력을 그대로 넘기는 방법, `assign()`으로 새 키를 덧붙이는 방법, 그리고 RAG 체인에서 질문을 그대로 전달하는 방법을 실습한다.

## 1. 환경변수 로드

```python
from dotenv import load_dotenv 
import os

# .env의 API 키를 환경변수로 불러온다.
load_dotenv()
# LangSmith 추적 프로젝트 이름
os.environ["LANGSMITH_PROJECT"] = "RunnablePassthrough"
```

## 2. RunnableParallel 안에서 세 가지 방식 비교

`RunnableParallel`은 같은 입력을 여러 Runnable에 동시에 넣고, 결과를 키별로 모은 딕셔너리를 돌려준다. 입력 `{"num": 1}`에 대해 세 키가 각각 다르게 동작한다.

| 키 | 사용한 것 | 결과 |
|---|---|---|
| `passed` | `RunnablePassthrough()` | 입력을 그대로 전달 → `{'num': 1}` |
| `extra` | `RunnablePassthrough.assign(mult=...)` | 입력을 유지하면서 `mult` 키를 추가 → `{'num': 1, 'mult': 3}` |
| `modified` | 일반 함수(lambda) | 함수의 반환값으로 대체 → `2` |

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough 

runnable = RunnableParallel(
    # 입력을 그대로 전달
    passed=RunnablePassthrough(),
    # 입력은 그대로 두고, 입력으로 계산한 mult 키를 추가
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    # lambda는 자동으로 RunnableLambda가 되어 반환값(num + 1)만 남는다.
    modified=lambda x: x["num"] + 1
)

runnable.invoke({"num": 1})
```

출력:

```txt
{'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}
```

## 3. RunnablePassthrough.assign() 단독으로 쓰기

`assign()`은 입력 딕셔너리의 기존 키를 모두 유지하면서 새 키를 추가한다. 체인 중간에서 "지금까지의 값은 그대로 두고 계산 결과 하나만 더 붙이고 싶을 때" 쓴다.

```python
# 입력 {"num": 1}에 mult = num * 3 을 추가 -> {"num": 1, "mult": 3}
r = RunnablePassthrough.assign(mult=lambda x: x["num"] * 3)
r.invoke({"num": 1})
```

출력:

```txt
{'num': 1, 'mult': 3}
```

## 4. RAG 체인에서 RunnablePassthrough 활용

검색 체인에 질문 문자열 하나만 넣으면, 딕셔너리의 두 키가 같은 입력을 받아서

- `"context"`: retriever가 질문으로 문서를 검색하고, `format_docs`로 문자열로 합친다.
- `"question"`: `RunnablePassthrough()`가 질문을 그대로 넘긴다.

그 결과 `{"context": ..., "question": ...}`가 만들어져 프롬프트의 두 변수를 채운다. 체인 맨 앞의 딕셔너리는 자동으로 `RunnableParallel`로 바뀐다.

```python
from langchain_community.vectorstores import FAISS 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 

# 예시 문장 4개로 FAISS 벡터 스토어 생성
vectorstore = FAISS.from_texts(
    [
        "허사비스는 랭체인 주식회사에서 근무를 하였습니다.",
        "가우스는 허사비스와 같은 회사에서 근무하였습니다.",
        "허사비스의 직업은 개발자입니다.",
        "가우스의 직업은 디자이너입니다.",
    ],
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()
# 검색된 문서(context)만 근거로 답하도록 하는 프롬프트
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4o-mini")

# 검색된 Document 리스트를 줄바꿈으로 이어 하나의 문자열로 만든다.
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

retrieval_chain = (
    # 질문 하나로 context(검색 결과 문자열)와 question(질문 그대로)을 동시에 만든다.
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt 
    | model 
    | StrOutputParser()
)
```

## 5. 체인 실행

질문 문자열만 넣으면 검색 → 프롬프트 → 모델 → 문자열 파싱이 한 번에 실행되고, 검색된 문장을 근거로 답한다.

```python
# 질문 문자열 하나만 넣으면 된다. (RunnablePassthrough가 question 자리에 그대로 전달)
retrieval_chain.invoke("허사비스의 직업은 무엇입니까?")
```

출력:

```txt
'허사비스의 직업은 개발자입니다.'
```

```python
retrieval_chain.invoke("가우스의 직업은 무엇입니까?")
```

출력:

```txt
'가우스의 직업은 디자이너입니다.'
```

## 정리

| 사용법 | 동작 | 언제 쓰나 |
|---|---|---|
| `RunnablePassthrough()` | 입력을 그대로 전달 | RAG 체인에서 질문을 프롬프트까지 그대로 넘길 때 |
| `RunnablePassthrough.assign(key=함수)` | 입력 딕셔너리를 유지하고 새 키를 추가 | 기존 값은 두고 계산 결과를 덧붙일 때 |
| `RunnableParallel(...)` / 체인 안의 딕셔너리 | 같은 입력을 여러 Runnable에 동시에 넣고 결과를 키별로 모음 | 프롬프트 변수 여러 개를 한 번에 만들 때 |
