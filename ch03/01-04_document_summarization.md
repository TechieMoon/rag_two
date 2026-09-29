# 문서 요약 (Stuff / Map-Reduce / Map-Refine / Chain of Density)

긴 문서를 LLM으로 요약하는 네 가지 방식을 실습한다.

| 방식 | 아이디어 |
|---|---|
| Stuff | 문서 전체를 프롬프트 하나에 통째로 넣어서 한 번에 요약 |
| Map-Reduce | 문서를 나눠서 각각 요약(Map)한 뒤, 요약들을 합쳐 최종 요약(Reduce) |
| Map-Refine | 나눠서 각각 요약(Map)한 뒤, 앞의 요약에 다음 요약을 하나씩 반영하며 다듬기(Refine) |
| Chain of Density (CoD) | 요약을 여러 번 반복하면서 매번 빠진 핵심 개체(entity)를 추가해 점점 밀도 높은 요약으로 만들기 |

프롬프트는 LangSmith Hub에 공개된 `teddynote/...` 프롬프트를 불러와서 사용한다.

## 1. 환경변수 로드

```python
from dotenv import load_dotenv
import os 

# .env의 API 키를 환경변수로 불러온다.
load_dotenv()
# LangSmith 추적 프로젝트 이름
os.environ["LANGSMITH_PROJECT"] = "Summary"
```

## 2. Stuff: 문서 전체를 한 번에 요약

### 2-1. 요약할 뉴스 기사 로드

`TextLoader`로 `data/news.txt`(오픈 소스 LLM '올모' 출시 기사, 약 1,700자)를 불러온다. 짧은 문서라 통째로 프롬프트에 넣어도 컨텍스트 길이를 넘지 않는다.

```python
from langchain_community.document_loaders import TextLoader 

# 텍스트 파일 전체가 Document 1개로 로드된다.
loader = TextLoader("data/news.txt", encoding="utf-8")
docs = loader.load()
print(f"총 글자 수: {len(docs[0].page_content)}")
print("\n=========== 앞부분 미리보기 ===========\n")
print(docs[0].page_content[:500])
```

출력:

```txt
총 글자 수: 1708

=========== 앞부분 미리보기 ===========

제목: 
AI2, 상업 활용까지 자유로운 '진짜' 오픈 소스 LLM '올모' 출시

내용:
앨런AI연구소(AI2)가 완전한 오픈 소스 대형언어모델(LLM) '올모(OLMo)’를 출시했다. 데이터 수집, 학습, 배포의 전 과정을 투명하게 공개한 데다 상업적 사용까지 허용한 진정한 의미의 오픈 소스 LLM이라는 평가다.
벤처비트는 1일(현지시간) 비영리 민간 AI 연구기관인 AI2가 ‘최초의 진정한 오픈 소스 LLM 및 프레임워크’라고 소개한 ‘올모’를 출시했다고 보도했다. 
이에 따르면 올모는 모델 코드와 모델 가중치뿐만 아니라 훈련 코드, 훈련 데이터, 관련 툴킷 및 평가 툴킷도 제공한다. 이를 통해 모델이 어떻게 구축되었는지 심층적으로 분석, LLM의 작동 방식과 응답을 생성하는 원리를 더 잘 이해할 수 있다. 
올모 프레임워크는 70억 매개변수의 ‘올모 7B’ 등 4가지 변형 모델과 10억 매개변수의 ‘올모 1B’ 모델을 제공한다. 모델들은 훈련 데이터를 생성하는 코드를 포함해 
```

### 2-2. LangSmith Hub에서 요약 프롬프트 가져오기

`Client().pull_prompt()`로 Hub에 공개된 프롬프트를 불러온다. 다른 사람이 올린 공개 프롬프트를 가져올 때는 `dangerously_pull_public_prompt=True`를 명시해야 한다. 이 프롬프트는 "핵심을 한국어 bullet point로, 문장마다 알맞은 이모지를 붙여 요약하라"는 내용이다.

```python
from langsmith import Client 

client = Client()

# 공개 프롬프트는 dangerously_pull_public_prompt=True를 줘야 불러올 수 있다.
prompt = client.pull_prompt("teddynote/summary-stuff-documents-korean", dangerously_pull_public_prompt=True)
# 프롬프트 내용을 보기 좋게 출력
prompt.pretty_print()
```

출력:

```txt
Please summarize the sentence according to the following REQUEST.
REQUEST:
1. Summarize the main points in bullet points in KOREAN.
2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
3. Use various emojis to make the summary more interesting.
4. Translate the summary into KOREAN if it is written in ENGLISH.
5. DO NOT translate any technical terms.
6. DO NOT include any unnecessary information.

CONTEXT:
{context}

SUMMARY:"
```

### 2-3. Stuff 체인 실행

`format_docs()`로 Document들의 본문을 하나의 문자열로 이어 붙여 `{context}`에 넣고, 스트리밍으로 요약을 출력한다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 여러 Document의 본문을 빈 줄로 구분해서 하나의 문자열로 합친다.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


stuff_chain = prompt | llm | StrOutputParser()

answer = ""
# 문서 전체를 context에 통째로 넣고(Stuff) 스트리밍으로 받아서 출력
for chunk in stuff_chain.stream({"context": format_docs(docs)}):
    print(chunk, end="", flush=True)
    answer += chunk
```

출력:

```txt
- 🚀 앨런AI연구소(AI2)가 완전한 오픈 소스 LLM '올모(OLMo)'를 출시했다.  
- 📊 데이터 수집, 학습, 배포 과정을 투명하게 공개하고 상업적 사용을 허용한다.  
- 🛠️ 모델 코드, 가중치, 훈련 코드, 데이터 및 평가 툴킷을 제공한다.  
- 📈 '올모 7B'와 '올모 1B' 등 4가지 변형 모델을 포함한다.  
- 🔍 AI2의 '돌마(Dolma)' 데이터 세트를 기반으로 3조개의 토큰으로 훈련되었다.  
- 📜 아파치 2.0 라이선스에 따라 상업적 활용에 제한이 없다.  
- 🧪 연구자들이 모델의 작동 방식을 과학적으로 이해할 수 있도록 돕는다.  
- 🌟 올모는 상업용 제품과 동등한 성능을 보여준다.  
- ⚠️ 비영어권 언어에 대한 낮은 품질과 약한 코드 생성 기능이 있다.  
- 🔄 AI2는 올모를 계속해서 향상할 계획이다.  
- 🌐 올모의 모든 리소스는 깃허브 및 허깅페이스에서 무료로 제공된다.  
```

## 3. Map-Reduce: 나눠서 요약하고 합치기

### 3-1. PDF에서 5페이지만 가져오기

문서가 길면 Stuff 방식은 컨텍스트 길이를 넘거나 비용이 커진다. SPRi AI Brief PDF의 4~8페이지(`docs[3:8]`) 5페이지를 대상으로, 페이지마다 따로 요약한 뒤 합치는 방식을 실습한다.

```python
from langchain_community.document_loaders import PyPDFLoader 

loader = PyPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load() 
# 목차 등을 건너뛰고 본문 4~8페이지(인덱스 3~7)만 사용
docs = docs[3:8]
print(f"총 페이지 수: {len(docs)}")
```

출력:

```txt
총 페이지 수: 5
```

### 3-2. Map 단계: 문서별 핵심 내용 추출

`teddynote/map-prompt`는 문서에서 핵심 주장(main thesis)을 bullet point로 뽑아내는 프롬프트다.

```python
from langsmith import Client 

client = Client()
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser 

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
)

# Map 단계 프롬프트: 문서 하나에서 핵심 주장 목록을 추출
map_prompt = client.pull_prompt("teddynote/map-prompt", dangerously_pull_public_prompt=True)

map_prompt.pretty_print()
```

출력:

```txt
================================ System Message ================================

You are a professional main thesis extractor.

================================ Human Message =================================

Your task is to extract main thesis from given documents. Answer should be in same language as given document. 

#Format: 
- thesis 1
- thesis 2
- thesis 3
- ...

Here is a given document: 
{doc}

Write 1~5 sentences.
#Answer:
```

```python
# Map 체인: 문서 하나 -> 핵심 내용 요약 문자열
map_chain = map_prompt | llm | StrOutputParser()
```

`batch()`로 5페이지를 한 번에 넣으면 페이지별 요약이 병렬로 실행되어 5개의 요약 리스트가 나온다.

```python
# 페이지 5개를 병렬로 요약 -> 요약 문자열 5개가 담긴 리스트
doc_summaries = map_chain.batch(docs)
```

```python
# 입력한 페이지 수만큼 요약이 생성됐는지 확인
len(doc_summaries)
```

출력:

```txt
5
```

```python
# 첫 번째 페이지(미국 AI 행정명령)의 요약
print(doc_summaries[0])
```

출력:

```txt
- 미국 바이든 대통령이 안전하고 신뢰할 수 있는 AI 개발과 사용을 보장하기 위한 행정명령을 발표하였다.
- 이 행정명령은 AI의 안전과 보안 기준, 개인정보 보호, 형평성과 시민권 향상, 소비자 보호, 노동자 지원, 혁신과 경쟁 촉진, 국제협력을 포함한 광범위한 내용을 담고 있다.
- AI 시스템 개발 기업은 안전 테스트 결과와 시스템 정보를 미국 정부와 공유해야 하며, AI의 책임 있는 사용을 위한 모범사례와 지침이 마련될 예정이다.
- 또한, 중소기업과 개발자에게 기술과 인프라를 지원하고, AI 관련 분야의 전문 지식을 갖춘 외국인들이 미국에서 공부하고 취업할 수 있도록 비자 기준을 현대화할 계획이다.
```

### 3-3. Reduce 단계: 요약들을 하나로 합치기

`teddynote/reduce-prompt`는 여러 요약 목록을 받아서 하나의 최종 요약으로 정리하는 프롬프트다. `{language}`로 출력 언어를 지정할 수 있다.

```python
# Reduce 단계 프롬프트: 요약 목록 -> 최종 요약 하나
reduce_prompt = client.pull_prompt("teddynote/reduce-prompt", dangerously_pull_public_prompt=True)

reduce_prompt.pretty_print()
```

출력:

```txt
================================ System Message ================================

You are a professional summarizer. You are given a list of summaries of documents and you are asked to create a single summary of the documents.

================================ Human Message =================================

#Instructions: 
1. Extract main points from a list of summaries of documents
2. Make final summaries in bullet points format.
3. Answer should be written in {language}.

#Format: 
- summary 1
- summary 2
- summary 3
- ...

Here is a list of summaries of documents: 
{doc_summaries}

...
```

```python
reduce_chain = reduce_prompt | llm | StrOutputParser()
```

```python
answer = ""
# 5개 요약을 줄바꿈으로 이어 붙여 넣고, 한국어로 최종 요약을 스트리밍 출력
for chunk in reduce_chain.stream({"doc_summaries": "\n".join(doc_summaries), "language": "Korean"}):
    print(chunk, end="", flush=True)
    answer += chunk
```

출력:

```txt
- 미국 바이든 대통령이 AI 개발과 사용의 안전성을 보장하기 위한 행정명령을 발표하였다.
- 행정명령은 AI의 안전 기준, 개인정보 보호, 형평성, 소비자 보호, 노동자 지원, 혁신 촉진, 국제협력 등을 포함한다.
- AI 시스템 개발 기업은 안전 테스트 결과를 정부와 공유해야 하며, 책임 있는 사용을 위한 지침이 마련될 예정이다.
- 중소기업과 개발자에게 기술 지원을 제공하고, 외국인 AI 전문가의 비자 기준을 현대화할 계획이다.
- G7은 '히로시마 AI 프로세스'를 통해 AI 기업을 위한 국제 행동강령에 합의하였다.
- 행동강령은 AI의 위험 평가, 투명성 보장, 정보 공유 및 협력을 요구한다.
- AI 생성 콘텐츠의 신뢰성을 보장하기 위한 인증 및 출처 확인 메커니즘 개발이 제안되었다.
- 28개국이 AI 안전성 정상회의에서 AI 안전 보장을 위한 블레츨리 선언을 발표하였다.
- 선언은 AI 시스템의 안전을 위해 모든 이해관계자의 협력이 중요하다고 강조하였다.
- 영국 총리는 AI 안전 연구소 출범과 안전성 시험 계획 수립을 발표하였다.
- 한국은 영국 및 프랑스와 AI 정상회의를 개최할 예정이다.
- 미국 법원은 생성 AI 기업에 대한 저작권 침해 소송을 기각하였다.
- FTC는 생성 AI로 인한 소비자 피해 및 빅테크의 시장 지배력 우려를 표명하였다.
- FTC는 소비자 보호와 공정한 경쟁 시장 유지를 위해 모든 권한을 활용하겠다고 강조하였다.
```

### 3-4. Map-Reduce를 하나의 체인으로 묶기

`@chain` 데코레이터를 붙이면 일반 파이썬 함수가 Runnable이 되어 `invoke()`/`stream()`으로 호출할 수 있다. Map은 가볍고 빠른 `gpt-4o-mini`로, 최종 품질이 중요한 Reduce는 `gpt-4o`로 모델을 나눠 썼다. 함수 안에서 `yield from`으로 Reduce 결과를 흘려보내서 스트리밍이 된다.

```python
from langchain_core.runnables import chain 

# @chain: 함수를 Runnable로 만들어 stream()/invoke()로 호출 가능하게 한다.
@chain 
def map_reduce_chain(docs):
    # Map 단계용 LLM (문서 수만큼 호출되므로 가볍고 저렴한 모델)
    map_llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
    )

    map_prompt = client.pull_prompt("teddynote/map-prompt", dangerously_pull_public_prompt=True)

    map_chain = map_prompt | map_llm | StrOutputParser()

    # 1) Map: 문서별 요약을 병렬로 생성
    doc_summaries = map_chain.batch(docs)

    reduce_prompt = client.pull_prompt("teddynote/reduce-prompt", dangerously_pull_public_prompt=True)

    # Reduce 단계용 LLM (한 번만 호출되고 최종 결과 품질을 좌우하므로 더 좋은 모델)
    reduce_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    reduce_chain = reduce_prompt | reduce_llm | StrOutputParser()

    # 2) Reduce: 요약들을 합쳐 최종 요약을 스트리밍으로 흘려보낸다.
    yield from reduce_chain.stream(
        {"doc_summaries": "\n".join(doc_summaries), "language": "Korean"}
    )
```

```python
answer = ""
# 문서 리스트만 넣으면 Map -> Reduce가 한 번에 실행된다.
for chunk in map_reduce_chain.stream(docs):
    print(chunk, end="", flush=True)
    answer += chunk
```

출력:

```txt
- 미국 바이든 대통령은 AI의 안전하고 신뢰할 수 있는 개발과 사용을 위한 행정명령을 발표하였다.
- 행정명령은 AI의 안전, 개인정보 보호, 형평성, 소비자 보호, 노동자 지원, 혁신 촉진, 국제협력을 포함한다.
- AI 기업은 안전 테스트 결과와 시스템 정보를 미국 정부와 공유해야 하며, 책임 있는 사용 지침이 마련된다.
- 중소기업과 개발자에게 기술과 인프라 지원을 통해 AI 연구를 촉진하고, 외국인 전문가의 비자 기준을 현대화할 예정이다.
- G7은 '히로시마 AI 프로세스'를 통해 AI 기업을 위한 국제 행동강령에 합의하였다.
- 행동강령은 AI 시스템의 위험 식별과 완화를 위한 자발적 채택을 권장하며, 투명성과 책임성을 강조한다.
- G7은 AI 생성 콘텐츠의 신뢰성을 보장하기 위해 인증과 출처 확인 메커니즘 개발을 요구하고 있다.
- 28개국은 블레츨리 선언을 통해 AI 안전 보장을 위한 협력 방안을 발표하였다.
- 선언은 AI 시스템의 안전을 위해 모든 이해관계자의 협력이 중요하다고 강조하며, 첨단 AI 개발 기업의 책임을 지적했다.
- 영국은 AI 안전 연구소 출범과 첨단 AI 모델의 안전성 시험 계획을 발표하였다.
- 한국은 영국과 AI 미니 정상회의를 공동 개최하고, 프랑스와 대면 정상회의를 개최할 예정이다.
- 미국 법원은 예술가들이 제기한 생성 AI 기업에 대한 저작권 침해 소송을 기각하였다.
- FTC는 생성 AI로 인한 소비자와 창작자의 피해 가능성 및 빅테크의 시장 지배력 강화 우려를 표명했다.
- FTC는 생성 AI의 위험을 경고하며, 소비자 보호와 공정한 경쟁 시장 유지를 강조했다.
```

## 4. Map-Refine: 요약을 순서대로 다듬기

Map-Reduce는 요약들을 한 번에 합치지만, Map-Refine은 **첫 번째 요약에서 출발해서 다음 요약을 하나씩 반영하며 기존 요약을 계속 고쳐 나간다.** 문서 순서대로 누적되기 때문에 흐름이 자연스럽지만, 순차 실행이라 Map-Reduce보다 느리다.

### 4-1. Map 단계: 문서별 요약

`teddynote/map-summary-prompt`는 `{documents}`를 `{language}`로 요약하는 프롬프트다.

```python
from langsmith import Client 
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser 

map_llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
)

client = Client()
# Map-Refine의 Map 단계 프롬프트: 문서 하나를 지정한 언어로 bullet 요약
map_summary = client.pull_prompt("teddynote/map-summary-prompt", dangerously_pull_public_prompt=True)

map_summary.pretty_print()
```

출력:

```txt
================================ System Message ================================

You are an expert summarizer. Your task is to summarize the following document in {language}.

================================ Human Message =================================

Extract most important main thesis from the documents, then summarize in bullet points.

#Format:
- summary 1
- summary 2
- summary 3
-...

Here is a given document: 
{documents}

Write 1~5 sentences. Think step by step.
#Summary:
```

```python
# 앞에서 만든 llm(gpt-4o-mini)을 그대로 사용 (map_llm과 같은 설정)
map_chain = map_summary | llm | StrOutputParser()
```

```python
# 첫 페이지 하나만 먼저 요약해본다.
print(map_chain.invoke({"documents": docs[0], "language": "Korean"}))
```

출력:

```txt
- 미국 바이든 대통령이 안전하고 신뢰할 수 있는 AI 개발과 사용을 위한 행정명령을 발표하였다.
- 이 행정명령은 AI의 안전과 보안 기준, 개인정보 보호, 형평성과 시민권 향상, 소비자 보호, 노동자 지원, 혁신과 경쟁 촉진, 국제 협력을 포함한다.
- AI 시스템 개발 기업은 안전 테스트 결과와 시스템 정보를 정부와 공유해야 하며, AI의 책임 있는 사용을 위한 지침이 마련된다.
- 소비자 보호와 근로자 지원을 위해 의료 분야에서의 AI 사용 촉진 및 교육 도구 개발이 강조된다.
- 국가 AI 연구 자원을 통해 AI 연구를 촉진하고, 외국인 전문가들이 미국에서 공부하고 일할 수 있도록 지원하는 방안도 포함된다.
```

`batch()`에 넣으려면 문서마다 프롬프트 변수(`documents`, `language`)를 채운 딕셔너리를 만들어야 한다.

```python
# 문서마다 프롬프트 입력 딕셔너리를 만든다. (batch 입력용)
input_doc = [{"documents": doc, "language": "Korean"} for doc in docs]
```

```python
input_doc
```

출력:

```txt
[{'documents': Document(metadata={'producer': 'Hancom PDF 1.3.0.542', 'creator': 'Hwp 2018 10.0.0.13462', 'creationdate': '2023-12-08T13:28:38+09:00', 'author': 'dj', 'moddate': '2023-12-08T13:28:38+09:00', 'pdfversion': '1.4', 'source': 'data/SPRI_AI_Brief_2023년12월호_F.pdf', 'total_pages': 23, 'page ...
  'language': 'Korean'},
 {'documents': Document(metadata={'producer': 'Hancom PDF 1.3.0.542', 'creator': 'Hwp 2018 10.0.0.13462', 'creationdate': '2023-12-08T13:28:38+09:00', 'author': 'dj', 'moddate': '2023-12-08T13:28:38+09:00', 'pdfversion': '1.4', 'source': 'data/SPRI_AI_Brief_2023년12월호_F.pdf', 'total_pages': 23, 'page ...
  'language': 'Korean'},
 {'documents': Document(metadata={'producer': 'Hancom PDF 1.3.0.542', 'creator': 'Hwp 2018 10.0.0.13462', 'creationdate': '2023-12-08T13:28:38+09:00', 'author': 'dj', 'moddate': '2023-12-08T13:28:38+09:00', 'pdfversion': '1.4', 'source': 'data/SPRI_AI_Brief_2023년12월호_F.pdf', 'total_pages': 23, 'page ...
  'language': 'Korean'},
 {'documents': Document(metadata={'producer': 'Hancom PDF 1.3.0.542', 'creator': 'Hwp 2018 10.0.0.13462', 'creationdate': '2023-12-08T13:28:38+09:00', 'author': 'dj', 'moddate': '2023-12-08T13:28:38+09:00', 'pdfversion': '1.4', 'source': 'data/SPRI_AI_Brief_2023년12월호_F.pdf', 'total_pages': 23, 'page ...
  'language': 'Korean'},
 {'documents': Document(metadata={'producer': 'Hancom PDF 1.3.0.542', 'creator': 'Hwp 2018 10.0.0.13462', 'creationdate': '2023-12-08T13:28:38+09:00', 'author': 'dj', 'moddate': '2023-12-08T13:28:38+09:00', 'pdfversion': '1.4', 'source': 'data/SPRI_AI_Brief_2023년12월호_F.pdf', 'total_pages': 23, 'page ...
  'language': 'Korean'}]
```

```python
# 5페이지를 병렬로 요약
print(map_chain.batch(input_doc))
```

출력:

```txt
['- 미국 바이든 대통령이 안전하고 신뢰할 수 있는 AI 개발과 사용을 위한 행정명령을 발표하였다.\n- 이 행정명령은 AI의 안전과 보안 기준, 개인정보 보호, 형평성과 시민권 향상, 소비자 보호, 노동자 지원, 혁신과 경쟁 촉진, 국제 협력을 포함한다.\n- AI 시스템 개발 기업은 안전 테스트 결과와 시스템 정보를 정부와 공유해야 하며, AI의 책임 있는 사용을 위한 지침이 마련된다.\n- 소비자 보호와 근로자 지원을 위해 의료 분야에서의 AI 사용 촉진 및 교육 도구 개발이 강조된다.\n- 국가 AI 연구자원(NAIRR)을 ...
```

### 4-2. Refine 단계 프롬프트

`teddynote/refine-prompt`는 `{previous_summary}`(지금까지의 요약)와 `{current_summary}`(새 문서의 요약)를 받아서, 새 내용을 반영해 요약을 다듬는 프롬프트다.

```python
# Refine 단계 프롬프트: 기존 요약 + 새 요약 -> 다듬어진 요약
refine_prompt = client.pull_prompt("teddynote/refine-prompt", dangerously_pull_public_prompt=True)

refine_prompt.pretty_print()
```

출력:

```txt
================================ System Message ================================

You are an expert summarizer.

================================ Human Message =================================

Your job is to produce a final summary

We have provided an existing summary up to a certain point:
{previous_summary}

We have the opportunity to refine the existing summary(only if needed) with some more context below.
------------
{current_summary}
------------
Given the new context, refine the original summary in {language}.
If the context isn't useful, return the original summary.
```

```python
refine_llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
)

refine_chain = refine_prompt | refine_llm | StrOutputParser()
```

### 4-3. Map-Refine을 하나의 체인으로 묶기

1. Map: 모든 페이지를 병렬로 요약한다.
2. Refine: 첫 요약을 `previous_summary`로 두고, 나머지 요약을 순서대로 하나씩 반영해 `previous_summary`를 갱신한다.

Refine 과정이 한 단계씩 진행되는 걸 볼 수 있게 매 단계 결과를 스트리밍으로 출력한다.

```python
from langchain_core.runnables import chain 

@chain 
def map_refine_chain(docs):
    map_summary = client.pull_prompt("teddynote/map-summary-prompt", dangerously_pull_public_prompt=True)

    map_chain = (
        map_summary
        | ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        | StrOutputParser()
    )

    # 문서 본문(page_content)만 꺼내서 batch 입력을 만든다.
    input_doc = [{"documents": doc.page_content, "language": "Korean"} for doc in docs]

    # 1) Map: 페이지별 요약을 병렬로 생성
    doc_summaries = map_chain.batch(input_doc)

    refine_prompt = client.pull_prompt("teddynote/refine-prompt", dangerously_pull_public_prompt=True)

    refine_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    refine_chain = refine_prompt | refine_llm | StrOutputParser()

    # 2) Refine: 첫 번째 요약에서 출발
    previous_summary = doc_summaries[0]

    # 나머지 요약을 순서대로 하나씩 반영하며 요약을 다듬는다.
    for current_summary in doc_summaries[1:]:
        new_summary = ""
        for token in refine_chain.stream({
                "previous_summary": previous_summary,
                "current_summary": current_summary,
                "language": "Korean",
        }):
            print(token, end="", flush=True)
            new_summary += token
        # 다듬어진 요약이 다음 단계의 "기존 요약"이 된다.
        previous_summary = new_summary
        print("\n\n---------------------------\n\n")

    # 마지막까지 다듬어진 요약이 최종 결과
    return previous_summary
```

```python
# 단계별 Refine 결과가 출력되고, 최종 요약이 refined_summary에 저장된다.
refined_summary = map_refine_chain.invoke(docs)
```

출력:

```txt
- 미국 바이든 대통령이 안전하고 신뢰할 수 있는 AI 개발과 사용을 위한 행정명령을 발표하였다.
- 이 행정명령은 AI의 안전과 보안 기준, 개인정보 보호, 형평성과 시민권 향상, 소비자 보호, 노동자 지원, 혁신과 경쟁 촉진, 국제 협력을 포함한다.
- AI 시스템 개발 기업은 안전 테스트 결과와 시스템 정보를 정부와 공유해야 하며, AI의 책임 있는 사용을 위한 모범사례와 지침이 마련된다.
- 소비자 보호와 근로자 지원을 위해 의료 분야에서의 AI 사용 촉진 및 교육 도구 개발이 강조된다.
- 국가AI연구자원(NAIRR)을 통해 AI 연구를 촉진하고, 외국인 전문가들이 미국에서 공부하고 일할 수 있도록 지원하는 방안도 포함된다.
- 또한, G7은 '히로시마 AI 프로세스'를 통해 AI 기업을 위한 국제 행동강령에 합의하였으며, 이는 AI 시스템의 위험 식별과 완화를 위한 자발적 채택을 권장하고, 위험 평가와 투명성, 책임성을 강조한다. 
- G7은 정보공유와 협력, 보안 통제, 콘텐츠 인증 및 출처 확인 메커니즘 개발 등을 포함한 행동강령을 지속적으로 개정할 계획이며, 사회적 위험 완화와 글로벌 문제 해결을 위한 AI 시스템 개발에 우선 투자할 예정이다.

---------------------------


- 미국 바이든 대통령이 안전하고 신뢰할 수 있는 AI 개발과 사용을 위한 행정명령을 발표하였다.
- 이 행정명령은 AI의 안전과 보안 기준, 개인정보 보호, 형평성과 시민권 향상, 소비자 보호, 노동자 지원, 혁신과 경쟁 촉진, 국제 협력을 포함한다.
- AI 시스템 개발 기업은 안전 테스트 결과와 시스템 정보를 정부와 공유해야 하며, AI의 책임 있는 사용을 위한 모범사례와 지침이 마련된다.
- 소비자 보호와 근로자 지원을 위해 의료 분야에서의 AI 사용 촉진 및 교육 도구 개발이 강조된다.
- 국가AI연구자원(NAIRR)을 통해 AI 연구를 촉진하고, 외국인 전문가들이 미국에서 공부하고 일할 수 있도록 지원하는 방안도 포함된다.
- G7은 '히로시마 AI 프로세스'를 통해 AI 기업을 위한 국제 행동강령에 합의하였으며, 이는 AI 시스템의 위험 식별과 완화를 위한 자발적 채택을 권장하고, 위험 평가와 투명성, 책임성을 강조한다.
- G7은 정보공유와 협력, 보안 통제, 콘텐츠 인증 및 출처 확인 메커니즘 개발 등을 포함한 행동강령을 지속적으로 개정할 계획이며, 사회적 위험 완화와 글로벌 문제 해결을 위한 AI 시스템 개발에 우선 투자할 예정이다.
- 또한, 28개국이 영국 블레츨리 파크에서 열린 AI 안전성 정상회의에서 AI 안전 보장을 위한 블레츨리 선언을 발표하였다. 이 선언은 AI 시스템의 안전성을 보장하기 위해 모든 이해관계자의 협력이 필요하다고 강조하며, 특히 AI 개발 기업의 책임을 지적하였다.
- 영국 총리는 AI 안전 연구소의 출범과 함께 첨단 AI 모델에 대한 안전성 시험 계획을 발표하였고, 각국 정부는 테스트 결과를 공유하고 공동 표준 개발에 노력하기로 합의하였다.
...
```

## 5. Chain of Density (CoD): 점점 밀도 높은 요약

CoD는 요약을 한 번에 끝내지 않고 **여러 번(기본 5회) 반복하면서, 매번 이전 요약에 빠진 핵심 개체(entity) 1~3개를 찾아 추가하고 길이는 유지**하게 한다. 반복할수록 같은 길이 안에 정보가 더 촘촘하게 들어간다.

### 5-1. CoD 프롬프트 확인

프롬프트 변수는 `content`(요약할 글), `content_category`(글의 종류), `entity_range`(한 번에 추가할 개체 수), `max_words`(요약 최대 단어 수), `iterations`(반복 횟수)다. 결과는 `missing_entities`와 `denser_summary`를 가진 JSON 리스트로 나온다.

```python
cod_prompt = client.pull_prompt("teddynote/chain-of-density-prompt", dangerously_pull_public_prompt=True)

cod_prompt.pretty_print()
```

출력:

```txt
================================ System Message ================================

As an expert copy-writer, you will write increasingly concise, entity-dense summaries of the user provided {content_category}. The initial summary should be under {max_words} words and contain {entity_range} informative Descriptive Entities from the {content_category}.

A Descriptive Entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Faithful: present in the {content_category}.
- Anywhere: located anywhere in the {content_category}.

# Your Summarization Process
- Read through the {content_category} and the all the below sections to get an understanding of the task.
- Pick {entity_range} informative Descriptive Entities from the {content_category} (";" delimited, do not add spaces).
- In your output JSON list of dictionaries, write an initial summary of max {max_words} words containing the Entities.
- You now have `[{"missing_entities": "...", "denser_summary": "..."}]`

Then, repeat the below 2 steps {iterations} times:

- Step 1. In a new dict in the same list, identify {entity_range} new informative Descriptive Entities from the {content_category} which are missing from the previously generated summary.
- Step 2. Write a new, denser summary of identical length which covers every Entity and detail from the previous summary plus the new Missing Entities.
...
```

### 5-2. CoD 체인 만들기

입력 딕셔너리에서 값이 빠져 있으면 기본값을 채우도록 lambda로 전처리한 뒤 프롬프트에 넣는다. 결과가 JSON이므로 `SimpleJsonOutputParser`로 파싱한다. `cod_final_summary_chain`은 마지막 반복의 `denser_summary`만 꺼내는 체인이다.

```python
import textwrap 
from langsmith import Client 
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import SimpleJsonOutputParser 

# 프롬프트 변수 전처리: 입력에 값이 없으면 기본값을 사용
cod_chain_inputs = {
    "content": lambda d: d.get("content"),
    "content_category": lambda d: d.get("content_category", "Article"),
    "entity_range": lambda d: d.get("entity_range", "1-3"),
    "max_words": lambda d: int(d.get("max_words", 80)),
    "iterations": lambda d: int(d.get("iterations", 5)),
}

client = Client()
cod_prompt = client.pull_prompt("teddynote/chain-of-density-prompt", dangerously_pull_public_prompt=True)

# 전처리 -> 프롬프트 -> LLM -> JSON 파싱 (반복별 요약이 담긴 리스트)
cod_chain = (
    cod_chain_inputs 
    | cod_prompt 
    | ChatOpenAI(temperature=0, model="gpt-4o-mini")
    | SimpleJsonOutputParser()
)

# 마지막(가장 밀도 높은) 요약만 꺼내는 체인
cod_final_summary_chain = cod_chain | (
    lambda output: output[-1].get(
        "denser_summary", '오류: 마지막 딕셔너리에 "denser_summary" 키가 없습니다'
    )
)
```

### 5-3. 요약할 문서 준비

앞에서 불러온 5페이지 중 두 번째 페이지(G7 히로시마 AI 프로세스 기사)를 요약한다.

```python
# 두 번째 페이지(G7 히로시마 AI 프로세스)의 본문
content = docs[1].page_content 
print(content)
```

출력:

```txt
SPRi AI Brief |  
2023-12월호
2
G7, 히로시마 AI 프로세스를 통해 AI 기업 대상 국제 행동강령에 합의
n G7이 첨단 AI 시스템을 개발하는 기업을 대상으로 AI 위험 식별과 완화를 위해 자발적인 
채택을 권고하는 AI 국제 행동강령을 마련
n 행동강령은 AI 수명주기 전반에 걸친 위험 평가와 완화, 투명성과 책임성의 보장, 정보공유와 
이해관계자 간 협력, 보안 통제, 콘텐츠 인증과 출처 확인 등의 조치를 요구
KEY Contents
£ G7, 첨단 AI 시스템의 위험 관리를 위한 국제 행동강령 마련
n 주요 7개국(G7)*은 2023년 10월 30일 ‘히로시마 AI 프로세스’를 통해 AI 기업 대상의 AI 국제 
행동강령(International Code of Conduct for Advanced AI Systems)에 합의
∙ G7은 2023년 5월 일본 히로시마에서 개최된 정상회의에서 생성 AI에 관한 국제규범 마련과 
정보공유를 위해 ‘히로시마 AI 프로세스’를 출범**
∙ 기업의 자발적 채택을 위해 마련된 이번 행동강령은 기반모델과 생성 AI를 포함한 첨단 AI 시스템의 
위험 식별과 완화에 필요한 조치를 포함
* 주요 7개국(G7)은 미국, 일본, 독일, 영국, 프랑스, 이탈리아, 캐나다를 의미
** 5월 정상회의에는 한국, 호주, 베트남 등을 포함한 8개국이 초청을 받았으나, AI 국제 행동강령에는 우선 G7 국가만 포함하여 채택
n G7은 행동강령을 통해 아래의 조치를 제시했으며, 빠르게 발전하는 기술에 대응할 수 있도록 
이해관계자 협의를 통해 필요에 따라 개정할 예정
...
```

### 5-4. CoD 실행

`SimpleJsonOutputParser`는 스트리밍 중에도 지금까지 완성된 부분까지 JSON을 파싱해주기 때문에, `stream()`으로 받으면 반복 결과가 하나씩 늘어나는 걸 볼 수 있다. 스트리밍이 끝나면 반복마다 추가된 개체와 요약을 차례로 출력한다.

```python
results: list[dict[str, str]] = []

# 스트리밍 중에는 지금까지 파싱된 JSON 리스트가 계속 갱신되어 들어온다.
for partial_json in cod_chain.stream(
    {"content": content, "content_category": "Article"}
):
    results = partial_json 

    # \r로 같은 줄을 덮어쓰며 진행 상황 표시
    print(results, end="\r", flush=True)

total_summaries = len(results)
print("\n") 

# 반복(iteration)마다 새로 추가된 개체와 그때의 요약을 출력
i = 1 
for cod in results:
    # missing_entities는 "개체1;개체2" 형태의 문자열이라 ;로 나눠서 정리
    added_entities = ", ".join(
        [
            ent.strip()
            for ent in cod.get(
                "missing_entities", 'ERR: "missing_entities" key not found'
            ).split(";")
        ]
    )

    summary = cod.get("denser_summary", 'ERR: mssing key "denser_summary"')

    print(
        f"### CoD Summary {i}/{total_summaries}, 추가된 엔티티(entity): {added_entities}"
        + "\n"
    )

    # 80자마다 줄바꿈해서 보기 좋게 출력
    print(textwrap.fill(summary, width=80) + "\n")
    i += 1

# 반복문이 끝났을 때의 summary가 마지막(최종) 요약
print("\n======================= [최종 요약] =======================\n")
print(summary)
```

출력:

```txt
[{'missing_entities': 'G7;히로시마 AI 프로세스;AI 국제 행동강령', 'denser_summary': 'G7은 2023년 10월 30일 히로시마 AI 프로세스를 통해 AI 기업을 위한 AI 국제 행동강령을 마련했다. 이 행동강령은 AI 위험 식별과 완화를 위한 자발적 채택을 권장하며, AI 수명주기 전반의 위험 평가, 투명성, 책임성, 정보공유, 보안 통제, 콘텐츠 인증 등의 조치를 요구한다.'}, {'missing_entities': '위험 평가;투명성;정보공유', 'denser_summary': 'G7은  ...

### CoD Summary 1/5, 추가된 엔티티(entity): G7, 히로시마 AI 프로세스, AI 국제 행동강령

G7은 2023년 10월 30일 히로시마 AI 프로세스를 통해 AI 기업을 위한 AI 국제 행동강령을 마련했다. 이 행동강령은 AI 위험 식별과
완화를 위한 자발적 채택을 권장하며, AI 수명주기 전반의 위험 평가, 투명성, 책임성, 정보공유, 보안 통제, 콘텐츠 인증 등의 조치를
요구한다.

### CoD Summary 2/5, 추가된 엔티티(entity): 위험 평가, 투명성, 정보공유

G7은 2023년 10월 30일 히로시마 AI 프로세스를 통해 AI 기업을 위한 AI 국제 행동강령을 마련했다. 이 행동강령은 AI 위험 식별과
완화를 위한 자발적 채택을 권장하며, AI 수명주기 전반의 위험 평가, 투명성, 책임성, 정보공유, 보안 통제, 콘텐츠 인증 등의 조치를
요구한다.

### CoD Summary 3/5, 추가된 엔티티(entity): AI 수명주기, 보안 통제, 콘텐츠 인증

G7은 2023년 10월 30일 히로시마 AI 프로세스를 통해 AI 기업을 위한 AI 국제 행동강령을 마련했다. 이 행동강령은 AI 위험 식별과
완화를 위한 자발적 채택을 권장하며, AI 수명주기 전반의 위험 평가, 투명성, 책임성, 정보공유, 보안 통제, 콘텐츠 인증 등의 조치를
요구한다.

...
```

## 정리

| 방식 | LLM 호출 | 장점 | 단점 |
|---|---|---|---|
| Stuff | 1번 | 가장 단순하고 문맥이 끊기지 않음 | 문서가 길면 컨텍스트 길이 초과·비용 증가 |
| Map-Reduce | 문서 수 + 1번 | Map을 `batch()`로 병렬 처리해서 빠름, 긴 문서도 가능 | 문서 간 순서·흐름이 약해질 수 있음 |
| Map-Refine | 문서 수 + (문서 수 − 1)번 (Refine은 순차) | 앞 내용을 이어받아 흐름이 자연스러움 | 순차 실행이라 느림 |
| Chain of Density | 1번 (프롬프트 안에서 반복) | 같은 길이에 핵심 정보를 촘촘하게 담음 | JSON 출력 형식이 깨지면 파싱 오류 가능 |

- `client.pull_prompt(..., dangerously_pull_public_prompt=True)`: LangSmith Hub의 공개 프롬프트 불러오기
- `@chain`: 여러 단계로 된 파이썬 함수를 Runnable로 만들어 `invoke()`/`stream()`으로 호출
- Clustering-Map-Refine은 다음에 기회가 되면 공부해볼 예정
