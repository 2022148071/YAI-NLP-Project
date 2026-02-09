# NeuLoRA

LangGraph 기반 개선 RAG 튜터 시스템입니다. PDF 문서 검색, 관련성 평가, 필요 시 웹 검색을 거쳐 답변을 생성합니다.

## 주요 기능

- **PDF RAG**: PDF 문서를 로드·분할 후 FAISS 벡터 저장소에 넣고, 질문에 맞는 문서를 검색
- **관련성 평가(Relevance Check)**: 검색된 문서가 질문과 관련 있는지 판별 (Groundedness Check)
- **조건부 웹 검색**: 관련성이 낮으면 Tavily로 웹 검색 후 그 결과를 컨텍스트로 활용
- **LangGraph 워크플로**: `retrieve → relevance_check → (관련 있음: llm_answer / 관련 없음: web_search → llm_answer)` 순서로 실행

## 프로젝트 구조

```
YAI-NLP/
├── LangGraph/
│   └── LangGraph.ipynb   # 그래프 정의, 노드(retrieve, relevance_check, web_search, llm_answer), 실행(2026/02/09 retrieve_or_not함수 추가가)
├── rag/
│   ├── base.py           # RetrievalChain (임베딩, 벡터스토어, 리트리버, LLM, 프롬프트, 체인)
│   ├── pdf.py            # PDFRetrievalChain (PDF 로더, 텍스트 분할)
│   ├── utils.py          # format_docs 등 유틸
│   └── prompts/          # RAG 프롬프트 템플릿
├── data/                 # PDF 등 데이터 경로
├── requirements.txt
├── .env                  # API 키 (HF_API_KEY, TAVILY_API_KEY, LANGCHAIN_API_KEY 등)
└── README.md
```

## 기술 스택

- **LangGraph**: 상태 그래프(StateGraph), 노드·엣지, 조건부 분기
- **LangChain**: RAG 체인, HuggingFace Embeddings/Endpoint, 프롬프트
- **RAG**: FAISS, RecursiveCharacterTextSplitter, PDFPlumberLoader
- **임베딩**: `jhgan/ko-sroberta-multitask`
- **LLM**: HuggingFace Inference API (예: HuggingFaceH4/zephyr-7b-beta), 필요 시 OpenAI
- **웹 검색**: Tavily (langchain_teddynote)
- **추적**: LangSmith (선택)

## 환경 설정

1. 저장소 클론 후 가상환경 생성 및 패키지 설치:

   ```bash
   pip install -r requirements.txt
   ```

2. 프로젝트 루트에 `.env` 파일을 만들고 다음 변수를 설정:

   ```env
   HF_API_KEY=your_huggingface_token
   TAVILY_API_KEY=your_tavily_key
   LANGCHAIN_API_KEY=your_langsmith_key   # 선택
   OPENAI_API_KEY=your_openai_key         # 관련성 체크 등에 OpenAI 사용 시
   ```

3. PDF 파일은 `LangGraph/` 또는 `data/` 등 프로젝트 내 경로에 두고, 노트북/코드에서 해당 경로를 지정합니다.

## 실행 방법

1. `LangGraph/LangGraph.ipynb`를 Jupyter에서 엽니다.
2. 셀을 위에서부터 순서대로 실행합니다.
   - 환경 설정, PDFRetrievalChain 생성, GraphState·노드 정의, 워크플로 구성, `invoke_graph` 또는 `app.invoke` 실행
3. `invoke_graph(app, inputs, config)` 또는 `stream_graph(...)`로 그래프를 실행하고, 터미널/노트북 출력으로 결과를 확인합니다.

## 그래프 흐름 요약

1. **retrieve**: 사용자 질문으로 PDF 리트리버 검색 → `context`에 검색된 문서 저장
2. **relevance_check**: 질문과 검색 문서의 관련성 판별 (yes/no)
3. **관련 있음** → **llm_answer**: 검색 문서와 채팅 히스토리를 이용해 답변 생성
4. **관련 없음** → **web_search**: Tavily로 웹 검색 → **llm_answer**: 검색 결과를 컨텍스트로 답변 생성

## 라이선스

본 프로젝트는 저장소에 포함된 LICENSE 파일을 따릅니다.
