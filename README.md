# NeuLoRA - The Thinking tutor that connects to your neuron

LangGraph 기반 **RAG(Retrieval-Augmented Generation)** 챗봇 프로젝트입니다.  
문서(PDF/TXT)를 벡터 DB에 적재하고, 대화 맥락·문서 검색·웹 검색을 조합해 답변을 생성합니다.

## 주요 기능

- **RAG 파이프라인**: ChromaDB + LangChain/LangGraph로 문서 검색 후 LLM 답변 생성
- **대화 맥락 활용**: "그때 말한 거", "아까 질문" 등 모호한 질문 시 이전 대화 요약 검색으로 질문 재작성
- **검색 라우팅**: 질문 유형에 따라 문서 검색 필요 여부를 LLM이 판단
- **관련성 검사**: 검색된 문서가 질문과 무관하면 Tavily 웹 검색으로 보완
- **대화 메모리**: 일정 턴 이상 대화 시 요약을 별도 컬렉션에 저장해 장기 기억 활용
- **다양한 실행 환경**: Jupyter 노트북, Streamlit 데모, FastAPI + React 웹 앱

## 프로젝트 구조

```
YAI-NLP/
├── LangGraph/
│   ├── LangGraph.py      # RAG 파이프라인 모듈 (LangGraph 그래프 정의)
│   ├── LangGraph.ipynb   # 실험용 노트북
│   ├── api.py            # FastAPI 백엔드 (REST API)
│   ├── stream.py         # Streamlit 데모
│   ├── chroma_db/        # ChromaDB 저장 디렉터리 (자동 생성)
│   └── frontend/         # React + Vite 프론트엔드
├── rag/                  # RAG 공통 모듈
│   ├── base.py           # 임베딩·체인 기본
│   ├── chroma.py         # ChromaDB 검색 체인
│   ├── ingest.py         # 문서 적재 (PDF/TXT)
│   ├── pdf.py            # PDF 처리
│   ├── utils.py          # 유틸리티
│   └── graph_utils.py    # 그래프·세션 유틸
├── requirements.txt
└── README.md
```

## 환경 설정

### 1. Python 환경

- Python 3.10+ 권장
- 가상환경 생성 후 의존성 설치:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. API 키 (.env)

프로젝트 루트에 `.env` 파일을 만들고 다음 변수를 설정하세요.

| 변수 | 설명 |
|------|------|
| `HF_API_KEY` | Hugging Face API 토큰 (LLM·임베딩 호출용) |
| `TAVILY_API_KEY` | Tavily 검색 API 키 (관련 문서 없을 때 웹 검색용, 선택) |

예시:

```
HF_API_KEY=hf_xxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxx
```

## 실행 방법

### Streamlit 데모

```bash
cd LangGraph
streamlit run stream.py
```

- 브라우저에서 채팅 UI 접속
- 사이드바에서 PDF/TXT 업로드, 문서 목록·연결 상태 확인

### FastAPI + React 웹 앱

**백엔드**

```bash
cd LangGraph
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload --port 8800
```

**프론트엔드**

```bash
cd LangGraph/frontend
npm install
npm run dev
```

- 프론트: http://localhost:5173 (Vite 기본)
- API: http://localhost:8800

### Jupyter 노트북

`LangGraph/LangGraph.ipynb` 를 열어 셀 단위로 파이프라인 실험 가능.

## API 엔드포인트 (FastAPI)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/chat` | 메시지 전송 → RAG 답변 (body: `message`, 선택 `thread_id`) |
| POST | `/api/upload` | PDF/TXT 파일 업로드 → ChromaDB 적재 |
| GET | `/api/status` | 모델명, HF/Tavily/ChromaDB/파이프라인 연결 상태 |
| GET | `/api/documents` | ChromaDB 컬렉션 목록 및 문서 수 |
| POST | `/api/reset` | 새 대화 세션 ID 발급 |

## 사용 모델

- **라우팅·판단·요약**: `Qwen/Qwen2.5-14B-Instruct`
- **답변 생성(RAG)**: `Qwen/Qwen2.5-14B-Instruct`
- **임베딩**: `BAAI/bge-m3`

(Hugging Face Inference API 또는 동일 인터페이스 사용)

## 라이선스 및 참고

- `.env` 는 버전 관리에서 제외되어 있습니다 (`.gitignore`).
- ChromaDB 데이터는 `LangGraph/chroma_db/` 에 로컬로 저장됩니다.
