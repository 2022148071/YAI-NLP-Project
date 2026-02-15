"""
LangGraph.py RAG 파이프라인 모듈 (LangGraph 기반)
====================================================

LangGraph.ipynb 노트북의 전체 기능을 import 가능한 Python 모듈로 정리.
stream.py 에서 import 하여 Streamlit 데모에 활용합니다.

실행 순서 (의존성 고려):
  0.  표준 라이브러리 · sys.path 설정
  1.  환경 변수 로드 (.env)
  2.  로그 수집기 (stream.py → toast 연동)
  3.  상수 정의
  4.  외부 패키지 import (LangChain, LangGraph 등)
  5.  로컬 모듈 import (rag 패키지)
  6.  GraphState 정의
  7.  모듈 레벨 변수 (초기화 시 설정)
  8.  초기화 함수
  9.  문서 적재 API
  10. 헬퍼 함수
  11. 노드 함수
  12. 라우팅 함수 (conditional_edges 용)
  13. 그래프 구성 · 컴파일
  14. 공개 API (query 등)
"""

# ============================================================
# 0. 표준 라이브러리 · 경로 설정
# ============================================================
import os
import sys
import json
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any

# 이 파일은 <project_root>/LangGraph/ 에 위치.
# rag 패키지를 import 하려면 프로젝트 루트가 sys.path 에 있어야 한다.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================
# 1. 환경 변수 로드
# ============================================================
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 HF_API_KEY, TAVILY_API_KEY 등 로드

# ============================================================
# 2. 로그 수집기
#    - 노드 내부 print 대신 _log() 사용
#    - stream.py 에서 get_and_clear_logs() → st.toast()
# ============================================================
_log_buffer: List[str] = []


def _log(msg: str):
    """내부 메시지를 버퍼에 저장하고 콘솔에도 출력"""
    _log_buffer.append(msg)
    print(msg)


def get_and_clear_logs() -> List[str]:
    """쌓인 로그를 반환하고 버퍼를 비운다 (stream.py 가 호출)."""
    msgs = _log_buffer.copy()
    _log_buffer.clear()
    return msgs


# ============================================================
# 3. 상수
# ============================================================
PERSIST_DIR = "./chroma_db"  # ChromaDB 저장 경로 (LangGraph/ 기준 상대 경로)
COLLECTION_MAIN = "my_collection"  # 주요 문서 컬렉션
COLLECTION_CHAT_RAW = "chat_history_raw"  # 대화 원본 저장
COLLECTION_CHAT_SUMMARY = "chat_history_summarized"  # 대화 요약 저장

# LLM 모델 식별자
# ROUTER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # 라우팅·판단·요약용
# CHAIN_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # 답변 생성용 (rag.base)

ROUTER_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # 라우팅·판단·요약용
CHAIN_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # 답변 생성용 (rag.base)
EMBEDDING_MODEL = "BAAI/bge-m3"  # 임베딩 모델

MAX_CHARS_PER_DOC = 1500  # 웹 검색 결과 요약 임계치 (≈1000 토큰)

# ============================================================
# 4. 외부 패키지 import
# ============================================================
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# 5. 로컬 모듈 import (rag 패키지)
# ============================================================
from rag.base import create_embedding_auto
from rag.chroma import ChromaRetrievalChain
from rag.ingest import ingest_documents as _raw_ingest_docs
from rag.ingest import ingest_pdfs as _raw_ingest_pdfs
from rag.utils import format_docs
from rag.graph_utils import random_uuid  # 세션 ID 생성용 (re-export)

# ============================================================
# 6. GraphState 정의
# ============================================================


class GraphState(TypedDict):
    """LangGraph 노드 간 전달되는 상태 딕셔너리"""

    question: Annotated[str, "사용자 질문 (재작성 후 갱신됨)"]
    context: Annotated[str, "검색·웹 결과를 합친 문맥 텍스트"]
    answer: Annotated[str, "LLM 이 생성한 최종 답변"]
    messages: Annotated[list, add_messages]  # 대화 이력 (누적)
    relevance: Annotated[str, "검색 문서 관련성 yes/no"]


# ============================================================
# 7. 모듈 레벨 변수 — initialize() 에서 설정됨
# ============================================================
_retriever = None  # ChromaDB 기반 retriever
_chain = None  # RAG 답변 체인
_chat_hf = None  # 라우팅·판단·요약용 LLM
_embeddings = None  # 임베딩 모델 인스턴스
_app = None  # 컴파일된 LangGraph 앱
_initialized = False  # 초기화 완료 플래그

# ============================================================
# 8. 초기화 함수
# ============================================================


def _init_hf_login():
    """HuggingFace Hub 토큰 로그인"""
    from huggingface_hub import login

    token = os.getenv("HF_API_KEY")
    if token:
        os.environ["HF_API_KEY"] = token
        login(token=token)
        _log("✅ HuggingFace 로그인 성공")
    else:
        _log("⚠️ HF_API_KEY 가 설정되지 않았습니다")


def _init_chat_model():
    """라우팅 · 판단 · 요약용 LLM 초기화 (Llama-3.1-8B-Instruct)"""
    global _chat_hf
    llm = HuggingFaceEndpoint(
        repo_id=ROUTER_MODEL,
        task="text-generation",
        temperature=0.7,
        max_new_tokens=512,
    )
    _chat_hf = ChatHuggingFace(llm=llm)
    _log(f"✅ 라우팅 LLM 로드 완료: {ROUTER_MODEL}")


def _init_embeddings():
    """임베딩 모델 초기화 (create_embedding_auto → 로컬/API 자동 선택)"""
    global _embeddings
    _embeddings = create_embedding_auto()
    _log(f"✅ 임베딩 모델 로드 완료: {EMBEDDING_MODEL}")


def _init_rag_chain(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_MAIN,
    k: int = 10,
):
    """ChromaDB 기반 RAG 체인 (retriever + chain) 초기화"""
    global _retriever, _chain
    _log("🚀 ChromaDB 기반 RAG 체인 생성 시작...")
    rag = ChromaRetrievalChain(
        persist_directory=persist_directory,
        collection_name=collection_name,
        k=k,
    ).create_chain()
    _retriever = rag.retriever
    _chain = rag.chain
    _log("✅ RAG 체인 생성 완료")


def initialize(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_MAIN,
    k: int = 10,
):
    """
    전체 파이프라인 초기화 — 최초 1 회만 실행.

    순서: HF 로그인 → 라우팅 LLM → 임베딩 → RAG 체인
    """
    global _initialized
    if _initialized:
        return
    _init_hf_login()
    _init_chat_model()
    _init_embeddings()
    _init_rag_chain(persist_directory, collection_name, k)
    _initialized = True
    _log("✅ 파이프라인 초기화 완료")


# ============================================================
# 9. 문서 적재 API
# ============================================================


def ingest_uploaded_file(
    file_path: str,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_MAIN,
):
    """
    업로드된 단일 파일 (PDF / TXT) 을 ChromaDB 에 적재.
    stream.py 파일 업로드에서 호출.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        _raw_ingest_pdfs(
            pdf_paths=[file_path],
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    else:
        _raw_ingest_docs(
            file_paths=[file_path],
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    _log(f"✅ 파일 적재 완료: {Path(file_path).name}")


# ============================================================
# 10. 헬퍼 함수
# ============================================================


def _to_text(msg) -> str:
    """
    다양한 메시지 타입을 'role: content' 문자열로 변환.
    - LangChain BaseMessage (type, content 속성)
    - (role, content) 튜플/리스트
    - 기타 → str()
    """
    if hasattr(msg, "type") and hasattr(msg, "content"):
        return f"{msg.type}: {msg.content}"
    if isinstance(msg, (tuple, list)) and len(msg) >= 2:
        return f"{msg[0]}: {msg[1]}"
    return str(msg)


def _extract_question(raw) -> str:
    """state['question'] 이 어떤 타입이든 순수 문자열로 추출"""
    if hasattr(raw, "content"):
        return str(raw.content)
    if isinstance(raw, (list, tuple)) and raw:
        last = raw[-1]
        return str(last.content) if hasattr(last, "content") else str(last)
    return str(raw)


def _looks_ambiguous(q: str) -> bool:
    """짧거나 대명사 · 모호 표현이 포함된 질문인지 휴리스틱 판별"""
    q = (q or "").strip()
    if not q:
        return False
    ambiguous = [
        "그거", "그것", "그게", "이게", "이거", "저거", "그때", "저번", "아까",
        "그 내용", "그 이야기", "기억나", "기억해", "다시", "이어",
        "더 자세히", "뭐였지",
    ]
    short_followups = ["왜?", "어째서?", "뭐야?", "뭔데?", "그게 뭐야?", "설명해줘"]
    return any(t in q for t in ambiguous) or q in short_followups or len(q) <= 8


def _message_to_role_content(msg):
    """메시지 → (role, content) 튜플 변환"""
    if hasattr(msg, "type") and hasattr(msg, "content"):
        role = {"human": "user", "ai": "assistant"}.get(msg.type, msg.type)
        return role, str(msg.content)
    if isinstance(msg, (tuple, list)) and len(msg) >= 2:
        return str(msg[0]), str(msg[1])
    return "unknown", str(msg)


def _conversation_only(messages) -> list:
    """user/assistant 역할의 메시지만 필터링"""
    conv = []
    for m in messages:
        role, content = _message_to_role_content(m)
        if role in {"user", "assistant", "human", "ai"}:
            conv.append((role, content))
    return conv


def _summarize_if_long(content: str, max_chars: int = MAX_CHARS_PER_DOC) -> str:
    """텍스트가 max_chars 를 초과하면 _chat_hf 로 요약"""
    if len(content) <= max_chars:
        return content
    prompt = (
        f"아래 텍스트를 핵심만 남겨 {max_chars}자 이내로 요약해주세요. "
        f"한글로 작성하고 불필요한 반복은 제거하세요. 요약만 출력.\n\n"
        f"---\n{content[:8000]}\n---"
    )
    try:
        resp = _chat_hf.invoke(prompt)
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        return text[:max_chars]
    except Exception:
        return content[:max_chars] + "..."


# ============================================================
# 11. 노드 함수
# ============================================================


def contextualize(state: GraphState) -> GraphState:
    """
    [contextualize 노드]
    사용자 질문을 분석하여 과거 대화 맥락이 필요한지 판단.
    필요 시 chat_history_summarized 컬렉션에서 검색 후 질문을 재작성.

    판단 기준 (OR 조건):
      1) 키워드 매칭 (그때, 저번에, 아까, …)
      2) 모호한 표현 감지 (_looks_ambiguous)
      3) LLM 판단 (recall_judgment_prompt)
    """
    messages = state.get("messages", [])
    question = _extract_question(state.get("question", "")).strip()
    # 최근 대화 10 메시지를 텍스트로 변환
    recent_chat = "\n".join(_to_text(m) for m in messages[-10:])

    # ── 1) recall 필요 여부 판단 ──────────────────────────────
    keyword_recall = any(
        kw in question
        for kw in [
            "그때", "저번에", "아까", "이전", "기억나", "그게", "이게",
            "위에", "그거", "내 생일", "내 정보", "이건", "그건",
        ]
    )
    ambiguous_recall = _looks_ambiguous(question)

    llm_recall = False
    judge_prompt = f"""당신은 질의 라우팅 판별기입니다.
아래 사용자 질문이 과거 대화 맥락(특히 개인 정보/이전 대화 요약) 없이는 해석이 어려운지 판단하세요.

[Recent Chat]
{recent_chat}

[Question]
{question}

출력은 반드시 아래 둘 중 하나만:
YES
NO""".strip()

    try:
        resp = _chat_hf.invoke(judge_prompt)
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
        llm_recall = "YES" in text
    except Exception:
        pass

    is_recall_needed = keyword_recall or ambiguous_recall or llm_recall
    rewrite_question = question
    long_term_context = ""

    # ── 2) recall 필요 시 → 요약 DB 검색 → 질문 재작성 ──────
    if is_recall_needed:
        _log("🔍 과거 대화 요약 DB 검색 중...")

        summary_store = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_CHAT_SUMMARY,
            embedding_function=_embeddings,
        )

        # 검색 친화적 쿼리 생성
        rq_prompt = f"""사용자 질문으로 벡터 검색할 쿼리를 1 문장으로 만들어주세요.
- 과거 대화에서 찾아야 할 핵심 엔티티를 포함하세요.
- 불필요한 수식어 없이 검색 친화적으로 작성하세요.
- 질문에 답하지 말고 검색 쿼리 문장만 출력하세요.

[Recent Chat]
{recent_chat}

[Question]
{question}""".strip()

        retrieval_query = question
        try:
            rq = _chat_hf.invoke(rq_prompt)
            cand = (rq.content if hasattr(rq, "content") else str(rq)).strip()
            if cand:
                retrieval_query = cand
        except Exception:
            pass

        docs = summary_store.similarity_search(retrieval_query, k=3)
        if docs:
            long_term_context = "\n".join(d.page_content for d in docs)

        # 질문 재작성
        rewrite_prompt = f"""You are a query rewriter.
Rewrite the user's question to be clear and standalone.
Use retrieved long-term context if available. If not available, use only recent chat.
Do not answer. Return only one rewritten question in Korean.

[Recent Chat]
{recent_chat}

[Retrieved Long-term Context]
{long_term_context}

[Original Question]
{question}""".strip()

        try:
            resp = _chat_hf.invoke(rewrite_prompt)
            cand = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            if cand:
                rewrite_question = cand
        except Exception:
            rewrite_question = question

    return GraphState(question=rewrite_question)


def retrieve(state: GraphState) -> GraphState:
    """
    [retrieve 노드]
    ChromaDB retriever 로 사용자 질문과 관련된 문서를 검색.
    """
    docs = _retriever.invoke(state["question"])
    return GraphState(context=format_docs(docs))


def llm_answer(state: GraphState) -> GraphState:
    """
    [llm_answer 노드]
    RAG 체인을 호출하여 최종 답변을 생성.
    답변과 함께 (user, assistant) 메시지 쌍을 messages 에 추가.
    """
    question = state["question"]
    context = state.get("context", "")
    chat_history = state.get("messages", [])

    try:
        response = _chain.invoke(
            {
                "question": question,
                "context": context,
                "chat_history": chat_history,
            }
        )
    except Exception as e:
        _log(f"❌ LLM 답변 생성 실패: {type(e).__name__}: {e}")
        raise

    return GraphState(
        answer=response,
        messages=[("user", question), ("assistant", response)],
    )


def relevance_check(state: GraphState) -> GraphState:
    """
    [relevance_check 노드]
    검색된 문서(context)가 질문과 관련 있는지 _chat_hf 로 평가.
    결과를 state['relevance'] = 'yes' | 'no' 로 저장.
    """
    prompt = f"""You are a grader assessing whether a retrieved document is relevant to the given question.
Return ONLY valid JSON like: {{"score": "yes"}} or {{"score": "no"}}.

Question:
{state["question"]}

Retrieved document:
{state["context"]}""".strip()

    resp = _chat_hf.invoke(prompt)
    text = resp.content.strip()

    # JSON 부분만 추출 (모델이 앞뒤에 텍스트를 섞는 경우 대비)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
        score = data.get("score", "no").lower()
    except Exception:
        score = "no"

    if score not in ("yes", "no"):
        score = "no"

    _log(f"📋 관련성 평가: {score}")
    return {"relevance": score}


def web_search(state: GraphState) -> GraphState:
    """
    [web_search 노드]
    Tavily API 로 웹 검색 후 결과를 context 에 저장.
    검색 결과는 ChromaDB(my_collection)에도 적재하여 재활용.
    """
    _log("🌐 웹 검색 시작...")
    tavily = TavilySearchResults(max_results=5, search_depth="basic")
    query_text = state["question"]
    results = tavily.invoke(query_text)

    # 결과 포맷팅 (긴 본문은 요약)
    parts = []
    for r in results:
        url = r.get("url", "")
        content = _summarize_if_long(r.get("content", ""))
        parts.append(f"{content}\n출처: {url}")
    formatted = "\n\n---\n\n".join(parts)

    # ChromaDB 에도 저장
    if formatted.strip():
        doc = Document(
            page_content=formatted,
            metadata={
                "source": f"web_search:{query_text}",
                "origin": "tavily_merged",
            },
        )
        try:
            _raw_ingest_docs(
                documents=[doc],
                persist_directory=PERSIST_DIR,
                collection_name=COLLECTION_MAIN,
            )
            _log("✅ 웹 검색 결과 ChromaDB 저장 완료")
        except Exception as e:
            _log(f"⚠️ 웹 검색 결과 저장 실패: {e}")

    return GraphState(context=formatted)


def save_memory(state: GraphState) -> GraphState:
    """
    [save_memory 노드]
    누적 대화가 충분할 때 오래된 5 턴(10 messages)을
    raw / summary 컬렉션에 각각 저장.
    """
    messages = state.get("messages", [])
    conv = _conversation_only(messages)
    MIN_MSGS = 10  # 5 턴 = 10 메시지

    if len(conv) < MIN_MSGS:
        _log(f"ℹ️ save_memory 건너뜀: 대화 {len(conv)}개 (< {MIN_MSGS})")
        return {}

    oldest = conv[:MIN_MSGS]
    raw_text = "\n".join(f"{r}: {c}" for r, c in oldest).strip()
    if not raw_text:
        return {}

    # ── 요약 생성 ──
    summary_prompt = f"""다음은 사용자-어시스턴트 대화의 오래된 5 턴입니다.
핵심 사실(개인정보/선호/약속/중요 맥락)만 한국어로 4~6 문장 내로 요약하세요.
질문에 답하지 말고, 메모리 저장용 요약만 출력하세요.

[Conversation]
{raw_text}""".strip()

    try:
        resp = _chat_hf.invoke(summary_prompt)
        summary_text = (
            resp.content if hasattr(resp, "content") else str(resp)
        ).strip()
    except Exception as e:
        _log(f"⚠️ 요약 생성 실패: {e}")
        summary_text = raw_text[:1200]

    ts = datetime.now(timezone.utc).isoformat()
    mem_id = uuid.uuid4().hex

    raw_doc = Document(
        page_content=raw_text,
        metadata={
            "source": "chat_history_raw",
            "memory_id": mem_id,
            "saved_at": ts,
            "turn_count": 5,
            "message_count": MIN_MSGS,
        },
    )
    summary_doc = Document(
        page_content=summary_text,
        metadata={
            "source": "chat_history_summarized",
            "memory_id": mem_id,
            "saved_at": ts,
            "turn_count": 5,
            "message_count": MIN_MSGS,
        },
    )

    try:
        _raw_ingest_docs(
            documents=[raw_doc],
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_CHAT_RAW,
            chunk_size=1200,
            chunk_overlap=120,
        )
        _raw_ingest_docs(
            documents=[summary_doc],
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_CHAT_SUMMARY,
            chunk_size=400,
            chunk_overlap=40,
        )
        _log("✅ save_memory 완료 (raw + summary 저장)")
    except Exception as e:
        _log(f"⚠️ save_memory 저장 실패: {e}")

    return {}


# ============================================================
# 12. 라우팅 함수 (conditional_edges 용)
# ============================================================


def retrieve_or_not(state: GraphState) -> str:
    """
    사용자 질문에 대해 문서 검색(retrieve)이 필요한지 LLM 으로 판단.
    - 검색 불필요 → "not retrieve" → llm_answer 직행
    - 검색 필요   → "retrieve"     → retrieve 노드
    """
    question = state.get("question", "")
    if not question:
        return "not retrieve"

    prompt = f"""다음 사용자 질문에 답하려면 **문서/벡터DB 검색(retrieve)**이 필요한지 판단하세요.

판단 기준:
- 인사, 감정, 단순 대화("안녕", "고마워", "뭐해" 등), 잡담 → 검색 불필요
- 문서에 있을 법한 전문 지식 질문 → 검색 필요
- 최신 정보/뉴스 → 검색 필요

질문: {question}

*반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON 만 출력.
{{"need_retrieve": "yes"}} 또는 {{"need_retrieve": "no"}}""".strip()

    try:
        resp = _chat_hf.invoke(prompt)
        text = (resp.content or "").strip()
        match = re.search(r'\{[^{}]*"need_retrieve"[^{}]*\}', text)
        if match:
            data = json.loads(match.group(0))
            need = (data.get("need_retrieve") or "no").lower()
            if need in ("yes", "true", "1"):
                _log("📖 → retrieve 노드로 이동")
                return "retrieve"
        _log("💬 → llm_answer 노드로 직행")
        return "not retrieve"
    except Exception:
        return "retrieve"  # 에러 시 안전하게 검색 실행


def is_relevant(state: GraphState) -> str:
    """관련성 평가 결과에 따라 분기"""
    return "relevant" if state.get("relevance") == "yes" else "not relevant"


def save_or_not(state: GraphState) -> str:
    """메시지 수 > 20 이면 save_memory 로 분기"""
    return "save_chat" if len(state.get("messages", [])) > 20 else "too short"


# ============================================================
# 13. 그래프 구성 · 컴파일
# ============================================================


def build_app():
    """
    LangGraph 워크플로우를 구성하고 컴파일한다.

    그래프 구조:
      START → contextualize
              ├─ (retrieve 필요)  → retrieve → relevance_check
              │                                ├─ (relevant)     → llm_answer
              │                                └─ (not relevant) → web_search → llm_answer
              └─ (retrieve 불필요) → llm_answer
                                     ├─ (save_chat) → save_memory → END
                                     └─ (too short) → END

    Returns:
        컴파일된 LangGraph 앱
    """
    global _app

    workflow = StateGraph(GraphState)

    # ── 노드 등록 ──
    workflow.add_node("contextualize", contextualize)
    workflow.add_node("save_memory", save_memory)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("llm_answer", llm_answer)
    workflow.add_node("relevance_check", relevance_check)
    workflow.add_node("web_search", web_search)

    # ── 진입점 ──
    workflow.set_entry_point("contextualize")

    # ── 조건부 엣지: contextualize → retrieve | llm_answer ──
    workflow.add_conditional_edges(
        "contextualize",
        retrieve_or_not,
        {"retrieve": "retrieve", "not retrieve": "llm_answer"},
    )

    # ── retrieve → relevance_check ──
    workflow.add_edge("retrieve", "relevance_check")

    # ── 조건부 엣지: relevance_check → llm_answer | web_search ──
    workflow.add_conditional_edges(
        "relevance_check",
        is_relevant,
        {"relevant": "llm_answer", "not relevant": "web_search"},
    )

    # ── web_search → llm_answer ──
    workflow.add_edge("web_search", "llm_answer")

    # ── 조건부 엣지: llm_answer → save_memory | END ──
    workflow.add_conditional_edges(
        "llm_answer",
        save_or_not,
        {"save_chat": "save_memory", "too short": END},
    )

    # ── save_memory → END ──
    workflow.add_edge("save_memory", END)

    # ── 컴파일 (MemorySaver: 인메모리 체크포인터) ──
    memory = MemorySaver()
    _app = workflow.compile(checkpointer=memory)
    _log("✅ LangGraph 앱 컴파일 완료")
    return _app


# ============================================================
# 14. 공개 API
# ============================================================


def query(question: str, thread_id: str | None = None) -> Dict[str, Any]:
    """
    질문을 실행하고 최종 GraphState 를 반환.

    Args:
        question:  사용자 질문 문자열
        thread_id: 대화 세션 ID (None 이면 자동 생성)

    Returns:
        최종 상태 딕셔너리 (question, context, answer, messages, relevance)
    """
    if _app is None:
        raise RuntimeError("build_app() 을 먼저 호출하세요")

    if thread_id is None:
        thread_id = random_uuid()

    config = RunnableConfig(
        recursion_limit=10,
        configurable={"thread_id": thread_id},
    )
    inputs = GraphState(question=question)

    # stream 모드로 실행 — 각 노드 완료 시 로그
    for event in _app.stream(inputs, config=config):
        for node_name in event:
            _log(f"🔄 {node_name} 노드 실행 완료")

    return _app.get_state(config).values


def get_app():
    """컴파일된 LangGraph 앱 인스턴스 반환"""
    return _app


def is_initialized() -> bool:
    """파이프라인 초기화 완료 여부"""
    return _initialized
