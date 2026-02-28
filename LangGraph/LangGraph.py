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
import time
import json
import re
import uuid
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

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

STYLE_MODELS = {
    "direct": "RiverWon/NeuLoRA-direct",
    "socratic": "RiverWon/NeuLoRA-socratic",
    "scaffolding": "RiverWon/NeuLoRA-scaffolding",
    "feedback": "RiverWon/NeuLoRA-feedback",
}

MAX_CHARS_PER_DOC = 1500  # 웹 검색 결과 요약 임계치 (≈1000 토큰)

# llm_answer 프롬프트 내 chat_history 길이 제한
MAX_HISTORY_TURNS = 6  # 최근 N턴(2N개 메시지)까지 포함
MAX_HISTORY_CHARS = 2000  # 히스토리 블록 최대 문자 수, 초과 시 앞(오래된 턴)부터 생략

LORA_ROUTER_PATH = Path(__file__).parent / "router_model.json"

# PEFT 어댑터가 서브폴더에 있는 리포 (adapter_config.json 있음)
# direct:   https://huggingface.co/marimmo/multi-lora/tree/main/direct
# feedback: https://huggingface.co/marimmo/multi-lora/tree/main/feedback
# scaffolding: https://huggingface.co/marimmo/multi-lora/tree/main/scaffolding
# socratic: https://huggingface.co/marimmo/multi-lora/tree/main/socratic
PEFT_REPO = "marimmo/multi-lora" 
# ============================================================
# 4. 외부 패키지 import
# ============================================================
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage

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
    policy: Annotated[str, "학생에 대한 답안 방향성"]


# ============================================================
# 7. 모듈 레벨 변수 — initialize() 에서 설정됨
# ============================================================
_peft_model = None   # PEFT 어댑터 사용 시에만 설정; 전체 모델 로드 시 None
_rag_llm = None
_tokenizer = None
_use_peft_adapters = False  # True: set_adapter() 사용 / False: 단일 전체 모델
_retriever = None  # ChromaDB 기반 retriever
_chain = None  # RAG 답변 체인
_chat_hf = None  # 라우팅·판단·요약용 LLM
_embeddings = None  # 임베딩 모델 인스턴스
_app = None  # 컴파일된 LangGraph 앱
_initialized = False  # 초기화 완료 플래그
_answer_model_used: str | None = None  # 실제 체인 생성에 사용된 답변 모델명
_bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bg-db")

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


def _make_vessel_chat_model():
    """
    vessel(로컬 GPU)용 LLM 생성.
    transformers 파이프라인 → LangChain 호환 래퍼(.invoke() 반환값에 .content 있음).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    from langchain_huggingface import HuggingFacePipeline

    model_kwargs = {"device_map": "auto", "dtype": "auto"}
    q4 = os.getenv("LLM_4BIT", "").lower() in ("1", "true", "yes")
    q8 = os.getenv("LLM_8BIT", "").lower() in ("1", "true", "yes")
    if q4:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        _log("⚖️ 답변 LLM 4bit 양자화 사용")
    elif q8:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        _log("⚖️ 답변 LLM 8bit 양자화 사용")

    model = AutoModelForCausalLM.from_pretrained(ROUTER_MODEL, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(ROUTER_MODEL)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # ChatHuggingFace 가 문자열 출력(llm.invoke 결과)을 AIMessage(content=...)로 감싸주므로,
    # LangGraph 전체에서 기대하는 `.invoke(...).content` 인터페이스와 그대로 호환된다.
    return ChatHuggingFace(llm=llm)


def _init_chat_model():
    """라우팅·판단·요약용 LLM 초기화. LLM_MODE=vessel 이면 로컬 GPU, 아니면 API."""
    global _chat_hf

    mode = (os.getenv("LLM_MODE") or "api").strip().lower()

    if mode == "vessel":
        _chat_hf = _make_vessel_chat_model()
        _log(f"✅ 라우팅 LLM 로드 완료 (vessel 로컬): {ROUTER_MODEL}")
    else:
        llm = HuggingFaceEndpoint(
            repo_id=ROUTER_MODEL,
            task="text-generation",
            temperature=0.7,
            max_new_tokens=1024,
        )
        _chat_hf = ChatHuggingFace(llm=llm)
        _log(f"✅ 라우팅 LLM 로드 완료 (API): {ROUTER_MODEL}")


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
    global _retriever, _chain, _answer_model_used
    _log("🚀 ChromaDB 기반 RAG 체인 생성 시작...")
    rag = ChromaRetrievalChain(
        persist_directory=persist_directory,
        collection_name=collection_name,
        k=k,
    ).create_chain()
    _retriever = rag.retriever
    _chain = rag.chain
    _log("✅ RAG 체인 생성 완료")

def _init_peft_model():
    """8bit/4bit Multi-LoRA 또는 RiverWon/NeuLoRA-direct 같은 전체 모델 로드"""
    global _peft_model, _rag_llm, _tokenizer, _use_peft_adapters
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    )
    from langchain_community.llms import HuggingFacePipeline
    from peft import PeftModel, TaskType
    import torch
    import requests  # marimmo router 다운로드
    
    quant = os.getenv("LLM_QUANT", "8bit").lower()
    # GPU 메모리 부족 시 CPU 오프로드 허용 (VRAM 작을 때)
    enable_cpu_offload = os.getenv("LLM_CPU_OFFLOAD", "").lower() in ("1", "true", "yes")

    # 양자화 config
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _log("🔧 4bit 양자화 활성 (OOM 대비)")
    else:  # 8bit 우선
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weights=False,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,
        )
        if enable_cpu_offload:
            _log("🔧 8bit + CPU 오프로드 활성 (VRAM 부족 시 일부 레이어 CPU)")
    
    # 베이스 14B 로드 (flash_attn 미사용 시 sdpa/eager 사용 — PyTorch·flash_attn ABI 불일치 회피)
    attn_impl = (os.getenv("ATTN_IMPLEMENTATION") or "sdpa").strip().lower()
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except (ImportError, OSError) as e:
            _log(f"⚠️ flash_attention_2 로드 실패 ({e}) → sdpa 사용")
            attn_impl = "sdpa"
    if attn_impl not in ("flash_attention_2", "sdpa", "eager"):
        attn_impl = "sdpa"
    _log(f"🔧 attention 구현: {attn_impl}")

    # CPU 오프로드 시 device_map + max_memory로 GPU 한도 지정 후 나머지는 CPU
    if enable_cpu_offload and quant != "4bit":
        max_memory = {0: os.getenv("LLM_GPU_MAX_MEMORY", "20GiB"), "cpu": "30GiB"}
        device_map = "auto"
        _log(f"🔧 device_map=auto, max_memory={max_memory}")
    else:
        max_memory = None
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        CHAIN_MODEL,
        quantization_config=bnb_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(CHAIN_MODEL)
    _tokenizer.pad_token = _tokenizer.eos_token

    # 1) marimmo/multi-lora 서브폴더(direct, socratic 등)에서 PEFT 어댑터 로드 시도 (adapter_config.json 있음)
    # 2) 실패 시 RiverWon/NeuLoRA-* 개별 리포를 PEFT로 시도
    # 3) 그도 실패 시 RiverWon/NeuLoRA-direct 를 전체 모델로 로드
    run_model = None
    try:
        _peft_model = PeftModel.from_pretrained(
            model,
            PEFT_REPO,
            subfolder="direct",
            task_type=TaskType.CAUSAL_LM,
            adapter_name="direct",
        )
        for style in ["socratic", "scaffolding", "feedback"]:
            _peft_model.load_adapter(PEFT_REPO, adapter_name=style, subfolder=style)
        _use_peft_adapters = True
        run_model = _peft_model
        _log(f"✅ 14B {quant} Multi-LoRA 로드 (어댑터: marimmo/multi-lora direct/socratic/scaffolding/feedback)")
    except Exception as e1:
        _log(f"ℹ️ marimmo/multi-lora 서브폴더 로드 실패: {e1}")
        try:
            _peft_model = PeftModel.from_pretrained(
                model,
                STYLE_MODELS["direct"],
                task_type=TaskType.CAUSAL_LM,
                adapter_name="direct",
            )
            for style, path in list(STYLE_MODELS.items())[1:]:
                _peft_model.add_adapter(path, adapter_name=style)
            _use_peft_adapters = True
            run_model = _peft_model
            _log(f"✅ 14B {quant} Multi-LoRA 로드 (어댑터: {list(STYLE_MODELS)})")
        except Exception as e2:
            err_msg = str(e2)
            if "adapter_config" not in err_msg and "Entry Not Found" not in err_msg and "EntryNotFoundError" not in type(e2).__name__:
                raise
            _log(f"⚠️ PEFT 어댑터 없음 → {STYLE_MODELS['direct']} 전체 모델로 로드")
            del model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                STYLE_MODELS["direct"],
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                low_cpu_mem_usage=True,
            )
            _tokenizer = AutoTokenizer.from_pretrained(STYLE_MODELS["direct"])
            _tokenizer.pad_token = _tokenizer.eos_token
            _peft_model = None
            _use_peft_adapters = False
            run_model = model
            _log(f"✅ 14B {quant} 전체 모델 로드: {STYLE_MODELS['direct']}")

    if run_model is None:
        raise RuntimeError("PEFT/전체 모델 로드 실패")

    pipe = pipeline(
        "text-generation",
        model=run_model,
        tokenizer=_tokenizer,
        max_new_tokens=384,
        temperature=0.7,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id,
    )
    _rag_llm = HuggingFacePipeline(pipeline=pipe)
    torch.cuda.empty_cache()

def route_style(question: str) -> str:
    """쿼리를 임베딩하고 가장 가까운 스타일 centroid 선택"""
    if not LORA_ROUTER_PATH.exists():
        _log("⚠️ router_model.json 없음 → direct 기본")
        return "direct"
    
    try:
        with open(LORA_ROUTER_PATH, "r") as f:
            data = json.load(f)
        centroids = {
            style: np.array(vec, dtype=np.float32)
            for style, vec in data.get("centroids", {}).items()
        }
    except Exception as e:
        _log(f"⚠️ centroids 로드 실패: {e}")
        return "direct"
    
    if not centroids or _embeddings is None:
        return "direct"
    
    query_emb = np.array(_embeddings.embed_query(question), dtype=np.float32)
    
    best_style, best_sim = "direct", -1.0
    for style, centroid in centroids.items():
        sim = np.dot(query_emb, centroid) / (
            np.linalg.norm(query_emb) * np.linalg.norm(centroid) + 1e-9
        )
        if sim > best_sim:
            best_sim = sim
            best_style = style
    
    _log(f"📊 centroids 유사도: {best_style}={best_sim:.3f}")
    return best_style

def initialize(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_MAIN,
    k: int = 10,
):
    """
    전체 파이프라인 초기화 — 최초 1 회만 실행.

    순서: HF 로그인 → 임베딩 → 라우팅 LLM → PEFT(RAG 답변) LLM → RAG 체인.

    OOM 참고: LLM_MODE=vessel 이면 라우팅용 ROUTER_MODEL(14B)과 답변용 CHAIN_MODEL(14B)을
    둘 다 GPU에 올리므로 24GB 한 장으로는 부족할 수 있음. 단일 GPU 시 라우팅만 API로 두고
    LLM_MODE=api 권장 (답변용 PEFT 14B만 로컬).
    """
    global _initialized
    if _initialized:
        return

    _init_hf_login()
    _init_embeddings()
    # 라우팅/판단/요약용: vessel이면 동일 GPU에 14B 추가 로드 → 단일 24GB에서 PEFT 14B와 함께 OOM 가능
    if (os.getenv("LLM_MODE") or "").strip().lower() == "vessel":
        _log("ℹ️ LLM_MODE=vessel: 라우팅용 14B도 GPU 로드. 단일 24GB GPU면 OOM 시 LLM_MODE=api 로 라우팅만 API 사용 권장.")
    _init_chat_model()
    _init_peft_model()
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
    특수 토큰이 섞여있으면 제거하여 순수 대화 내용만 남긴다.
    """
    if hasattr(msg, "type") and hasattr(msg, "content"):
        return f"{msg.type}: {_strip_chat_tokens(str(msg.content))}"
    if isinstance(msg, (tuple, list)) and len(msg) >= 2:
        return f"{msg[0]}: {_strip_chat_tokens(str(msg[1]))}"
    return _strip_chat_tokens(str(msg))


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

def _strip_chat_tokens(text: str) -> str:
    """ChatHuggingFace 로컬 모델이 출력에 포함시키는 특수 토큰·시스템 프롬프트를 제거"""
    import re as _re
    last_assistant = text.rfind("<|im_start|>assistant")
    if last_assistant != -1:
        text = text[last_assistant + len("<|im_start|>assistant"):]
    text = _re.sub(r"<\|im_start\|>\s*(system|user|assistant)", "", text)
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


def _clean_answer_for_display(raw: str) -> str:
    """
    llm_answer 최종 답변에서 사용자에게 보여주면 안 되는 내용 제거.
    - 역할 레이블(system, user, assistant, Context:, Policy:, History:)
    - 웹 검색 출처(출처: http...)
    - 프롬프트가 그대로 에코된 블록
    """
    if not (raw or "").strip():
        return ""
    text = _strip_chat_tokens(raw)
    text = re.sub(r"<\|im_[a-z_]+\|>", "", text).strip()
    # 출처 URL 제거 (줄 단위 또는 문장 중간)
    text = re.sub(r"\n\s*출처:\s*https?://[^\n]+", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*출처:\s*https?://[^\n]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*출처:\s*https?://\S+", "", text, flags=re.IGNORECASE)
    # 라인 단위로 역할/메타 라벨 제거 (줄 시작이 system, Context:, Policy:, History:, user, assistant 인 경우)
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if re.match(r"^(system|Context:\s*$|Policy:\s*$|History:\s*$)", s, re.IGNORECASE):
            continue
        if re.match(r"^(user|assistant)\s*$", s, re.IGNORECASE):
            continue
        if s.lower().startswith("context:") and len(s) < 80:
            continue
        if s.lower().startswith("policy:") and len(s) < 80:
            continue
        if s.lower().startswith("history:") and len(s) < 80:
            continue
        if s.lower().startswith("user "):
            s = s[5:].strip()
        if s.lower().startswith("assistant "):
            s = s[9:].strip()
        lines.append(line)
    text = "\n".join(lines)
    # 마지막 'assistant ' 블록만 남기기 (모델이 전체 대화를 에코한 경우)
    if "\nassistant " in text or "\nuser " in text:
        parts = re.split(r"\n(?:user|assistant)\s+", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()
    # 앞뒤 공백·과도한 줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _invoke_clean(prompt) -> str:
    """_chat_hf.invoke() 호출 후 특수 토큰/시스템 프롬프트를 제거한 순수 텍스트 반환"""
    resp = _chat_hf.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    return _strip_chat_tokens(raw)

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
        text = _invoke_clean(prompt)
        return text[:max_chars]
    except Exception:
        return content[:max_chars] + "..."


# ============================================================
# 11. 노드 함수
# ============================================================


def _timed_node(node_func, node_name: str):
    """
    노드 진입 시·퇴장 시 시간을 측정하는 래퍼.
    진입 시 t0 기록 → 원본 노드 실행 → 퇴장 시 소요 시간 로그.
    """
    def wrapped(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        _log(f"⏱️ [{node_name}] 진입 @ {t0:.3f}s")
        try:
            out = node_func(state)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            _log(f"⏱️ [{node_name}] 퇴장 @ {t1:.3f}s (소요: {elapsed:.3f}s)")
            return out
        except Exception as e:
            t1 = time.perf_counter()
            _log(f"⏱️ [{node_name}] 예외로 퇴장 @ {t1:.3f}s (소요: {(t1 - t0):.3f}s) — {e}")
            raise
    return wrapped


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
        text = _invoke_clean(judge_prompt).upper()
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
            cand = _invoke_clean(rq_prompt)
            if cand:
                retrieval_query = cand
        except Exception:
            pass

        docs = summary_store.similarity_search(retrieval_query, k=3)
        _log(f"retrieval_query: {retrieval_query}")
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
            cand = _invoke_clean(rewrite_prompt)
            if cand:
                rewrite_question = cand
        except Exception:
            rewrite_question = question

        _log(f"재작성된 쿼리: {rewrite_question}")

    return GraphState(question=rewrite_question)


def retrieve(state: GraphState) -> GraphState:
    """
    [retrieve 노드]
    ChromaDB retriever 로 사용자 질문과 관련된 문서를 검색.
    """
    docs = _retriever.invoke(state["question"])
    return GraphState(context=format_docs(docs))


def _format_chat_history_for_prompt(messages: list, max_turns: int = MAX_HISTORY_TURNS, max_chars: int = MAX_HISTORY_CHARS) -> str:
    """
    대화 이력을 프롬프트용 문자열로 변환. 턴 수·문자 수 제한 적용.
    - max_turns: 최근 N턴(2N개 메시지)만 사용
    - max_chars: 전체 히스토리 문자열이 이 길이를 넘으면 앞(오래된 턴)부터 생략
    """
    conv = _conversation_only(messages)
    if not conv:
        return ""
    # 최근 2*max_turns개 메시지 = max_turns 턴
    recent = conv[-(2 * max_turns) :] if len(conv) >= 2 * max_turns else conv
    # user/assistant 쌍으로 포맷 (홀수 개면 마지막 메시지 제외)
    pair_count = len(recent) // 2
    if pair_count == 0:
        return ""
    lines = []
    for i in range(pair_count):
        user_content = (recent[i * 2][1] or "").strip()
        asst_content = (recent[i * 2 + 1][1] or "").strip()
        lines.append(f"User: {user_content}\nAssistant: {asst_content}")
    history_str = "\n\n".join(lines)
    if len(history_str) <= max_chars:
        return history_str
    # 문자 수 초과 시 앞쪽 턴부터 제거 (한 턴 단위로)
    while len(history_str) > max_chars and len(lines) > 1:
        lines.pop(0)
        history_str = "\n\n".join(lines)
    return history_str


def llm_answer(state: GraphState) -> GraphState:
    question = state["question"]
    context = state.get("context", "")
    chat_history = state.get("messages", [])
    policy = state.get("policy", "")

    style = route_style(question)  # centroids → direct/socratic/scaffolding/feedback
    _log(f"🎯 LoRA 라우팅: {style}")
    # PEFT 멀티 어댑터일 때만 쿼리별 어댑터 스위칭 (전체 모델 단일 로드 시 무시)
    if _peft_model is not None:
        try:
            _peft_model.set_adapter(style)
        except (ValueError, KeyError) as e:
            _log(f"⚠️ 어댑터 '{style}' 적용 실패 ({e}) → direct 사용")
            style = "direct"
            _peft_model.set_adapter(style)

    history_str = _format_chat_history_for_prompt(chat_history)
    if history_str:
        history_block = f"History:\n{history_str}\n\n"
    else:
        history_block = ""

    prompt = f"""<|im_start|>system
You are a helpful tutor. Use the context below only to inform your answer. Do not repeat or output the words "system", "Context:", "Policy:", "History:", "user", "assistant" or any URLs. Reply only with the assistant's answer in natural language.
{history_block}Context: {context}
Policy: {policy}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    try:
        response = _rag_llm.invoke(prompt).strip()
        response = _clean_answer_for_display(response)
    except Exception as e:
        _log(f"❌ 생성 실패: {e}")
        response = "생성 중 오류 발생."

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

    text = _invoke_clean(prompt)

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
    tavily = TavilySearch(max_results=5, search_depth="basic")
    query_text = state["question"]
    try:
        results = tavily.invoke(query_text)
    except Exception as e:
        _log(f"⚠️ 웹 검색 실패: {e}")
        return GraphState(context="웹 검색 결과를 가져오지 못했습니다.")

    # TavilySearch는 버전에 따라 list / dict / str 을 반환할 수 있음
    if isinstance(results, dict):
        results = results.get("results", [results])
    if isinstance(results, str):
        results = [{"content": results}]
    if not isinstance(results, list):
        _log(f"⚠️ 웹 검색 결과 파싱 불가: {type(results)}")
        return GraphState(context="웹 검색 결과를 가져오지 못했습니다.")

    parts = []
    for r in results:
        if isinstance(r, dict):
            url = r.get("url", "")
            content = _summarize_if_long(r.get("content", ""))
            parts.append(f"{content}\n출처: {url}" if url else content)
        else:
            parts.append(_summarize_if_long(str(r)))
    formatted = "\n\n---\n\n".join(parts)

    # ChromaDB 에도 저장 (비동기 — 노드 반환을 블로킹하지 않음)
    if formatted.strip():
        doc = Document(
            page_content=formatted,
            metadata={
                "source": f"web_search:{query_text}",
                "origin": "tavily_merged",
            },
        )

        def _bg_save_web():
            try:
                _raw_ingest_docs(
                    documents=[doc],
                    persist_directory=PERSIST_DIR,
                    collection_name=COLLECTION_MAIN,
                )
                _log("✅ 웹 검색 결과 ChromaDB 저장 완료 (bg)")
            except Exception as e:
                _log(f"⚠️ 웹 검색 결과 저장 실패 (bg): {e}")

        _bg_executor.submit(_bg_save_web)

    return GraphState(context=formatted)


def save_memory(state: GraphState) -> GraphState:
    """
    [save_memory 노드]
    10턴마다 실행되어:
      1) 오래된 대화 10턴(20개 메시지)을 raw 컬렉션에 저장
      2) 최근 대화를 분석하여 다음 10턴간의 답변 방향성(policy)을 생성
      3) 저장한 10턴은 state.messages에서 제거하여 이후 context/prompt에 포함되지 않도록 함
    """
    messages = state.get("messages", [])
    conv = _conversation_only(messages)
    MIN_MSGS = 20  # 10 턴 = 20 메시지

    if len(conv) < MIN_MSGS:
        _log(f"ℹ️ save_memory 건너뜀: 대화 {len(conv)}개 (< {MIN_MSGS})")
        return {}

    oldest = conv[:MIN_MSGS]
    raw_text = "\n".join(f"{r}: {c}" for r, c in oldest).strip()
    if not raw_text:
        return {}

    ts = datetime.now(timezone.utc).isoformat()
    mem_id = uuid.uuid4().hex

    # ── raw 대화 저장 (10턴) ──
    raw_doc = Document(
        page_content=raw_text,
        metadata={
            "source": "chat_history_raw",
            "memory_id": mem_id,
            "saved_at": ts,
            "turn_count": 10,
            "message_count": MIN_MSGS,
        },
    )

    delete_messages = [RemoveMessage(id=msg.id) for msg in messages[:MIN_MSGS]]
    _log(f"🧹 메시지 정리: {MIN_MSGS}개 제거")
    
    return {
        "policy": policy_text,
        "messages": delete_messages  # add_messages 리듀서가 자동 처리
    }

    def _bg_save_raw():
        try:
            _raw_ingest_docs(
                documents=[raw_doc],
                persist_directory=PERSIST_DIR,
                collection_name=COLLECTION_CHAT_RAW,
                chunk_size=1200,
                chunk_overlap=120,
            )
            _log("✅ save_memory raw 저장 완료 (bg)")
        except Exception as e:
            _log(f"⚠️ save_memory raw 저장 실패 (bg): {e}")

    _bg_executor.submit(_bg_save_raw)

    # ── policy 생성 (최근 대화 기반) ──
    recent = conv[-20:]  # 최근 10턴
    conv_text = "\n".join(f"{r}: {c}" for r, c in recent).strip()

    policy_prompt = f"""당신은 학습 튜터의 교육 전략 분석가입니다.
아래 학생과 튜터의 최근 대화 내용을 분석하여, 앞으로 10턴 동안 튜터가 취해야 할 답변 방향성(policy)을 결정하세요.

[최근 대화 내용]
{conv_text}

아래 보기 중에서 학생에게 가장 적합한 방향성을 1~2개 선택하고, 해당 형식 그대로 출력하세요.
여러 개 선택 시 줄바꿈으로 구분합니다.

- 개념 이해 부족 -> 예시를 통한 개념 설명
- 응용능력 부족 -> 유사 문제 추천
- 암기 능력 강화 -> 앞글자를 따 암기방식 추천
- 개념 간 연결 부족 -> 연관 개념 및 비교 설명
- 자주 틀리는 유형 -> 오답 분석 및 반복 학습 유도
- 심화 학습 필요 -> 난이도 높은 질문 유도
- 기초 부족 -> 선수 개념부터 단계적 설명

방향성만 출력하세요. 다른 설명은 불필요합니다.""".strip()

    try:
        policy_text = _invoke_clean(policy_prompt)
    except Exception as e:
        _log(f"⚠️ policy 생성 실패: {e}")
        policy_text = state.get("policy", "")

    _log(f"📋 policy 갱신: {policy_text}")

    # ── 저장한 10턴(20개 메시지)을 state.messages에서 제거 ──
    # add_messages 리듀서: REMOVE_ALL 후 남길 메시지만 다시 넣으면, 이후 context/prompt에 저장분이 포함되지 않음
    remaining = list(messages[MIN_MSGS:])
    new_messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + remaining
    _log(f"🧹 메시지 정리: 저장한 {MIN_MSGS}개 제거, {len(remaining)}개만 유지")

    return {"policy": policy_text, "messages": new_messages}


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
        text = _invoke_clean(prompt)
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
    """대화가 10턴 단위(20 메시지)일 때 save_memory 로 분기하여 policy 갱신"""
    conv = _conversation_only(state.get("messages", []))
    turn_count = len(conv) // 2
    if turn_count > 0 and turn_count % 10 == 0:
        return "save_chat"
    return "too short"


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

    # # ── 노드 등록 ──
    workflow.add_node("contextualize", _timed_node(contextualize, "contextualize"))
    workflow.add_node("save_memory", _timed_node(save_memory, "save_memory"))
    workflow.add_node("retrieve", _timed_node(retrieve, "retrieve"))
    workflow.add_node("llm_answer", _timed_node(llm_answer, "llm_answer"))
    workflow.add_node("relevance_check", _timed_node(relevance_check, "relevance_check"))
    workflow.add_node("web_search", _timed_node(web_search, "web_search"))

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


def verify_chain_and_lora_config() -> Dict[str, Any]:
    """
    체인 생성·LoRA 라우팅 설정이 정상인지 검증 (모델 로드 없이 설정만 검사).
    - router_model.json centroids 키와 STYLE_MODELS 키 일치 여부
    - 어댑터 이름 일치 시 set_adapter(style) 정상 동작 가능 여부
    Returns:
        {"ok": bool, "checks": {...}, "errors": [...]}
    """
    result = {"ok": True, "checks": {}, "errors": []}

    # 1) STYLE_MODELS 키 = 라우터/어댑터에서 쓰는 이름
    style_keys = set(STYLE_MODELS.keys())
    result["checks"]["style_keys"] = list(style_keys)
    if "direct" not in style_keys:
        result["ok"] = False
        result["errors"].append("STYLE_MODELS에 'direct'가 없음")

    # 2) router_model.json 존재 및 centroids 키와 STYLE_MODELS 일치
    if not LORA_ROUTER_PATH.exists():
        result["checks"]["router_file"] = "없음 (route_style은 direct 반환)"
    else:
        try:
            with open(LORA_ROUTER_PATH, "r") as f:
                data = json.load(f)
            centroids = data.get("centroids") or data.get("styles", [])
            if isinstance(centroids, dict):
                centroid_keys = set(centroids.keys())
            else:
                centroid_keys = set(centroids) if isinstance(centroids, list) else set()
            result["checks"]["centroid_keys"] = list(centroid_keys)
            missing_in_router = style_keys - centroid_keys
            if missing_in_router:
                result["ok"] = False
                result["errors"].append(f"router_model에 없는 스타일: {missing_in_router}")
            extra = centroid_keys - style_keys
            if extra:
                result["checks"]["extra_in_router"] = list(extra)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"router_model.json 로드 실패: {e}")

    # 3) 초기화 시 첫 어댑터 이름이 "direct"로 로드되므로 route_style("direct") → set_adapter("direct") 일치
    result["checks"]["adapter_name_note"] = "PeftModel.from_pretrained(..., adapter_name='direct') 사용으로 direct 일치"

    return result


if __name__ == "__main__":
    import sys
    r = verify_chain_and_lora_config()
    print("체인·LoRA 설정 검증:", "OK" if r["ok"] else "실패")
    for k, v in r["checks"].items():
        print(f"  {k}: {v}")
    if r["errors"]:
        for e in r["errors"]:
            print("  오류:", e)
        sys.exit(1)
    print("(실제 체인 생성·어댑터 적용은 initialize() → build_app() → query() 호출 시 수행)")
