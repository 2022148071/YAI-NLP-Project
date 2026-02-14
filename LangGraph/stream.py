"""
stream.py - Streamlit 기반 LangGraph RAG 데모
=============================================

실행 방법:
    cd LangGraph
    streamlit run stream.py

기능:
  1. 멀티턴 대화 인터페이스
  2. PDF / TXT 파일 업로드 → ChromaDB 적재  (➕ 버튼 + 사이드바)
  3. 사이드바: 디버깅 패널(모델명·연결 상태), 저장 문서 목록
  4. 내부 로그 → 팝업 토스트 메시지 (1 초 후 소멸)
"""

import os
import sys
import tempfile
from pathlib import Path

# ── 프로젝트 루트 path 설정 (rag 패키지 import 용) ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# ── LangGraph 파이프라인 모듈 ──
import LangGraph as lg

# ============================================================
# 페이지 설정 (가장 먼저 호출해야 함)
# ============================================================
st.set_page_config(
    page_title="RAG Chat – LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 커스텀 CSS
# ============================================================
st.markdown(
    """
<style>
/* ── 상태 인디케이터 (초록/빨강 동그라미) ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
    font-size: 0.9rem;
}
.dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.dot-green { background-color: #2ecc71; }
.dot-red   { background-color: #e74c3c; }

/* ── 채팅 영역 여백 ── */
.block-container { padding-top: 1rem; padding-bottom: 0; }

/* ── 팝오버 최소 너비 ── */
.stPopover > div { min-width: 340px; }

/* ── 사이드바 구분선 ── */
.sidebar-divider { margin: 12px 0; border-top: 1px solid #444; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 세션 상태 초기화
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # 화면 표시용 [{role, content}, …]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = lg.random_uuid()
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

# ============================================================
# 파이프라인 초기화 (세션당 최초 1 회, 캐시)
# ============================================================


@st.cache_resource(show_spinner=False)
def _init_pipeline():
    """
    LangGraph 파이프라인을 전체 초기화하고 앱을 컴파일.
    @st.cache_resource 로 서버 프로세스 내 1 회만 실행.
    """
    try:
        lg.initialize()
        app = lg.build_app()
        return app, None
    except Exception as e:
        return None, str(e)


if not st.session_state.pipeline_ready:
    with st.spinner("🔧 파이프라인 초기화 중… (최초 1 회, 30 초~1 분 소요)"):
        _app, _err = _init_pipeline()
    if _err:
        st.error(f"파이프라인 초기화 실패: {_err}")
        st.stop()
    else:
        st.session_state.pipeline_ready = True
        # 초기화 과정 로그를 토스트로 표시
        for msg in lg.get_and_clear_logs():
            st.toast(msg, icon="✅")


# ============================================================
# 유틸리티 함수
# ============================================================


def _status_dot(ok: bool) -> str:
    """연결 상태 HTML 인디케이터"""
    cls = "dot-green" if ok else "dot-red"
    return f'<span class="dot {cls}"></span>'


def _process_uploaded_files(files):
    """업로드된 파일 리스트를 ChromaDB 에 적재"""
    for uf in files:
        suffix = Path(uf.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            lg.ingest_uploaded_file(tmp_path)
            st.toast(f"✅ {uf.name} 적재 완료", icon="📄")
        except Exception as e:
            st.toast(f"❌ {uf.name} 적재 실패: {e}", icon="⚠️")
        finally:
            os.unlink(tmp_path)
    # 적재 과정 로그도 토스트
    for log_msg in lg.get_and_clear_logs():
        st.toast(log_msg, icon="📄")


# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.markdown("## 🧭 RAG Chat")
    st.caption("LangGraph 기반 멀티턴 RAG 챗봇")
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── 🔧 디버깅 패널 ────────────────────────────────────────
    with st.expander("🔧 디버깅", expanded=True):
        st.markdown(f"**답변 LLM**  \n`{lg.CHAIN_MODEL}`")
        st.markdown(f"**라우팅 LLM**  \n`{lg.ROUTER_MODEL}`")
        st.markdown(f"**임베딩 모델**  \n`{lg.EMBEDDING_MODEL}`")
        st.markdown("---")

        # 연결 상태 체크
        hf_ok = bool(os.getenv("HF_API_KEY"))
        tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
        chroma_ok = Path(lg.PERSIST_DIR).exists()
        pipe_ok = lg.is_initialized()

        st.markdown(
            f"""
{_status_dot(hf_ok)} **HuggingFace API** {'연결됨' if hf_ok else '키 없음'}

{_status_dot(tavily_ok)} **Tavily Search API** {'연결됨' if tavily_ok else '키 없음'}

{_status_dot(chroma_ok)} **ChromaDB 저장소** {'존재' if chroma_ok else '없음'}

{_status_dot(pipe_ok)} **파이프라인** {'준비 완료' if pipe_ok else '미초기화'}
""",
            unsafe_allow_html=True,
        )

    # ── 📚 저장된 문서 보기 ──────────────────────────────────
    with st.expander("📚 저장된 문서", expanded=False):
        if chroma_ok:
            try:
                import chromadb

                client = chromadb.PersistentClient(path=lg.PERSIST_DIR)
                collections = client.list_collections()
                if collections:
                    for col in collections:
                        count = col.count()
                        st.markdown(f"- **{col.name}** : `{count}` 개 문서")
                else:
                    st.info("컬렉션이 아직 없습니다.")
            except Exception as e:
                st.warning(f"ChromaDB 조회 실패: {e}")
        else:
            st.info("ChromaDB 디렉토리가 존재하지 않습니다.")

    # ── 📁 파일 업로드 (사이드바) ────────────────────────────
    with st.expander("📁 파일 업로드", expanded=False):
        sidebar_files = st.file_uploader(
            "PDF 또는 TXT 파일을 드래그앤드롭하세요",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="sidebar_uploader",
        )
        if sidebar_files:
            if st.button("📤 적재 시작", key="sidebar_ingest_btn"):
                with st.spinner("적재 중…"):
                    _process_uploaded_files(sidebar_files)
                st.rerun()

    # ── 대화 초기화 ──────────────────────────────────────────
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = lg.random_uuid()
        st.toast("대화가 초기화되었습니다.", icon="🗑️")
        st.rerun()

# ============================================================
# 메인 채팅 영역
# ============================================================
st.markdown("## 💬 RAG Chat")

# ── 기존 대화 메시지 표시 ────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── ➕ 파일 첨부 팝오버 (채팅 입력 근처) ─────────────────────
with st.popover("➕ 파일 첨부"):
    st.markdown("**PDF 또는 TXT 파일을 드래그앤드롭하세요**")
    quick_files = st.file_uploader(
        "파일 선택",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="quick_uploader",
        label_visibility="collapsed",
    )
    if quick_files:
        if st.button("📤 적재", key="quick_ingest_btn"):
            _process_uploaded_files(quick_files)

# ── 채팅 입력 ────────────────────────────────────────────────
if user_input := st.chat_input("메시지를 입력하세요…"):
    if not st.session_state.pipeline_ready:
        st.error("파이프라인이 초기화되지 않았습니다. 페이지를 새로고침하세요.")
    else:
        # 사용자 메시지 저장 & 표시
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── 그래프 실행 ──
        with st.chat_message("assistant"):
            with st.spinner("생각 중…"):
                try:
                    result = lg.query(
                        user_input,
                        thread_id=st.session_state.thread_id,
                    )
                    answer = result.get("answer", "답변을 생성하지 못했습니다.")
                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"

            st.markdown(answer)

        # 답변 저장
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # ── 내부 로그 → 토스트 팝업 ──
        for log_msg in lg.get_and_clear_logs():
            st.toast(log_msg, icon="ℹ️")
