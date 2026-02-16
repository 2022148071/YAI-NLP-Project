/**
 * Sidebar – 디버깅 패널 + 저장 문서 목록
 *
 * Props:
 *   open      : boolean – 사이드바 열림 여부
 *   status    : object  – { models, connections }
 *   documents : array   – [{ name, count }, …]
 *   onReset   : fn      – 대화 초기화 콜백
 *   onClose   : fn      – 사이드바 닫기 (모바일)
 */
export default function Sidebar({ open, status, documents, onReset, onClose }) {
  const models = status?.models || {};
  const conn = status?.connections || {};

  return (
    <aside className={`sidebar${open ? "" : " closed"}`}>
      {/* ── 헤더 ── */}
      <div className="sidebar-header">
        <h2>🧭 NeuLoRA</h2>
        <p>The Thinking tutor that connects to your neuron</p>
        {/* <button className="sidebar-close" onClick={onClose}>
          ✕
        </button> */}
      </div>

      {/* ── 디버깅 : 모델 정보 ── */}
      <div className="sidebar-section">
        <h3>🔧 모델 정보</h3>
        <div className="model-info">
          <strong>답변 LLM</strong>
          <br />
          <code>{models.answer_llm || "–"}</code>
        </div>
        <div className="model-info">
          <strong>라우팅 LLM</strong>
          <br />
          <code>{models.router_llm || "–"}</code>
        </div>
        <div className="model-info">
          <strong>임베딩</strong>
          <br />
          <code>{models.embedding || "–"}</code>
        </div>
      </div>

      {/* ── 디버깅 : 연결 상태 ── */}
      <div className="sidebar-section">
        <h3>📡 연결 상태</h3>
        <StatusRow label="HuggingFace API" ok={conn.huggingface} />
        <StatusRow label="Tavily Search API" ok={conn.tavily} />
        <StatusRow label="ChromaDB 저장소" ok={conn.chromadb} />
        <StatusRow label="파이프라인" ok={conn.pipeline} />
      </div>

      {/* ── 저장된 문서 ── */}
      <div className="sidebar-section">
        <h3>📚 저장된 문서</h3>
        {documents.length > 0 ? (
          documents.map((d) => (
            <div className="doc-item" key={d.name}>
              <span className="doc-name">{d.name}</span>
              <span className="doc-count">{d.count}개</span>
            </div>
          ))
        ) : (
          <p className="empty-text">컬렉션 없음</p>
        )}
      </div>

      {/* ── 대화 초기화 ── */}
      <div className="sidebar-footer">
        <button className="reset-btn" onClick={onReset}>
          🗑️ 대화 초기화
        </button>
      </div>
    </aside>
  );
}

/* 초록/빨강 상태 인디케이터 행 */
function StatusRow({ label, ok }) {
  return (
    <div className="status-row">
      <span className={`dot ${ok ? "green" : "red"}`} />
      <span>
        {label} {ok ? "" : "(미연결)"}
      </span>
    </div>
  );
}
