import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";

function normalizeMathMarkdown(text) {
  const src = String(text ?? "");

  // 표준 LaTeX 블록 구분자: \[ ... \] -> $$ ... $$
  let out = src.replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_, expr) => {
    return `\n$$\n${expr.trim()}\n$$\n`;
  });

  // 비표준 수식 표기: [ ... ] -> $$ ... $$ (링크 [text](url) 는 제외)
  out = out.replace(/\[\s*([^\[\]\n]+?)\s*\](?!\()/g, (full, expr) => {
    const candidate = expr.trim();
    const looksLikeMath =
      /\\[a-zA-Z]+/.test(candidate) || /[=^_]/.test(candidate);
    if (!looksLikeMath) return full;
    const cleaned = candidate.replace(/\s*,\s*/g, " ");
    return `\n$$\n${cleaned}\n$$\n`;
  });

  return out;
}

/**
 * ChatArea – 채팅 메시지 + 입력 바 + 파일 업로드
 *
 * Props:
 *   messages : [{ role, content }, …]
 *   loading  : boolean
 *   onSend   : (text) => void
 *   onUpload : (file) => void
 */
export default function ChatArea({ messages, loading, onSend, onUpload }) {
  const [input, setInput] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [dragging, setDragging] = useState(false);
  const bottomRef = useRef(null);
  const fileRef = useRef(null);

  // 새 메시지 시 자동 스크롤
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // 전송
  const handleSend = () => {
    if (!input.trim()) return;
    onSend(input.trim());
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── 파일 업로드 핸들러 ──
  const handleFiles = useCallback(
    (fileList) => {
      const allowed = ["application/pdf", "text/plain"];
      Array.from(fileList).forEach((f) => {
        if (allowed.includes(f.type) || f.name.endsWith(".txt") || f.name.endsWith(".pdf")) {
          onUpload(f);
        }
      });
      setShowUpload(false);
    },
    [onUpload]
  );

  const onDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = () => setDragging(false);
  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };
  const onFileChange = (e) => {
    if (e.target.files.length) handleFiles(e.target.files);
  };

  const isStreamingAssistant =
    messages.length > 0 &&
    messages[messages.length - 1].role === "assistant" &&
    messages[messages.length - 1].streaming;

  return (
    <main className="chat-area">
      {/* ── 메시지 영역 ── */}
      <div className="messages">
        {messages.length === 0 && !loading ? (
          <div className="empty-chat">
            <span className="emoji">💬</span>
            <p>대화를 시작해보세요!</p>
            <p style={{ fontSize: "0.82rem" }}>
              PDF / TXT 파일을 업로드하고 질문할 수 있습니다.
            </p>
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`message-row ${m.role}`}>
              <div className={`avatar ${m.role}`}>
                {m.role === "user" ? "👤" : "🤖"}
              </div>
              <div className={`bubble ${m.role}`}>
                <div className="markdown-body">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {normalizeMathMarkdown(m.content)}
                  </ReactMarkdown>
                </div>
                {m.streaming ? "▌" : ""}
              </div>
            </div>
          ))
        )}

        {/* 로딩 애니메이션 */}
        {loading && !isStreamingAssistant && (
          <div className="message-row assistant">
            <div className="avatar assistant">🤖</div>
            <div className="bubble assistant">
              <div className="typing-indicator">
                <span />
                <span />
                <span />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── 입력 바 ── */}
      <div className="chat-input-wrapper">
        <div className="chat-input-bar">
          {/* 첨부 파일 + 버튼 (채팅 박스 내부 왼쪽) */}
          <button
            className="attach-btn"
            onClick={() => setShowUpload(true)}
            title="파일 첨부 (PDF, TXT)"
          >
            +
          </button>

          <input
            type="text"
            placeholder="메시지를 입력하세요…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />

          <button
            className="send-btn"
            onClick={handleSend}
            disabled={loading || !input.trim()}
            title="전송"
          >
            ➤
          </button>
        </div>
      </div>

      {/* ── 파일 업로드 오버레이 ── */}
      {showUpload && (
        <div className="upload-overlay" onClick={() => setShowUpload(false)}>
          <div className="upload-modal" onClick={(e) => e.stopPropagation()}>
            <h3>📄 파일 업로드</h3>

            <div
              className={`drop-zone${dragging ? " dragging" : ""}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
            >
              <div className="icon">📁</div>
              <p>PDF 또는 TXT 파일을 여기에 드래그앤드롭</p>
              <p className="hint">또는 클릭하여 파일 선택</p>
            </div>

            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.txt"
              multiple
              style={{ display: "none" }}
              onChange={onFileChange}
            />

            <button
              className="cancel-btn"
              onClick={() => setShowUpload(false)}
            >
              닫기
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
