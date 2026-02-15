import { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import ToastContainer from "./components/Toast";

/**
 * App – 최상위 레이아웃
 *
 * ┌─────────┬──────────────────────────┐
 * │ Sidebar │       ChatArea           │
 * │         │  messages …              │
 * │ Debug   │  [+] input …      [Send] │
 * │ Docs    │                          │
 * └─────────┴──────────────────────────┘
 */
export default function App() {
  // ── 상태 ──
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // 시스템 상태 (디버깅 패널용)
  const [status, setStatus] = useState(null);
  const [documents, setDocuments] = useState([]);

  // ── 토스트 헬퍼 ──
  const addToast = useCallback((msg) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, msg }]);
    // 2.5초 후 자동 제거 (CSS 페이드 1s + 여유)
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 2500);
  }, []);

  // 서버 로그 배열 → 토스트로 변환
  const showLogs = useCallback(
    (logs) => {
      if (!logs) return;
      logs.forEach((l) => addToast(l));
    },
    [addToast]
  );

  // ── 시스템 상태 조회 ──
  const fetchStatus = useCallback(async () => {
    try {
      const [sRes, dRes] = await Promise.all([
        fetch("/api/status"),
        fetch("/api/documents"),
      ]);
      if (sRes.ok) setStatus(await sRes.json());
      if (dRes.ok) {
        const d = await dRes.json();
        setDocuments(d.collections || []);
      }
    } catch {
      /* 서버 미실행 */
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // ── 메시지 전송 ──
  const sendMessage = useCallback(
    async (text) => {
      if (!text.trim() || loading) return;

      // 사용자 메시지 추가
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setLoading(true);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, thread_id: threadId }),
        });
        const data = await res.json();

        // thread_id 기억
        if (data.thread_id) setThreadId(data.thread_id);

        // 어시스턴트 답변 추가
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.answer },
        ]);

        // 서버 로그 토스트
        showLogs(data.logs);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `오류 발생: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, threadId, showLogs]
  );

  // ── 파일 업로드 ──
  const uploadFile = useCallback(
    async (file) => {
      const form = new FormData();
      form.append("file", file);

      addToast(`📤 ${file.name} 업로드 중…`);

      try {
        const res = await fetch("/api/upload", { method: "POST", body: form });
        const data = await res.json();

        if (data.status === "ok") {
          addToast(`✅ ${file.name} 적재 완료`);
        } else {
          addToast(`❌ ${file.name} 적재 실패: ${data.error}`);
        }
        showLogs(data.logs);
        // 문서 목록 갱신
        fetchStatus();
      } catch (err) {
        addToast(`❌ 업로드 실패: ${err.message}`);
      }
    },
    [addToast, showLogs, fetchStatus]
  );

  // ── 대화 초기화 ──
  const resetChat = useCallback(async () => {
    try {
      const res = await fetch("/api/reset", { method: "POST" });
      const data = await res.json();
      setThreadId(data.thread_id);
    } catch {
      setThreadId(null);
    }
    setMessages([]);
    addToast("🗑️ 대화가 초기화되었습니다.");
  }, [addToast]);

  // ── 렌더링 ──
  return (
    <div className="app">
      {/* 사이드바 토글 (모바일) */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen((o) => !o)}
        aria-label="사이드바 토글"
      >
        ☰
      </button>

      <Sidebar
        open={sidebarOpen}
        status={status}
        documents={documents}
        onReset={resetChat}
        onClose={() => setSidebarOpen(false)}
      />

      <ChatArea
        messages={messages}
        loading={loading}
        onSend={sendMessage}
        onUpload={uploadFile}
      />

      <ToastContainer toasts={toasts} />
    </div>
  );
}
