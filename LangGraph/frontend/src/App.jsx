import { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
// import ToastContainer from "./components/Toast";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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
  // const [toasts, setToasts] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // 시스템 상태 (디버깅 패널용)
  const [status, setStatus] = useState(null);
  const [documents, setDocuments] = useState([]);

  // 어시스턴트 답변을 점진 출력(스트리밍 UI)
  const streamAssistantText = useCallback(async (text) => {
    const safeText = String(text ?? "");
    const chars = Array.from(safeText);
    const total = chars.length;
    const chunkSize = total > 500 ? 8 : total > 250 ? 5 : 2;

    setMessages((prev) => [...prev, { role: "assistant", content: "", streaming: true }]);

    for (let i = 0; i < total; i += chunkSize) {
      const partial = chars.slice(0, i + chunkSize).join("");
      setMessages((prev) => {
        const next = [...prev];
        const lastIdx = next.length - 1;
        if (lastIdx >= 0 && next[lastIdx].role === "assistant") {
          next[lastIdx] = { ...next[lastIdx], content: partial, streaming: true };
        }
        return next;
      });
      await sleep(16);
    }

    setMessages((prev) => {
      const next = [...prev];
      const lastIdx = next.length - 1;
      if (lastIdx >= 0 && next[lastIdx].role === "assistant") {
        next[lastIdx] = { ...next[lastIdx], streaming: false };
      }
      return next;
    });
  }, []);

  // ── 토스트 헬퍼 (토스트 끄기: 아래 주석 해제하고, 맨 아래 ToastContainer·호출부도 주석 해제) ──
  // const [toasts, setToasts] = useState([]);  ← 상태는 위에서 이미 주석됨
  // const addToast = useCallback((msg) => {
  //   const id = Date.now() + Math.random();
  //   setToasts((prev) => [...prev, { id, msg }]);
  //   setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 2500);
  // }, []);
  // const showLogs = useCallback((logs) => { if (!logs) return; logs.forEach((l) => addToast(l)); }, [addToast]);
  const addToast = useCallback(() => {}, []);
  const showLogs = useCallback(() => {}, []);

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

        // 어시스턴트 답변 스트리밍 출력
        await streamAssistantText(data.answer);

        // 서버 로그 토스트 (토스트 끄면 주석)
        // showLogs(data.logs);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `오류 발생: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, threadId, streamAssistantText]
  );

  // ── 파일 업로드 ──
  const uploadFile = useCallback(
    async (file) => {
      const form = new FormData();
      form.append("file", file);

      // addToast(`📤 ${file.name} 업로드 중…`);

      try {
        const res = await fetch("/api/upload", { method: "POST", body: form });
        const data = await res.json();

        if (data.status === "ok") {
          // addToast(`✅ ${file.name} 적재 완료`);
        } else {
          // addToast(`❌ ${file.name} 적재 실패: ${data.error}`);
        }
        // showLogs(data.logs);
        fetchStatus();
      } catch (err) {
        // addToast(`❌ 업로드 실패: ${err.message}`);
      }
    },
    [fetchStatus]
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
    // addToast("🗑️ 대화가 초기화되었습니다.");
  }, []);

  // ── 렌더링 ──
  return (
    <div className="app">
      {/* 사이드바 토글 */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen((o) => !o)}
        aria-label={sidebarOpen ? "사이드바 닫기" : "사이드바 열기"}
      >
        {sidebarOpen ? "✕" : "☰"}
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

      {/* 토스트 팝업 끄기: 위에서 toasts/addToast/showLogs 주석 해제하고 아래 주석 해제 */}
      {/* <ToastContainer toasts={toasts} /> */}
    </div>
  );
}
