/**
 * ToastContainer – 팝업 토스트 메시지
 *
 * 서버의 내부 로그(파일 업로드 완료, 체인 생성 완료 등)를
 * 화면 우측 상단에 1 초간 표시 후 서서히 사라지게 합니다.
 *
 * Props:
 *   toasts : [{ id, msg }, …]
 */
export default function ToastContainer({ toasts }) {
  if (!toasts.length) return null;

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className="toast">
          {t.msg}
        </div>
      ))}
    </div>
  );
}
