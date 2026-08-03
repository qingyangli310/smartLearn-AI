import { useState } from "react";
import { askQuestion } from "./api.js";

function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    const question = message.trim();
    if (!question || !enabled || loading || disabled) return;

    setMessages((current) => [
      ...current,
      { role: "user", content: question },
    ]);
    setMessage("");
    setError("");
    setLoading(true);
    onBusy?.(true);

    try {
      const result = await askQuestion(question);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: result.answer,
          citations: result.citations || [],
          sources: result.sources || [],
        },
      ]);
    } catch (requestError) {
      setError(requestError.message || "Chat failed.");
    } finally {
      setLoading(false);
      onBusy?.(false);
    }
  };

  return (
    <section className="chat-panel" aria-label="Document chat">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Conversation</p>
          <h2>Ask the PDF</h2>
        </div>
        <span className="message-count">{messages.length} messages</span>
      </div>

      <div className="message-list" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty-state chat-empty">
            <p>{enabled ? "Ask a question about the uploaded PDF." : "Upload a PDF to begin."}</p>
          </div>
        )}

        {messages.map((item, index) => (
          <article className={`message message-${item.role}`} key={`${item.role}-${index}`}>
            <p className="message-role">{item.role === "user" ? "You" : "SmartLearn"}</p>
            <p>{item.content}</p>

            {item.role === "assistant" && item.citations?.length > 0 && (
              <div className="citation-list" aria-label="Citations">
                {item.citations.map((page) => (
                  <button
                    className="citation-button"
                    type="button"
                    key={page}
                    onClick={() => onJumpToPage(page)}
                  >
                    Page {page}
                  </button>
                ))}
              </div>
            )}

            {item.role === "assistant" && item.sources?.length > 0 && (
              <details className="source-details">
                <summary>View retrieved sources</summary>
                {item.sources.map((source) => (
                  <p key={source.chunk_id}>
                    Page {source.page}: {source.preview}
                  </p>
                ))}
              </details>
            )}
          </article>
        ))}

        {loading && <p className="thinking">Searching the PDF…</p>}
      </div>

      {error && <div role="alert" className="chat-error">{error}</div>}

      <form className="chat-form" onSubmit={handleSubmit}>
        <label htmlFor="question">Question</label>
        <textarea
          id="question"
          rows={3}
          value={message}
          onChange={(event) => {
            setMessage(event.target.value);
            setError("");
          }}
          placeholder="What would you like to know?"
          disabled={!enabled || loading || disabled}
        />
        <button
          type="submit"
          disabled={!enabled || !message.trim() || loading || disabled}
        >
          {loading ? "Searching…" : "Ask"}
        </button>
      </form>
    </section>
  );
}

export default ChatPanel;
