import { useState } from "react";
import { uploadPDF, askQuestion } from "./api.js";

function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const busy = status !== "idle";

  const handleUpload = async () => {
    setUpload(null);
    setAnswer(null);
    setError("");
    setStatus("uploading");
    try {
      const data = await uploadPDF(file);
      setUpload(data);
    } catch (requestError) {
      setError(requestError.message || "Upload failed.");
    } finally {
      setStatus("idle");
    }
  };

  const handleAsk = async () => {
    setError("");
    setAnswer(null);
    setStatus("asking");
    try {
      const data = await askQuestion(message.trim());
      setAnswer(data);
    } catch (requestError) {
      setError(requestError.message || "Chat failed.");
    } finally {
      setStatus("idle");
    }
  };

  return (
    <div className="app">
      <h1>SmartLearn Lite</h1>
      <p>Your AI-powered learning assistant</p>

      {error && <div role="alert">{error}</div>}

      <form onSubmit={(e) => { e.preventDefault(); handleUpload(); }}>
        <label htmlFor="pdf-file">PDF</label>
        <input
          id="pdf-file"
          type="file"
          accept=".pdf,application/pdf"
          onChange={(e) => {
            setFile(e.target.files[0] || null);
            setUpload(null);
            setAnswer(null);
            setError("");
          }}
        />
        <button type="submit" disabled={!file || busy}>
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>
      </form>

      {upload && (
        <p>{upload.filename} — {upload.pages} pages, {upload.characters} characters</p>
      )}

      <form onSubmit={(e) => { e.preventDefault(); handleAsk(); }}>
        <label htmlFor="question">Question</label>
        <textarea
          id="question"
          rows={3}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            setError("");
          }}
        />
        <button type="submit" disabled={!upload || !message.trim() || busy}>
          {status === "asking" ? "Asking…" : "Ask"}
        </button>
      </form>

      {answer && (
        <>
          <section>
            <h2>Answer</h2>
            <p>{answer.answer}</p>
          </section>
          {answer.citations?.length > 0 && (
            <section>
              <h2>Citations</h2>
              {answer.citations.map((page) => (
                <span key={page}>Page {page}</span>
              ))}
            </section>
          )}
        </>
      )}
    </div>
  );
}

export default App;
