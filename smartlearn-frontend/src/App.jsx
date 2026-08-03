import { useState } from "react";
import ChatPanel from "./ChatPanel.jsx";
import PdfPreview from "./PdfPreview.jsx";
import { uploadPDF } from "./api.js";

function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [activePage, setActivePage] = useState(1);
  const [uploadKey, setUploadKey] = useState(0);
  const [status, setStatus] = useState("idle");
  const [chatBusy, setChatBusy] = useState(false);
  const [error, setError] = useState("");

  const uploading = status === "uploading";

  const handleUpload = async () => {
    if (!file) return;

    setError("");
    setStatus("uploading");
    try {
      const data = await uploadPDF(file);
      setUpload(data);
      setActivePage(1);
      setUploadKey((current) => current + 1);
    } catch (requestError) {
      setError(requestError.message || "Upload failed.");
    } finally {
      setStatus("idle");
    }
  };

  const handleJumpToPage = (page) => {
    if (Number.isInteger(page) && page > 0) {
      setActivePage(page);
    }
  };

  return (
    <main className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">Retrieval-augmented learning</p>
          <h1>SmartLearn Lite</h1>
          <p>Upload a PDF, ask grounded questions, and open cited pages.</p>
        </div>
      </header>

      {error && <div role="alert">{error}</div>}

      <form
        className="upload-form"
        onSubmit={(event) => {
          event.preventDefault();
          handleUpload();
        }}
      >
        <label htmlFor="pdf-file">PDF</label>
        <input
          id="pdf-file"
          type="file"
          accept=".pdf,application/pdf"
          onChange={(event) => {
            setFile(event.target.files[0] || null);
            setError("");
          }}
        />
        <button type="submit" disabled={!file || uploading || chatBusy}>
          {uploading ? "Preparing PDF…" : "Upload"}
        </button>
      </form>

      {upload && (
        <p className="upload-summary">
          {upload.filename} — {upload.pages} pages, {upload.characters.toLocaleString()} characters
        </p>
      )}

      <div className="workspace">
        <PdfPreview
          upload={upload}
          activePage={activePage}
          previewKey={uploadKey}
        />
        <ChatPanel
          key={uploadKey}
          enabled={Boolean(upload)}
          disabled={uploading}
          onBusy={setChatBusy}
          onJumpToPage={handleJumpToPage}
        />
      </div>
    </main>
  );
}

export default App;
