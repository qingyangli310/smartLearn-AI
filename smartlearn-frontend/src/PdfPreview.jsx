import { getDocumentFileURL } from "./api.js";

function PdfPreview({ upload, activePage, previewKey }) {
  return (
    <section className="preview-panel" aria-label="PDF preview">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Document</p>
          <h2>PDF Preview</h2>
        </div>
        {upload && <span className="page-label">Page {activePage}</span>}
      </div>

      {upload ? (
        <iframe
          key={`${previewKey}-${activePage}`}
          className="pdf-frame"
          src={getDocumentFileURL(activePage, previewKey)}
          title={`${upload.filename} — page ${activePage}`}
        />
      ) : (
        <div className="empty-state">
          <p>Upload a PDF to preview it here.</p>
        </div>
      )}
    </section>
  );
}

export default PdfPreview;
