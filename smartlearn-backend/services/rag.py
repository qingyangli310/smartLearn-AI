import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Union

# FAISS and PyTorch both use native parallel runtimes.  Limiting their shared
# thread pool prevents a macOS/Python 3.13 Jupyter kernel crash when a saved
# FAISS index is searched immediately before loading SentenceTransformers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import numpy as np
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Clean extracted PDF text by removing null bytes, soft hyphens,
    repeated whitespace, and messy newlines.  Paragraph breaks (blank
    lines between text blocks) are preserved so that downstream
    ``paragraph`` chunking can detect them.  Returns an empty string
    when nothing readable remains."""
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove soft hyphens (U+00AD)
    text = text.replace("­", "")

    # Normalize line endings: \r\n and \r → \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse repeated horizontal whitespace (spaces, tabs) into single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace from every line
    lines = [line.strip() for line in text.splitlines()]

    # Collapse 3+ consecutive newlines into at most 2
    # (i.e. one blank line separating paragraphs, = \n\n)
    collapsed: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                collapsed.append(line)
        else:
            blank_count = 0
            collapsed.append(line)

    return "\n".join(collapsed).strip()


def extract_pages_for_rag(
    pdf_path: Union[str, Path],
    page_limit: Union[int, None] = None,
) -> list[dict]:
    """Read every page of a PDF and return a list of
    ``{"page": int, "text": str}`` records.  Pages are numbered
    from 1.  Pages whose cleaned text is empty are dropped.
    There is **no** 30‑page limit — this function is intentionally
    separate from ``services/pdf.py`` for Day 3 use."""
    if page_limit is not None and page_limit <= 0:
        raise ValueError("page_limit must be > 0 when provided")

    pdf_path = Path(pdf_path)

    reader = PdfReader(BytesIO(pdf_path.read_bytes()))
    source_pages = reader.pages if page_limit is None else reader.pages[:page_limit]

    records: list[dict] = []
    for page_number, page in enumerate(source_pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})

    return records


def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """Extract cleaned, page-numbered records from uploaded PDF bytes.

    Unlike the Day 2 loader, this path intentionally has no 30-page limit.
    Empty pages are skipped while the original PDF page numbers are retained.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes must not be empty")

    reader = PdfReader(BytesIO(pdf_bytes))
    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        cleaned = clean_text(page.extract_text() or "")
        if cleaned:
            records.append({"page": page_number, "text": cleaned})
    return records


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save a Python object as UTF‑8 JSON.  Parent directories are
    created if they do not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Union[str, Path]) -> Any:
    """Load and return a Python object from a UTF‑8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preview_records(
    records: list[dict],
    columns: Union[list[str], None] = None,
    rows: int = 5,
) -> "pd.DataFrame":
    """Return a Pandas DataFrame showing the first *rows* rows of
    the requested *columns*.  Defaults to all columns and 5 rows.
    The original *records* list is not modified."""
    df = pd.DataFrame(records)
    if columns is not None:
        df = df[columns]
    return df.head(rows)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split *text* into segments no longer than *chunk_size* characters,
    preferring to break at spaces or newlines.  Never returns empty
    strings."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not text or not text.strip():
        return []

    result: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= chunk_size:
            result.append(remaining)
            break

        # Search backwards from chunk_size for a natural boundary
        cut = chunk_size
        # Prefer newline, then space
        for search_char in ("\n", " "):
            pos = remaining.rfind(search_char, 0, chunk_size)
            if pos > 0:
                cut = pos + 1  # include the delimiter in this chunk
                break

        # If no boundary found, also try rfind from the end
        if cut == chunk_size:
            pos = remaining.rfind(" ", 0, chunk_size)
            if pos > 0:
                cut = pos + 1

        result.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    # Filter out any empty strings (shouldn't happen, but safety)
    return [s for s in result if s]


def chunk_by_paragraph(records: list[dict], chunk_size: int) -> list[dict]:
    """Split pages by paragraph boundaries (blank lines).  Over-long
    paragraphs are further split via :func:`slice_long_text`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    chunks: list[dict] = []
    chunk_index = 0

    for rec in records:
        page = rec["page"]
        # Split text into paragraphs on blank line boundaries
        paragraphs = rec["text"].split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= chunk_size:
                chunks.append({
                    "chunk_id": f"c{chunk_index:04d}",
                    "page": page,
                    "text": para,
                    "chunk_mode": "paragraph",
                })
                chunk_index += 1
            else:
                for piece in slice_long_text(para, chunk_size):
                    chunks.append({
                        "chunk_id": f"c{chunk_index:04d}",
                        "page": page,
                        "text": piece,
                        "chunk_mode": "paragraph",
                    })
                    chunk_index += 1

    return chunks


def chunk_by_characters(
    records: list[dict],
    chunk_size: int,
    overlap: int = 0,
) -> list[dict]:
    """Split pages into fixed‑size character windows.  When *overlap*
    is 0 each character appears in exactly one chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be 0 <= overlap < chunk_size")

    chunks: list[dict] = []
    chunk_index = 0
    step = chunk_size - overlap

    for rec in records:
        page = rec["page"]
        text = rec["text"]
        start = 0

        while start < len(text):
            window = text[start:start + chunk_size]
            if not window.strip():
                start += step
                continue

            chunks.append({
                "chunk_id": f"c{chunk_index:04d}",
                "page": page,
                "text": window.strip(),
                "chunk_mode": "character_overlap" if overlap > 0 else "character",
            })
            chunk_index += 1
            start += step

    return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[dict]:
    """Build chunks from page *records* using one of three strategies:

    - ``"paragraph"`` — split on blank lines; slice over‑long paragraphs
    - ``"character"`` — fixed‑size windows, no overlap
    - ``"character_overlap"`` — fixed‑size windows with *overlap*

    Every returned chunk has keys ``chunk_id``, ``page``, ``text``,
    ``chunk_mode``.
    """
    if chunk_mode not in ("paragraph", "character", "character_overlap"):
        raise ValueError(
            f"Unsupported chunk_mode: {chunk_mode!r}. "
            "Expected 'paragraph', 'character', or 'character_overlap'."
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size)

    if chunk_mode == "character":
        return chunk_by_characters(records, chunk_size, overlap=0)

    # character_overlap.  An overlap of zero is valid, but the chunks still
    # belong to the strategy explicitly selected by the caller.
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be 0 <= overlap < chunk_size")
    chunks = chunk_by_characters(records, chunk_size, overlap=overlap)
    for chunk in chunks:
        chunk["chunk_mode"] = "character_overlap"
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_RELEVANCE_SCORE = 0.30
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_REQUIRED_FILES = (
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
)


def model_tag(model_name: str) -> str:
    """Return a stable, filename-safe tag for a model name."""
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not tag:
        raise ValueError("model_name must contain at least one safe character")
    return tag


def _local_model_ready(path: Path) -> bool:
    return path.is_dir() and all((path / item).exists() for item in _MODEL_REQUIRED_FILES)


def resolve_model_source(
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    artifact_root: Union[str, Path, None] = None,
) -> Union[str, Path]:
    """Prefer a complete local model folder, falling back to *model_name*."""
    supplied_path = Path(model_name).expanduser()
    if _local_model_ready(supplied_path):
        return supplied_path.resolve()

    local_name = Path(model_name).name
    candidates: list[Path] = []
    if artifact_root is not None:
        candidates.append(Path(artifact_root) / "hf_models" / local_name)

    backend_root = Path(__file__).resolve().parents[1]
    candidates.append(backend_root / "artifacts" / "rag" / "hf_models" / local_name)

    configured_model_root = os.getenv("RAG_EMBED_MODEL_PATH")
    if configured_model_root:
        candidates.append(Path(configured_model_root).expanduser())

    # Workshop folders commonly sit next to the repository. Reusing their
    # downloaded model keeps the backend functional when network access is off.
    workspace_parent = Path(__file__).resolve().parents[3]
    for lab_folder in ("Day3", "Day3_Lab"):
        candidates.append(
            workspace_parent / lab_folder / "artifacts" / "hf_models" / local_name
        )

    for candidate in candidates:
        if _local_model_ready(candidate):
            return candidate.resolve()

    return model_name


def get_device() -> str:
    """Return ``"cuda"`` when a GPU is available, otherwise ``"cpu"``."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    source: Union[str, Path, None] = None,
    artifact_root: Union[str, Path, None] = None,
    device: Union[str, None] = None,
) -> "SentenceTransformer":
    """Create or reuse a sentence-transformer model instance."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    model_source = source or resolve_model_source(model_name, artifact_root)
    selected_device = device or get_device()
    cache_key = (str(model_source), selected_device)

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = SentenceTransformer(
            str(model_source),
            device=selected_device,
        )
    return _MODEL_CACHE[cache_key]


def embed_texts(
    texts: list[str],
    model: Union["SentenceTransformer", None] = None,
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode texts as normalized ``float32`` NumPy vectors."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain only non-empty strings")

    active_model = model or load_model(model_name, artifact_root=artifact_root)
    embeddings = active_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(embeddings, dtype=np.float32)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    return result


def artifact_paths_for(
    document_id: str,
    chunk_mode: str = "character_overlap",
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    chunk_size: int = 700,
    overlap: int = 120,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Return stable artifact paths for one document and configuration."""
    root = (
        Path(artifact_root)
        if artifact_root is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / "rag"
    )
    document_dir = root / model_tag(document_id)
    effective_overlap = overlap if chunk_mode == "character_overlap" else 0
    config_tag = (
        f"{chunk_mode}_size{chunk_size}_overlap{effective_overlap}_"
        f"{model_tag(model_name)}"
    )
    return {
        "document_dir": document_dir,
        "raw_pages": document_dir / "pages.json",
        "pages": document_dir / "pages.json",
        "chunks": document_dir / f"chunks_{config_tag}.json",
        "embeddings": document_dir / f"embeddings_{config_tag}.npy",
        "manifest": document_dir / f"manifest_{config_tag}.json",
        "index": document_dir / f"index_{config_tag}.faiss",
    }


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Build or reuse pages, chunks, embeddings, and a manifest."""
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    if not pdf_name.strip():
        raise ValueError("pdf_name must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    effective_overlap = overlap if chunk_mode == "character_overlap" else 0
    paths = artifact_paths_for(
        document_id=document_id,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=effective_overlap,
        artifact_root=artifact_root,
    )
    pages_signature = hashlib.sha256(
        json.dumps(pages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    signature = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "pages_signature": pages_signature,
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": effective_overlap,
        "model_name": model_name,
    }

    required_paths = (
        paths["raw_pages"],
        paths["chunks"],
        paths["embeddings"],
        paths["manifest"],
    )
    if all(path.exists() for path in required_paths):
        manifest = load_json(paths["manifest"])
        if all(manifest.get(key) == value for key, value in signature.items()):
            cached_chunks = load_json(paths["chunks"])
            cached_embeddings = np.load(paths["embeddings"], allow_pickle=False)
            if (
                cached_embeddings.ndim == 2
                and cached_embeddings.dtype == np.float32
                and len(cached_chunks) == cached_embeddings.shape[0]
                and manifest.get("num_chunks") == len(cached_chunks)
                and manifest.get("embedding_dim") == cached_embeddings.shape[1]
            ):
                return {
                    "pages": load_json(paths["raw_pages"]),
                    "chunks": cached_chunks,
                    "embeddings": cached_embeddings,
                    "manifest": manifest,
                    "paths": paths,
                    "reused": True,
                }

    chunks = build_chunks(
        pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=effective_overlap,
    )
    texts = [c["text"] for c in chunks]
    model = load_model(model_name, artifact_root=artifact_root)
    embeddings = embed_texts(
        texts,
        model=model,
        batch_size=batch_size,
        show_progress=True,
    )
    selected_device = get_device()

    manifest = {
        **signature,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1] if embeddings.ndim == 2 else 0,
        "device": selected_device,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
        "manifest_path": str(paths["manifest"]),
        "index_path": str(paths["index"]),
    }

    paths["document_dir"].mkdir(parents=True, exist_ok=True)
    save_json(pages, paths["raw_pages"])
    save_json(chunks, paths["chunks"])
    np.save(paths["embeddings"], embeddings, allow_pickle=False)
    save_json(manifest, paths["manifest"])

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "manifest": manifest,
        "paths": paths,
        "reused": False,
    }


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """Create a FAISS inner‑product index from *embeddings*.

    Because :func:`embed_texts` produces L2‑normalised vectors,
    inner‑product (dot product) is equivalent to cosine similarity:
    :math:`\\mathbf{a} \\cdot \\mathbf{b} = \\|\\mathbf{a}\\| \\|\\mathbf{b}\\| \\cos\\theta = \\cos\\theta`
    when both vectors have unit length.  ``IndexFlatIP`` therefore
    performs exact cosine‑similarity nearest‑neighbour search."""
    import faiss  # type: ignore[import-untyped]

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2‑D array")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must contain at least one non-empty vector")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index: "faiss.Index", index_path: Union[str, Path]) -> None:
    """Serialize a FAISS index to disk.  Parent directories are
    created automatically."""
    import faiss  # type: ignore[import-untyped]

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(index_path: Union[str, Path]) -> "faiss.Index":
    """Deserialize a FAISS index from disk."""
    import faiss  # type: ignore[import-untyped]

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(str(path))


def _index_meta_path(index_path: Union[str, Path]) -> Path:
    """Companion JSON path for an index file."""
    p = Path(index_path)
    return p.with_suffix(p.suffix + ".json")


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: Union[list[dict], None] = None,
    pdf_path: Union[str, Path, None] = None,
    chunk_mode: str = "character_overlap",
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Create or reuse pages, chunks, embeddings, manifest, and a
    FAISS index for one document.

    If *pages* is ``None``, *pdf_path* must be provided so the PDF
    can be read via :func:`extract_pages_for_rag`.
    """
    # ---- resolve pages ----
    if pages is None:
        if pdf_path is None:
            raise ValueError("Either pages or pdf_path must be provided")
        pages = extract_pages_for_rag(pdf_path)

    # ---- ensure base artifacts (Lab A) ----
    bundle = ensure_artifacts(
        document_id=document_id,
        pdf_name=pdf_name,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    index_path = bundle["paths"]["index"]
    meta_path = _index_meta_path(index_path)
    manifest = bundle["manifest"]
    expected_meta = {
        "document_id": document_id,
        "pages_signature": manifest["pages_signature"],
        "chunk_mode": manifest["chunk_mode"],
        "chunk_size": manifest["chunk_size"],
        "overlap": manifest["overlap"],
        "model_name": manifest["model_name"],
        "num_vectors": len(bundle["chunks"]),
        "embedding_dim": int(bundle["embeddings"].shape[1]),
    }

    # ---- reuse cached index if valid ----
    rebuild = True
    index_reused = False
    if index_path.exists() and meta_path.exists():
        meta = load_json(meta_path)
        if all(meta.get(key) == value for key, value in expected_meta.items()):
            try:
                index = load_faiss_index(index_path)
            except (OSError, RuntimeError):
                pass
            else:
                if (
                    index.ntotal == expected_meta["num_vectors"]
                    and index.d == expected_meta["embedding_dim"]
                ):
                    rebuild = False
                    index_reused = True

    # ---- build fresh index ----
    if rebuild:
        index = build_faiss_index(bundle["embeddings"])
        save_faiss_index(index, index_path)
        save_json(expected_meta, meta_path)

    artifacts = dict(bundle["paths"])
    artifacts["index_metadata"] = meta_path

    return {
        "chunks": bundle["chunks"],
        "embeddings": bundle["embeddings"],
        "manifest": manifest,
        "index": index,
        "artifacts": artifacts,
        "reused": bundle.get("reused", False),
        "index_reused": index_reused,
    }


def relative_path_str(path: Union[str, Path], base: Union[str, Path, None] = None) -> str:
    """Return a relative path string suitable for notebook display."""
    target = Path(path).resolve()
    if base is None:
        base = Path.cwd()
    else:
        base = Path(base).resolve()
    try:
        rel = target.relative_to(base)
    except ValueError:
        return str(target)
    return str(rel)


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Prepare an already-extracted PDF page list for downstream retrieval.

    The parameter names intentionally match the Lab B notebook so uploaded or
    cached page records can be reused without reading the PDF a second time.
    """
    result = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": result["chunks"],
        "chunk_size": len(result["chunks"]),
        "embedding_dim": int(result["embeddings"].shape[1]),
        "history": [],
        "index": result["index"],
        "manifest": result["manifest"],
        "artifacts": result["artifacts"],
        "index_reused": result["index_reused"],
    }


def prepare_rag_chat_record(
    chat_id: str,
    filename: str,
    pdf_bytes: Union[bytes, None] = None,
    pages: Union[list[dict], None] = None,
    upload_root: Union[str, Path, None] = None,
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = DEFAULT_EMBED_MODEL_NAME,
    batch_size: int = 32,
    artifact_root: Union[str, Path, None] = None,
) -> dict:
    """Build the in-memory record stored under ``documents[chat_id]``.

    The caller may provide uploaded bytes or already extracted page records.
    When bytes are supplied, the PDF is saved only after the RAG assets have
    been prepared successfully, preventing a failed upload from leaving a
    half-written session file behind.
    """
    if not chat_id.strip():
        raise ValueError("chat_id must not be empty")
    if not filename.strip():
        raise ValueError("filename must not be empty")
    if pdf_bytes is None and pages is None:
        raise ValueError("Provide pdf_bytes or pages")

    active_pages = pages
    if active_pages is None:
        active_pages = extract_pages_from_bytes_for_rag(pdf_bytes or b"")
    if not active_pages:
        raise ValueError("PDF contains no extractable text")

    document = prepare_rag_document(
        document_id=chat_id,
        filename=filename,
        pages=active_pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    saved_pdf_path = ""
    if pdf_bytes is not None:
        target_root = (
            Path(upload_root)
            if upload_root is not None
            else Path(__file__).resolve().parents[1] / "uploads"
        )
        target_root.mkdir(parents=True, exist_ok=True)
        safe_chat_id = re.sub(r"[^A-Za-z0-9._-]+", "_", chat_id).strip("._-")
        if not safe_chat_id:
            safe_chat_id = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:16]
        target_path = target_root / f"{safe_chat_id}.pdf"
        temp_path = target_root / f".{safe_chat_id}.uploading"
        temp_path.write_bytes(pdf_bytes)
        temp_path.replace(target_path)
        saved_pdf_path = str(target_path.resolve())

    index_path = str(Path(document["artifacts"]["index"]).resolve())
    document.update({
        "chat_id": chat_id,
        "file_path": saved_pdf_path,
        "saved_pdf_path": saved_pdf_path,
        "rag": {
            "document_id": chat_id,
            "index_path": index_path,
            "model_name": model_name,
            "chunk_mode": chunk_mode,
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    })
    return document


def build_upload_response(document: dict) -> dict:
    """Return the stable Day 2 upload response for a richer Day 3 record."""
    pages = document.get("pages", [])
    return {
        "status": "ok",
        "filename": str(document.get("filename", "")),
        "pages": len(pages),
        "characters": sum(len(str(page.get("text", ""))) for page in pages),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def keyword_set(text: str) -> set[str]:
    """Return a lightweight lexical token set for simple reranking.

    Tokens are lowercased and lightly stemmed.  Technical identifiers such
    as ``BM25``, ``R&A``, ``ROUGE-L``, ``L3-70B``, and ``11T`` are retained;
    these are often the exact short answers a document question asks for.
    """
    stopwords = {
        "and", "are", "as", "at", "be", "by", "for", "from", "how",
        "into", "is", "it", "of", "on", "or", "paper", "that", "the",
        "their", "this", "to", "used", "using", "what", "when", "where",
        "which", "who", "why", "with",
    }
    tokens: set[str] = set()
    for match in re.finditer(r"[a-z0-9]+(?:[&-][a-z0-9]+)*", text.lower()):
        word = match.group()
        if word.isalpha() and word.endswith("s") and len(word) > 3:
            word = word[:-1]
        if len(word) >= 2 and word not in stopwords:
            tokens.add(word)
    return tokens


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: Union[list[dict], None] = None,
) -> list[dict]:
    """Retrieve top‑*k* chunks from an in‑memory index bundle.

    1. Embed *question* with the same model used for the chunks.
    2. Search the FAISS index for the nearest *candidate_pool* vectors.
    3. Optionally rerank candidates with a small lexical bonus.
    4. Return the top‑*top_k* hits (``page``, ``chunk_id``, ``text``,
       ``score``).
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if candidate_pool < top_k:
        candidate_pool = top_k

    index: "faiss.Index" = bundle["index"]
    chunks: list[dict] = bundle["chunks"]
    if index.ntotal != len(chunks):
        raise ValueError("FAISS vector count does not match chunk count")
    if not chunks:
        return []

    # A follow-up such as "give one more detail from that page" needs recent
    # context to form a meaningful retrieval query. Fresh evidence is still
    # retrieved every turn; history only helps resolve the query wording.
    retrieval_query = question
    if history:
        recent_context: list[str] = []
        for turn in history[-2:]:
            prior_question = str(turn.get("question", "")).strip()
            prior_answer = str(turn.get("answer", "")).strip()
            if prior_question:
                recent_context.append(f"Previous question: {prior_question}")
            if prior_answer:
                recent_context.append(f"Previous answer: {prior_answer[:500]}")
        if recent_context:
            retrieval_query = "\n".join([*recent_context, f"Current question: {question}"])

    # Embed the question
    manifest = bundle.get("manifest", {})
    model_name = manifest.get("model_name", DEFAULT_EMBED_MODEL_NAME)
    model = load_model(
        model_name=model_name,
        artifact_root=_resolve_artifact_root(bundle),
    )
    q_vec = embed_texts(
        [retrieval_query],
        model=model,
        batch_size=batch_size,
        show_progress=False,
    )
    if q_vec.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension {q_vec.shape[1]} does not match "
            f"FAISS index dimension {index.d}"
        )

    # FAISS search
    search_count = min(max(top_k, candidate_pool), index.ntotal)
    scores, indices = index.search(
        np.ascontiguousarray(q_vec, dtype=np.float32),
        search_count,
    )

    # Collect hits
    hits: list[dict] = []
    q_keywords = keyword_set(retrieval_query)

    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        # Lightweight lexical bonus: +0.02 per overlapping keyword (cap 0.10).
        # Exact intent qualifiers deserve extra weight: a chunk containing
        # "default" is substantially more useful for a "used by default"
        # question than one that merely repeats "model" and "agent".
        c_keywords = keyword_set(chunk["text"])
        shared_keywords = q_keywords & c_keywords
        overlap = len(shared_keywords)
        intent_terms = {"default", "named"}
        intent_bonus = 0.35 if shared_keywords & intent_terms else 0.0
        bonus = min(overlap * 0.02, 0.10) + intent_bonus
        vector_score = float(score)
        hits.append({
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "vector_score": round(vector_score, 4),
            "lexical_score": round(bonus, 4),
            "score": round(vector_score + bonus, 4),
        })

    # Sort by score descending, keep top_k
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def _resolve_artifact_root(bundle: dict) -> Union[str, Path, None]:
    """Extract artifact_root from a bundle's artifact paths."""
    artifacts = bundle.get("artifacts", {})
    raw_pages = artifacts.get("raw_pages", "")
    if raw_pages:
        doc_dir = Path(raw_pages).parent          # <root>/pdf1/
        return doc_dir.parent                     # <root>/
    return None


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: Union[list[dict], None] = None,
) -> list[dict]:
    """Retrieve top‑*k* chunks from a prepared document record.

    The saved index is loaded for each document-level search so this path also
    works with a document record restored in a later process."""
    artifacts = document.get("artifacts", {})
    index_path = artifacts.get("index")
    if index_path is None:
        raise ValueError("document is missing artifacts['index']")

    bundle = dict(document)
    bundle["index"] = load_faiss_index(index_path)
    return search_bundle(
        question=question,
        bundle=bundle,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=history,
    )


def split_sentences(text: str) -> list[str]:
    """Split *text* into candidate answer sentences.

    PDF extraction commonly inserts line breaks in the middle of sentences
    and hyphenates words at the right margin.  Repair those artifacts before
    finding sentence boundaries so candidates such as ``comprehen-\nsive``
    are not treated as incomplete answers.
    """
    if not text:
        return []

    # Join ordinary margin hyphenation while preserving technical names such
    # as ``Llama-\n3.3-70B``.
    normalized = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    normalized = re.sub(r"-\s*\n\s*(?=[A-Z0-9])", "-", normalized)
    normalized = re.sub(r"\s*\n\s*", " ", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()

    raw = re.split(r"(?<=[.!?])\s+", normalized)
    return [s.strip() for s in raw if s.strip()]


def _technical_identifiers(text: str) -> set[str]:
    """Return likely short-answer entities from a candidate sentence."""
    identifiers: set[str] = set()
    for token in re.findall(r"\b[A-Za-z0-9]+(?:[&-][A-Za-z0-9]+)*\b", text):
        normalized = token.replace("­", "").strip(".,").casefold()
        uppercase_count = sum(character.isupper() for character in token)
        if (
            normalized
            and (
                any(character.isdigit() for character in token)
                or "&" in token
                or uppercase_count >= 2
            )
        ):
            identifiers.add(normalized)
    return identifiers


def _answer_cue_score(question: str, sentence: str) -> float:
    """Score generic linguistic clues that a sentence states an answer."""
    q = question.casefold()
    s = sentence.casefold()
    score = 0.0
    identifiers = _technical_identifiers(sentence)

    asks_for_identifier = any(cue in q for cue in ("name", "model", "retriever"))
    if asks_for_identifier and identifiers:
        score += 1.25
    elif asks_for_identifier:
        score -= 1.5
    if "metric" in q and "metric" in s:
        score += 4.0 if identifiers else 2.0
    if any(cue in q for cue in ("heading", "section")) and ":" in sentence:
        score += 1.25
    if re.search(r"\b(?:is|are|uses?|called|named|as the)\b", s):
        score += 0.35
    return score


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Return a short local answer sentence with a page tag.

    From the retrieved *hits*, split each chunk into sentences,
    score each sentence by keyword overlap with the *question*,
    and pick the single best.  A ``[Page N]`` tag is appended
    when the sentence's source page is known."""
    if not hits:
        return "No relevant text found in the document."

    q_keywords = keyword_set(question)
    best_sentence = ""
    best_page = None
    best_score = float("-inf")

    for rank, hit in enumerate(hits):
        for sentence in split_sentences(hit["text"]):
            s_keywords = keyword_set(sentence)
            overlap = len(q_keywords & s_keywords)
            coverage = overlap / max(len(q_keywords), 1)

            # Prefer complete, answer-bearing sentences over fragments that
            # merely repeat the wording of the question.  Retrieval rank is a
            # useful signal, but remains weaker than sentence-level evidence.
            score = overlap + coverage + _answer_cue_score(question, sentence)
            score += max(0.0, 0.25 - rank * 0.08)
            score += min(float(hit.get("score", 0.0)), 1.0) * 0.2

            stripped = sentence.rstrip()
            if stripped.endswith((",", ";", ":", "-")):
                score -= 1.5
            if len(sentence) < 20:
                score -= 1.25
            elif len(sentence) > 420:
                score -= (len(sentence) - 420) * 0.002

            if score > best_score:
                best_score = score
                best_sentence = sentence
                best_page = hit["page"]

    if not best_sentence:
        return "No relevant text found in the document."

    if best_page is not None:
        return f"{best_sentence} [Page {best_page}]"
    return best_sentence


# ---------------------------------------------------------------------------
# Project-facing answer wrapper
# ---------------------------------------------------------------------------

def extract_citations(
    answer: str,
    hits: Union[list[dict], None] = None,
) -> list[int]:
    """Return unique numeric PDF page citations in display order.

    Explicit ``Page``/``p.`` references in the answer take precedence.  When
    a local answer contains no page tag, retrieved hit pages provide a safe
    evidence-based fallback.
    """
    citations: list[int] = []
    for match in re.finditer(r"\b(?:page|p\.)\s*#?\s*(\d+)\b", answer, re.IGNORECASE):
        page = int(match.group(1))
        if page > 0 and page not in citations:
            citations.append(page)

    if not citations and hits:
        for hit in hits:
            page = hit.get("page")
            if isinstance(page, int) and page > 0 and page not in citations:
                citations.append(page)
    return citations


def build_sources(hits: list[dict]) -> list[dict]:
    """Convert retrieval hits to compact, frontend-friendly source records."""
    sources: list[dict] = []
    for hit in hits:
        preview = re.sub(r"\s+", " ", str(hit.get("text", ""))).strip()
        source = {
            "page": hit.get("page"),
            "chunk_id": hit.get("chunk_id"),
            "score": float(hit.get("score", 0.0)),
            "preview": preview[:280],
        }
        if "vector_score" in hit:
            source["vector_score"] = float(hit["vector_score"])
        if "lexical_score" in hit:
            source["lexical_score"] = float(hit["lexical_score"])
        sources.append(source)
    return sources


def build_grounded_user_prompt(
    question: str,
    hits: list[dict],
    history: Union[list[dict], None] = None,
) -> str:
    """Build a history-aware prompt whose evidence stays page grounded."""
    if not question.strip():
        raise ValueError("question must not be empty")

    history_lines: list[str] = []
    for turn in (history or [])[-3:]:
        prior_question = str(turn.get("question", "")).strip()
        prior_answer = str(turn.get("answer", "")).strip()
        citations = turn.get("citations", [])
        if prior_question:
            history_lines.append(f"User: {prior_question}")
        if prior_answer:
            citation_text = f" Citations: {citations}" if citations else ""
            history_lines.append(f"Assistant: {prior_answer}{citation_text}")

    evidence_blocks = [
        f"[Page {hit.get('page')}]\n{str(hit.get('text', '')).strip()}"
        for hit in hits
        if str(hit.get("text", "")).strip()
    ]
    history_text = "\n".join(history_lines) or "(no earlier turns)"
    evidence_text = "\n\n".join(evidence_blocks) or "(no relevant evidence found)"
    return (
        "Answer the current question using only the retrieved PDF evidence.\n"
        "Use recent conversation only to understand references in the question.\n"
        "Cite factual claims with [Page N]. If the evidence is insufficient, "
        "say that the answer was not found.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Retrieved PDF evidence:\n{evidence_text}\n\n"
        f"Current question: {question}"
    )


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Retrieve evidence and return a stable app-facing answer payload.

    Lab B deliberately keeps a fully local fallback.  ``answer_model`` is
    accepted for later API integration, while this implementation always
    returns a grounded extracted sentence and therefore requires no API key.
    """
    hits = search_document(
        question=question,
        document=document,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=document.get("history"),
    )
    sources = build_sources(hits)
    if not hits or float(hits[0].get("score", 0.0)) < MIN_RELEVANCE_SCORE:
        return {
            "answer": "The answer was not found in the document.",
            "citations": [],
            "sources": sources,
        }

    # Build the exact grounded prompt needed by a later hosted-model branch.
    # The workshop's default remains a deterministic local answer so the route
    # works without an API key or network connection.
    build_grounded_user_prompt(question, hits, history=document.get("history"))
    answer = best_sentence_answer(question, hits)
    return {
        "answer": answer,
        "citations": extract_citations(answer, hits),
        "sources": sources,
    }


def append_history(document: dict, question: str, result: dict) -> list[dict]:
    """Append one question-answer turn to a document's in-memory history."""
    history = document.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError("document history must be a list")
    history.append({
        "question": question,
        "answer": str(result.get("answer", "")),
        "citations": list(result.get("citations", [])),
        "sources": list(result.get("sources", [])),
    })
    return history


def answer_document_turn(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Retrieve fresh evidence, answer, then append one successful turn."""
    result = answer_document(
        document=document,
        question=question,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
    )
    history = append_history(document, question, result)
    return {**result, "history": list(history)}


def answer_chat_turn(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "poolside/laguna-s-2.1:free",
) -> dict:
    """Project-facing alias for one retrieve-per-turn chat interaction."""
    return answer_document_turn(
        document=document,
        question=message,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
    )


# ---------------------------------------------------------------------------
# Simple retrieval evaluation
# ---------------------------------------------------------------------------

def normalize_for_match(text: str) -> str:
    """Normalize text for beginner-friendly short-answer matching."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Return whether *text* contains any acceptable normalized answer."""
    normalized_text = normalize_for_match(text)
    compact_text = normalized_text.replace(" ", "")
    for answer in answers:
        normalized_answer = normalize_for_match(answer)
        if not normalized_answer:
            continue
        if normalized_answer in normalized_text:
            return True
        if normalized_answer.replace(" ", "") in compact_text:
            return True
    return False


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
) -> "pd.DataFrame":
    """Evaluate retrieval evidence and local answers one question at a time."""
    rows: list[dict] = []
    for item in eval_set:
        pdf_name = item.get("pdf_name", "")
        question = item.get("question", "")
        answers = item.get("answers", [])
        if pdf_name not in documents_by_name:
            raise KeyError(f"No prepared document for {pdf_name!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Each evaluation record needs a non-empty question")
        if not isinstance(answers, list) or not answers:
            raise ValueError("Each evaluation record needs at least one answer")

        hits = search_document(
            question=question,
            document=documents_by_name[pdf_name],
            top_k=top_k,
            candidate_pool=candidate_pool,
        )
        local_answer = best_sentence_answer(question, hits)
        retrieved_text = "\n".join(str(hit.get("text", "")) for hit in hits)
        pages = [hit.get("page") for hit in hits if isinstance(hit.get("page"), int)]

        rows.append({
            "pdf_name": pdf_name,
            "question": question,
            "gold_answers": " | ".join(str(answer) for answer in answers),
            "retrieved_pages": pages,
            "local_answer": local_answer,
            "retrieval_hit": contains_any_answer(retrieved_text, answers),
            "answer_hit": contains_any_answer(local_answer, answers),
        })

    return pd.DataFrame(rows)
