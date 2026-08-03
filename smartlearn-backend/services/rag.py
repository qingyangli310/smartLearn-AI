import json
import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Union

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


def extract_pages_for_rag(pdf_path: Union[str, Path]) -> list[dict]:
    """Read every page of a PDF and return a list of
    ``{"page": int, "text": str}`` records.  Pages are numbered
    from 1.  Pages whose cleaned text is empty are dropped.
    There is **no** 30‑page limit — this function is intentionally
    separate from ``services/pdf.py`` for Day 3 use."""
    pdf_path = Path(pdf_path)

    reader = PdfReader(BytesIO(pdf_path.read_bytes()))

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
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
