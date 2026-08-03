import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .services import rag

app = FastAPI(title="SmartLearn Lite API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BACKEND_DIR / "uploads"
ARTIFACT_ROOT = Path(
    os.getenv("RAG_ARTIFACT_ROOT", str(BACKEND_DIR / "artifacts" / "rag"))
)

documents: dict[str, dict] = {}


class ChatRequest(BaseModel):
    chat_id: str = "day2-demo"
    message: str = Field(..., min_length=2, max_length=2000)


@app.get("/")
def root():
    return {"message": "SmartLearn Lite API is running"}

@app.get("/health")
def health_check():
    return {"ok": True}


@app.post("/upload")
async def upload_pdf(chat_id: str, file: UploadFile = File(...)):
    # 1. Reject non-PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # 2. Read file into memory
    pdf_bytes = await file.read()

    # 3. Reject empty file
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # 4. Prepare a complete Day 3 record before replacing the active session.
    try:
        document = rag.prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename or "upload.pdf",
            pdf_bytes=pdf_bytes,
            upload_root=UPLOAD_ROOT,
            artifact_root=ARTIFACT_ROOT,
        )
    except ValueError as e:
        if "no extractable text" in str(e).lower():
            raise HTTPException(status_code=422, detail=str(e)) from e
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Could not prepare the PDF for retrieval.",
        ) from e

    documents[chat_id] = document
    return rag.build_upload_response(document)


@app.get("/documents/{chat_id}/file")
def get_document_file(chat_id: str):
    document = documents.get(chat_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document session not found")

    raw_path = document.get("file_path") or document.get("saved_pdf_path")
    file_path = Path(raw_path) if raw_path else None
    if file_path is None or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Uploaded PDF file not found")

    return FileResponse(file_path, media_type="application/pdf")


@app.post("/chat")
async def chat(request: ChatRequest):
    document = documents.get(request.chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF uploaded for chat_id '{request.chat_id}'. Please upload a PDF first.",
        )

    try:
        result = rag.answer_chat_turn(
            document=document,
            message=request.message,
            top_k=3,
            candidate_pool=60,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail="Document retrieval failed. Please try again later.",
        ) from e

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
