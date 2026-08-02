import os
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.llm import answer_from_pages
from .services.pdf import extract_pages

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

documents: dict[str, list[dict]] = {}


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

    # 4. Extract pages (raises ValueError if > 30 pages)
    try:
        pages = extract_pages(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 5. Reject scanned PDFs with no extractable text
    if all(page["text"] == "" for page in pages):
        raise HTTPException(
            status_code=422,
            detail="PDF contains no extractable text — scanned PDFs requiring OCR are not supported",
        )

    # 6. Store pages in memory (no disk write)
    documents[chat_id] = pages

    # 7. Success response
    total_chars = sum(len(page["text"]) for page in pages)
    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Look up stored pages
    pages = documents.get(request.chat_id)
    if pages is None:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF uploaded for chat_id '{request.chat_id}'. Please upload a PDF first.",
        )

    # 2. Call LLM (upstream failure → 502, no internal details leaked)
    try:
        answer = answer_from_pages(pages, request.message)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Upstream AI service failed. Please try again later.",
        )

    # 3. Extract [Page X] citations from the answer
    raw_numbers = re.findall(r"\[Page\s+(\d+)\]", answer)

    # 4. Keep only pages that exist, deduplicate, sort
    valid_pages = {page["page"] for page in pages}
    citations = sorted(set(
        int(n) for n in raw_numbers if int(n) in valid_pages
    ))

    # 5. Response
    return {"answer": answer, "citations": citations}
