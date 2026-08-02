import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions using ONLY the supplied PDF text. "
    "Cite factual claims with [Page X], where X MUST be the page that contains "
    "the DIRECT supporting evidence for that specific claim. "
    "Do NOT cite a page just because it mentions a related concept or background — "
    "the evidence itself must appear on that page. "
    "If the PDF does not contain direct evidence to answer the question, "
    "say so clearly and do NOT generate any citations. "
    "Respond in plain text only. No Markdown, no bold, no LaTeX, no code fences. "
    "Never invent a page number."
)

def answer_from_pages(pages: list[dict], message: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is not configured")

    document_text = "\n\n".join(
        f"### [Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"PDF text:\n{document_text}\n\nmessage: {message}",
            },
        ],
    )
    logger.info("OpenRouter model: %s", response.model)
    return response.choices[0].message.content or ""