import importlib
import unittest
from unittest.mock import patch


rag = importlib.import_module("smartlearn-backend.services.rag")


class LabCAnswerTests(unittest.TestCase):
    def test_split_sentences_repairs_pdf_line_breaks(self):
        text = (
            "Document summarization requires a comprehen-\n"
            "sive answer. The default agent is Llama-\n3.3-70B."
        )

        sentences = rag.split_sentences(text)

        self.assertEqual(
            sentences,
            [
                "Document summarization requires a comprehensive answer.",
                "The default agent is Llama-3.3-70B.",
            ],
        )

    def test_best_sentence_prefers_complete_identifier_answer(self):
        hits = [
            {
                "page": 4,
                "chunk_id": "c1",
                "score": 0.60,
                "text": (
                    "This adds thematic keywords for the sparse retriever.\n"
                    "Traditional RAG uses the sparse retriever BM25."
                ),
            }
        ]

        answer = rag.best_sentence_answer(
            "What is the name of the sparse retriever?",
            hits,
        )

        self.assertEqual(answer, "Traditional RAG uses the sparse retriever BM25. [Page 4]")

    def test_best_sentence_prefers_explicit_metric_statement(self):
        hits = [
            {
                "page": 6,
                "chunk_id": "c2",
                "score": 0.70,
                "text": (
                    "Document-level summarization requires a comprehensive understanding. "
                    "For this task, we use ROUGE-L as the metric."
                ),
            }
        ]

        answer = rag.best_sentence_answer(
            "What metric is used for document-level summarization?",
            hits,
        )

        self.assertEqual(answer, "For this task, we use ROUGE-L as the metric. [Page 6]")

    def test_low_relevance_question_returns_not_found_without_citations(self):
        document = {"history": []}
        low_relevance_hits = [
            {
                "page": 23,
                "chunk_id": "c-test",
                "text": "Unrelated document text.",
                "score": 0.17,
            }
        ]

        with patch.object(rag, "search_document", return_value=low_relevance_hits):
            result = rag.answer_document(
                document,
                "What is the cafeteria Wi-Fi password?",
            )

        self.assertEqual(result["answer"], "The answer was not found in the document.")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["sources"][0]["page"], 23)

    def test_grounded_prompt_contains_history_evidence_and_current_question(self):
        history = [
            {
                "question": "Which retriever is used?",
                "answer": "BM25. [Page 4]",
                "citations": [4],
            }
        ]
        hits = [{"page": 4, "text": "The sparse index uses BM25."}]

        prompt = rag.build_grounded_user_prompt(
            "Give one more detail from that page.",
            hits,
            history,
        )

        self.assertIn("Which retriever is used?", prompt)
        self.assertIn("[Page 4]", prompt)
        self.assertIn("Give one more detail from that page.", prompt)


if __name__ == "__main__":
    unittest.main()
