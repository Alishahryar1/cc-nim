"""Semantic RAG engine for conversation history."""

from __future__ import annotations

import math
import re
from collections import Counter

from api.models.anthropic import ContentBlockText, Message


class RagEngine:
    def __init__(self):
        pass

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace and punctuation tokenization with basic stemming."""
        tokens = re.findall(r"\w+", text.lower())
        stemmed = []
        for t in tokens:
            if len(t) <= 3:
                stemmed.append(t)
                continue
            # Very aggressive/naive stemming
            if t.endswith("ies"):
                t = t[:-3] + "y"
            elif t.endswith("s"):
                t = t[:-1]
            elif t.endswith("ing"):
                t = t[:-3]
            elif t.endswith("ed"):
                t = t[:-2]
            stemmed.append(t)
        return stemmed

    def _get_vector(self, tokens: list[str], vocabulary: list[str]) -> list[int]:
        """Create a frequency vector based on the vocabulary."""
        counts = Counter(tokens)
        return [counts.get(word, 0) for word in vocabulary]

    def _cosine_similarity(self, v1: list[int], v2: list[int]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def retrieve_relevant(
        self, query: str, history: list[Message], top_k: int = 5
    ) -> list[Message]:
        """Retrieve the top_k most relevant messages from history using windowed TF-IDF."""
        if not history:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return history[-top_k:] if len(history) > top_k else history

        # Prepare windowed documents to capture context (e.g., User query + Assistant answer)
        # We index each message but include tokens from immediate neighbors
        docs = []
        for i, msg in enumerate(history):
            # Join current message with one before and one after for context
            context_range = range(max(0, i - 1), min(len(history), i + 2))
            context_tokens = []
            for j in context_range:
                m = history[j]
                text = ""
                if isinstance(m.content, str):
                    text = m.content
                elif isinstance(m.content, list):
                    for block in m.content:
                        if isinstance(block, ContentBlockText):
                            text += block.text
                        elif isinstance(block, dict) and isinstance(
                            block.get("text"), str
                        ):
                            text += str(block["text"])
                context_tokens.extend(self._tokenize(text))

            docs.append((msg, context_tokens))

        # Build vocabulary
        vocabulary = list(set(query_tokens))
        for _, doc_tokens in docs:
            vocabulary.extend([t for t in doc_tokens if t not in vocabulary])
        vocabulary = list(set(vocabulary))

        query_vector = self._get_vector(query_tokens, vocabulary)

        scored_docs = []
        for msg, doc_tokens in docs:
            doc_vector = self._get_vector(doc_tokens, vocabulary)
            score = self._cosine_similarity(query_vector, doc_vector)
            scored_docs.append((msg, score))

        # Sort by score descending, then by original index to preserve some order for ties
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate while preserving order if same message appears multiple times in results
        # (Though in this implementation it won't)
        selected = []
        seen_ids = set()
        for msg, score in scored_docs:
            if score <= 0 and len(selected) >= top_k:
                break
            mid = id(msg)
            if mid not in seen_ids:
                selected.append(msg)
                seen_ids.add(mid)
            if len(selected) >= top_k:
                break

        return selected
