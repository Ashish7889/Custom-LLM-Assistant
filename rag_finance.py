"""
RAG (Retrieval-Augmented Generation) for financial data.

Uses the existing finance_instructions.jsonl as a knowledge base,
performs keyword-based retrieval, and augments generation with retrieved context.
"""

import json
import re
from typing import List, Dict, Optional


def load_finance_knowledge_base(path: str = "data/finance_instructions.jsonl") -> List[Dict]:
    """Load the finance instruction dataset as a simple knowledge base."""
    entries = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: {path} not found, returning empty knowledge base")
    return entries


def _extract_words(text: str) -> set:
    """Extract lowercase content words, filtering out common stop words."""
    stop_words = {
        "what", "is", "the", "of", "in", "a", "an", "was", "for", "and", "to",
        "how", "does", "do", "are", "by", "that", "this", "it", "with", "from",
        "at", "or", "as", "be", "on", "your", "my", "our", "tell", "me", "about",
        "who", "can", "you", "i", "he", "she", "they", "we", "no", "noo", "yes",
        "not", "u", "ur", "gave", "wrong", "answer"
    }
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return words - stop_words


def retrieve_relevant_context(
    question: str,
    knowledge_base: List[Dict],
    top_k: int = 2,
    min_score: int = 2,
) -> Optional[str]:
    """Retrieve verified financial concepts adhering to ECC RAG quality standards:
    - Pruned relevance matching (no noisy fallback chunks)
    - Instruction and output term overlap
    - Source attribution tagging
    """
    if not knowledge_base:
        return None

    question_words = _extract_words(question)
    if not question_words:
        return None

    # Core finance terminology
    finance_vocab = {
        "interest", "compound", "market", "bull", "bear", "dividend",
        "stock", "bond", "etf", "rate", "inflation", "recession",
        "portfolio", "risk", "return", "yield", "asset", "liability",
        "credit", "loan", "mortgage", "insurance", "tax", "capital",
        "equity", "share", "invest", "sell", "buy", "brokerage", "pe",
        "order", "limit", "reit", "401k", "ira", "sec", "insider"
    }

    scored = []
    for entry in knowledge_base:
        instruction = entry.get("instruction", "")
        output = entry.get("output", "")
        
        instr_words = _extract_words(instruction)
        out_words = _extract_words(output)

        # High priority: overlap with the specific concept instruction
        instr_overlap = len(question_words & instr_words)
        out_overlap = len(question_words & out_words)

        # Direct overlap is mandatory (prevent hallucinated topical matches)
        direct_matches = (instr_overlap * 3) + out_overlap
        if direct_matches < min_score:
            continue

        # Topic relevance boost
        finance_boost = len(question_words & out_words & finance_vocab)
        total_score = direct_matches + (finance_boost * 2)

        scored.append((total_score, entry))

    if not scored:
        return None  # ECC rule: Return None rather than irrelevant chunks

    scored.sort(key=lambda x: x[0], reverse=True)
    top_entries = [entry for _, entry in scored[:top_k]]

    # Format with citation attribution
    contexts = []
    for entry in top_entries:
        title = entry.get("instruction", "Financial Knowledge")
        content = entry.get("output", "").strip()
        contexts.append(f"[Verified Source: {title}]\n{content}")

    return "\n\n".join(contexts)


def augment_prompt_with_rag(
    question: str,
    knowledge_base_path: str = "data/finance_instructions.jsonl",
    system_prompt: str = "You are a knowledgeable financial assistant.",
    top_k: int = 3,
) -> str:
    """Augment a user question with retrieved financial knowledge.

    Returns the augmented question string that can be used as a user message
    in the chat template, with context prepended if relevant knowledge was found.
    """

    knowledge_base = load_finance_knowledge_base(knowledge_base_path)
    context = retrieve_relevant_context(question, knowledge_base, top_k=top_k)

    if context:
        augmented_question = f"Context: {context}\n\nQuestion: {question}"
    else:
        augmented_question = question

    return augmented_question


if __name__ == "__main__":
    # Quick test
    questions = [
        "What is compound interest?",
        "What is a bull market?",
        "How do I open a brokerage account?",
    ]

    kb = load_finance_knowledge_base()
    print(f"Loaded {len(kb)} finance knowledge entries\n")

    for q in questions:
        context = retrieve_relevant_context(q, kb, top_k=3)
        print(f"Q: {q}")
        if context:
            print(f"Retrieved context: {context[:300]}...")
        else:
            print("No context retrieved")
        print()