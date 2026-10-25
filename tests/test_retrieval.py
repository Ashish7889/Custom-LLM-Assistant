"""
tests/test_retrieval.py -- Unit tests for hybrid BM25 retrieval and session isolation.
"""

from retrieval.engine import SessionDocStore, OkapiBM25, tokenize_words
from orchestrator.types import SourceChunk


def test_bm25_ranking_accuracy():
    corpus = [
        tokenize_words("Apple Inc designs mobile phones and personal computers."),
        tokenize_words("The Reserve Bank of India regulates monetary policy and interest rates."),
        tokenize_words("Tesla produces electric cars and energy storage systems."),
    ]
    bm25 = OkapiBM25(corpus)
    query = tokenize_words("monetary policy interest")

    scores = [bm25.score(query, i) for i in range(len(corpus))]
    # Document 1 (RBI) must have the highest score
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_session_doc_store_isolation():
    store = SessionDocStore()
    sess_a = "session_A"
    sess_b = "session_B"

    chunk_a = SourceChunk(id=1, text="Secret fiscal numbers for Session A.", source_name="doc_a.txt")
    chunk_b = SourceChunk(id=2, text="Public market rates for Session B.", source_name="doc_b.txt")

    store.add_chunks(sess_a, [chunk_a])
    store.add_chunks(sess_b, [chunk_b])

    # Search in Session A must NEVER retrieve Session B
    results_a = store.search(sess_a, "rates market")
    assert all(c.source_name == "doc_a.txt" for c in results_a)

    results_b = store.search(sess_b, "rates market")
    assert all(c.source_name == "doc_b.txt" for c in results_b)


def test_trusted_domain_boost():
    store = SessionDocStore()
    sess_id = "test_boost_sess"

    blog_chunk = SourceChunk(id=1, text="Repo rate is 6.50% says finance blog.", source_name="random_blog.txt", trusted_level="verified")
    rbi_chunk = SourceChunk(id=2, text="Repo rate is 6.50% as published by RBI.", source_name="rbi_circular.txt", trusted_level="official")

    store.add_chunks(sess_id, [blog_chunk, rbi_chunk])
    results = store.search(sess_id, "Repo rate")

    assert len(results) == 2
    # Official source chunk must be ranked first due to the 1.4x trust boost
    assert results[0].id == 2
    assert results[0].trusted_level == "official"
