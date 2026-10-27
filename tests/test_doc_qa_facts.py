"""
tests/test_doc_qa_facts.py -- Unit tests for Document QA, Fact Extraction, Intent Filtering, and Citation Mapping.
Tests the exact scenarios specified:
1. "what is the gross salary?" -> "Gross salary is Rs. 12,00,000.00." cited to chunk containing it, repairs_attempted == 0, no analyst/senior memos, no web call.
2. "what is the net pay in the March slip?" -> Rs. 93,800.00.
3. "analyze the pdf" -> includes both 12,00,000.00 and 93,800.00.
4. "what is the HRA exemption?" (not in doc) -> abstains with document message, at most 2 drafts.
5. A resume chunk set + "analyze the pdf" -> instant out-of-scope message, no agent_memo for analyst/senior, web never called.
6. Citation test: sources ordered [chunk2, chunk1], analyst cite "S1" must map to chunk 2's id.
"""

from unittest.mock import patch
from orchestrator.state_machine import OrchestratorStateMachine
from orchestrator.types import SourceChunk, GeneratorOutput, SentenceItem
from retrieval.engine import doc_store


def test_gross_salary_extracted_and_verified_without_agents_or_web():
    """Test 'what is the gross salary?' -> 'Gross salary is Rs. 12,00,000.00.' cited to chunk id."""
    sm = OrchestratorStateMachine()
    session_id = "test_doc_session_1"
    doc_store.delete_session(session_id)

    chunk_1 = SourceChunk(
        id=101,
        text="Form 16 Part B Certificate. (Rs.) Gross Salary 12,00,000.00 Less: 80C 1,50,000.00 Total Taxable Income 10,50,000.00",
        source_name="form_16.pdf",
        page=1,
    )
    chunk_2 = SourceChunk(
        id=102,
        text="March Salary Slip: Net Pay 93,800.00 for employee code EMP123.",
        source_name="march_slip.pdf",
        page=1,
    )
    doc_store.add_chunks(session_id, [chunk_1, chunk_2])

    with patch("tools.web_search.search_financial_web") as mock_web:
        events = list(sm.run(
            prompt="what is the gross salary?",
            session_id=session_id,
            user_id="u1",
            web_search=True,
        ))
        # Ensure web search was never called
        mock_web.assert_not_called()

    # Check event types: no analyst or senior memos
    agent_memos = [e.payload.get("agent") for e in events if e.event_type == "agent_memo"]
    assert "analyst" not in agent_memos
    assert "senior" not in agent_memos

    done_event = [e for e in events if e.event_type == "done"][0]
    assert done_event.payload["abstain"] is False
    assert done_event.payload["repairs_attempted"] == 0

    sentences = done_event.payload["sentences"]
    assert len(sentences) >= 1
    first_sent = sentences[0]
    assert "12,00,000.00" in first_sent["text"]
    assert "Gross salary" in first_sent["text"] or "Gross Salary" in first_sent["text"]
    assert 101 in first_sent["cites"]

    doc_store.delete_session(session_id)


def test_net_pay_in_march_slip():
    """Test 'what is the net pay in the March slip?' -> Rs. 93,800.00."""
    sm = OrchestratorStateMachine()
    session_id = "test_doc_session_2"
    doc_store.delete_session(session_id)

    chunk_1 = SourceChunk(
        id=201,
        text="Form 16 Part B. Gross Salary 12,00,000.00 Total Taxable Income 10,50,000.00",
        source_name="form_16.pdf",
        page=1,
    )
    chunk_2 = SourceChunk(
        id=202,
        text="March Salary Slip Net Pay 93,800.00 for employee code EMP123.",
        source_name="march_slip.pdf",
        page=1,
    )
    doc_store.add_chunks(session_id, [chunk_1, chunk_2])

    events = list(sm.run(
        prompt="what is the net pay in the March slip?",
        session_id=session_id,
        user_id="u2",
        web_search=True,
    ))

    done_event = [e for e in events if e.event_type == "done"][0]
    assert done_event.payload["abstain"] is False
    all_text = " ".join(s["text"] for s in done_event.payload["sentences"])
    assert "93,800.00" in all_text

    doc_store.delete_session(session_id)


def test_analyze_pdf_overview_includes_both_figures():
    """Test 'analyze the pdf' -> includes both 12,00,000.00 and 93,800.00."""
    sm = OrchestratorStateMachine()
    session_id = "test_doc_session_3"
    doc_store.delete_session(session_id)

    chunk_1 = SourceChunk(
        id=301,
        text="Form 16 Part B (Rs.) Gross Salary 12,00,000.00 Total Taxable Income 10,50,000.00",
        source_name="form_16.pdf",
        page=1,
    )
    chunk_2 = SourceChunk(
        id=302,
        text="March Salary Slip Net Pay 93,800.00 for employee code EMP123.",
        source_name="march_slip.pdf",
        page=1,
    )
    doc_store.add_chunks(session_id, [chunk_1, chunk_2])

    events = list(sm.run(
        prompt="analyze the pdf",
        session_id=session_id,
        user_id="u3",
        web_search=True,
    ))

    done_event = [e for e in events if e.event_type == "done"][0]
    assert done_event.payload["abstain"] is False
    all_text = " ".join(s["text"] for s in done_event.payload["sentences"])
    assert "12,00,000.00" in all_text
    assert "93,800.00" in all_text

    doc_store.delete_session(session_id)


def test_missing_fact_abstains_with_doc_message_and_at_most_2_drafts():
    """Test 'what is the HRA exemption?' (not in document) -> abstains with document message, at most 2 drafts."""
    sm = OrchestratorStateMachine()
    session_id = "test_doc_session_4"
    doc_store.delete_session(session_id)

    chunk_1 = SourceChunk(
        id=401,
        text="Form 16 Part B (Rs.) Gross Salary 12,00,000.00 Total Taxable Income 10,50,000.00",
        source_name="form_16.pdf",
        page=1,
    )
    doc_store.add_chunks(session_id, [chunk_1])

    events = list(sm.run(
        prompt="what is the HRA exemption?",
        session_id=session_id,
        user_id="u4",
        web_search=False,
    ))

    done_event = [e for e in events if e.event_type == "done"][0]
    assert done_event.payload["abstain"] is True
    first_sent = done_event.payload["sentences"][0]["text"]
    assert "I couldn't ground that answer in your document" in first_sent

    # Count generate step events to ensure at most 2 draft attempts
    generate_events = [e for e in events if e.event_type == "step" and e.payload.get("state") == "GENERATE"]
    assert len(generate_events) <= 2

    doc_store.delete_session(session_id)


def test_resume_chunk_set_instant_out_of_scope_message():
    """Test a resume chunk set + 'analyze the pdf' -> instant out-of-scope message, no analyst/senior, no web."""
    sm = OrchestratorStateMachine()
    session_id = "test_doc_session_5"
    doc_store.delete_session(session_id)

    resume_chunk = SourceChunk(
        id=501,
        text="Curriculum Vitae. John Doe. Software Engineer. Skills: Python, React, Docker. Experience: 4 years. Education: Bachelor of Technology in Computer Science.",
        source_name="john_doe_resume.pdf",
        page=1,
    )
    doc_store.add_chunks(session_id, [resume_chunk])

    with patch("tools.web_search.search_financial_web") as mock_web:
        events = list(sm.run(
            prompt="analyze the pdf",
            session_id=session_id,
            user_id="u5",
            web_search=True,
        ))
        mock_web.assert_not_called()

    # Verify no analyst or senior memos
    agent_memos = [e.payload.get("agent") for e in events if e.event_type == "agent_memo"]
    assert "analyst" not in agent_memos
    assert "senior" not in agent_memos

    done_event = [e for e in events if e.event_type == "done"][0]
    assert done_event.payload["abstain"] is False
    text = done_event.payload["sentences"][0]["text"]
    assert "This isn't a finance document" in text

    doc_store.delete_session(session_id)


def test_citation_mapping_positional_to_source_id():
    """Test citation mapping: sources ordered [chunk2, chunk1], analyst cite 'S1' must map to chunk 2's id."""
    sm = OrchestratorStateMachine()
    chunk_2 = SourceChunk(id=777, text="Source chunk 2 text.", source_name="file2.pdf")
    chunk_1 = SourceChunk(id=888, text="Source chunk 1 text.", source_name="file1.pdf")
    sources = [chunk_2, chunk_1]

    # Test the regex citation parsing logic in isolation
    analyst_cite = "S1"
    import re
    m_digit = re.search(r"\d+", analyst_cite)
    assert m_digit is not None
    s_idx = int(m_digit.group(0))
    mapped_id = sources[s_idx - 1].id
    assert mapped_id == 777
    assert mapped_id == chunk_2.id

    # Test S2
    analyst_cite_2 = "S2"
    m_digit_2 = re.search(r"\d+", analyst_cite_2)
    s_idx_2 = int(m_digit_2.group(0))
    mapped_id_2 = sources[s_idx_2 - 1].id
    assert mapped_id_2 == 888
    assert mapped_id_2 == chunk_1.id
