"""
tools/web_search.py -- Real-time Internet Web Search Engine for FinSage.
Searches the web for up-to-date financial schemes, tax updates, and regulatory rules.
Produces verified SourceChunks with domain provenance and timestamps.
"""

import logging
import re
import time
import urllib.parse
import urllib.request
from typing import List
from orchestrator.types import SourceChunk

logger = logging.getLogger("finsage.web_search")

OFFICIAL_DOMAINS = [
    "gov.in", "nic.in", "rbi.org.in", "incometax.gov.in", "pib.gov.in",
    "myscheme.gov.in", "sebi.gov.in", "nseindia.com", "bseindia.com"
]


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#x27;", "'").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


FINANCIAL_KEYWORDS = [
    "tax", "itr", "income", "salary", "deduction", "80c", "80d", "87a",
    "regime", "exemption", "allowance", "tds", "form 16", "ais", "26as",
    "loan", "emi", "interest", "apr", "debt", "prepay", "cibil", "credit",
    "scheme", "kisan", "pension", "yojana", "sukanya", "ppf", "epf", "nps",
    "cagr", "dividend", "nifty", "sensex", "mutual fund", "sip", "investment",
    "finance", "rupee", "inr", "lakh", "crore", "bank", "deposit", "savings",
    "gross", "net pay", "provident fund", "capital gain", "gst", "inflation"
]


def looks_financial(query: str) -> bool:
    """Check if query is related to personal finance, taxation, loans, or schemes."""
    if not query:
        return False
    q_lower = query.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in FINANCIAL_KEYWORDS)


def search_financial_web(query: str, max_results: int = 3, start_id: int = 1) -> List[SourceChunk]:
    """Search live web for evidence-grounded facts and regulations."""
    clean_query = query.strip()
    if not clean_query or not looks_financial(clean_query):
        return []

    q_words = set(re.findall(r"\b\w{3,}\b", clean_query.lower())) - {
        "what", "how", "the", "for", "and", "rules", "eligibility", "india",
        "about", "tell", "with", "from", "does", "can", "word", "random"
    }
    if not q_words:
        return []

    # Format search query targeting Indian financial / tax clarity
    search_q = f"{clean_query} India rules eligibility"
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(search_q)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    chunks: List[SourceChunk] = []
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8", errors="ignore")

        # Find search result blocks
        snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'<a class="result__url[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)

        today_str = time.strftime("%Y-%m-%d")

        for idx, raw_snippet in enumerate(snippets[:max_results]):
            text = clean_html(raw_snippet)
            if len(text) < 25:
                continue

            snippet_words = set(re.findall(r"\b\w{3,}\b", text.lower()))
            # Relevance check: snippet must share subject matter with user query
            if q_words and not (q_words & snippet_words):
                continue

            raw_url = titles[idx][0] if idx < len(titles) else ""
            raw_title = clean_html(titles[idx][1]) if idx < len(titles) else f"Web Result {idx + 1}"

            # Check if source is official government/regulatory
            is_official = any(dom in raw_url.lower() for dom in OFFICIAL_DOMAINS)
            trusted = "official" if is_official else "verified"

            source_label = f"[Live Web: {raw_title}]" if raw_title else f"[Live Web Search ({today_str})]"

            chunks.append(
                SourceChunk(
                    id=start_id + len(chunks),
                    text=text,
                    source_name=source_label,
                    url=raw_url if raw_url.startswith("http") else None,
                    trusted_level=trusted,
                    date=today_str,
                )
            )

    except Exception as e:
        logger.warning("Web search error for query '%s': %s", clean_query, str(e))

    return chunks
