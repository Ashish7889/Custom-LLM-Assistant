"""
Tool calling module for financial data lookups.

Provides tools that the LLM can call to get real-time/financial data:
- get_stock_price(ticker): Get current stock price
- get_financial_metrics(ticker): Get key financial ratios
- search_tickers(query): Search for ticker symbols

Uses Yahoo Finance (yfinance) if available, falls back to simulated data.
"""

import json
import sys
from typing import Dict, List, Optional, Any

# Try to import yfinance, fall back to simulated data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available, using simulated data")


def get_stock_price(ticker: str) -> Dict[str, Any]:
    """Get the current stock price for a ticker symbol.

    Returns a dict with price, currency, and timestamp information.
    """
    ticker = ticker.upper().strip()

    if YFINANCE_AVAILABLE:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
            currency = info.get("currency", "USD")
            timestamp = info.get("regularMarketTimestamp", "N/A")
            return {
                "ticker": ticker,
                "price": price,
                "currency": currency,
                "timestamp": timestamp,
                "source": "yfinance",
                "success": True
            }
        except Exception as e:
            return {"ticker": ticker, "price": "Error", "error": str(e), "success": False}
    else:
        # Simulated data for common tickers
        simulated = {
            "AAPL": 185.42,
            "MSFT": 412.17,
            "GOOGL": 168.55,
            "TSLA": 248.33,
            "AMZN": 185.12,
            "NVDA": 972.15,
        }
        price = simulated.get(ticker, 0.0)
        return {
            "ticker": ticker,
            "price": price,
            "currency": "USD",
            "timestamp": "Simulated",
            "source": "simulated",
            "success": True
        }


def get_financial_metrics(ticker: str) -> Dict[str, Any]:
    """Get key financial metrics/ratios for a ticker symbol.

    Returns P/E ratio, market cap, revenue, profit margin, etc.
    """
    ticker = ticker.upper().strip()

    if YFINANCE_AVAILABLE:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "revenue": info.get("totalRevenue", "N/A"),
                "profit_margin": info.get("profitMargins", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "source": "yfinance",
                "success": True
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e), "success": False}
    else:
        # Simulated metrics
        simulated = {
            "AAPL": {"pe_ratio": 28.5, "market_cap": "2.8T", "revenue": "383B", "profit_margin": 0.25},
            "MSFT": {"pe_ratio": 35.2, "market_cap": "3.1T", "revenue": "211B", "profit_margin": 0.33},
        }
        m = simulated.get(ticker, {})
        return {
            "ticker": ticker,
            "pe_ratio": m.get("pe_ratio", "N/A"),
            "market_cap": m.get("market_cap", "N/A"),
            "revenue": m.get("revenue", "N/A"),
            "profit_margin": m.get("profit_margin", "N/A"),
            "source": "simulated",
            "success": True
        }


def search_tickers(query: str) -> List[Dict[str, str]]:
    """Search for ticker symbols matching a query string.

    Returns a list of matching tickers with company names.
    """
    query = query.lower().strip()

    if YFINANCE_AVAILABLE:
        try:
            # yf.Ticker doesn't have a search, use known list or try to parse
            # For now, use a basic approach
            return _yf_search_fallback(query)
        except Exception:
            return _manual_search(query)
    else:
        return _manual_search(query)


def _manual_search(query: str) -> List[Dict[str, str]]:
    """Manual ticker search with common companies."""
    known_tickers = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corp"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc"},
        {"symbol": "AMZN", "name": "Amazon.com Inc"},
        {"symbol": "NVDA", "name": "Nvidia Corp"},
        {"symbol": "META", "name": "Meta Platforms"},
        {"symbol": "JPM", "name": "JPMorgan Chase"},
        {"symbol": "JNJ", "name": "Johnson & Johnson"},
        {"symbol": "V", "name": "Visa Inc"},
    ]

    matches = []
    for t in known_tickers:
        if query in t["symbol"].lower() or query in t["name"].lower():
            matches.append(t)

    # If no exact matches, return top 5
    return matches[:5] if matches else [{"symbol": query.upper(), "name": f"{query} Corp"}]


def _yf_search_fallback(query: str) -> List[Dict[str, str]]:
    """Fallback using yfinance if available."""
    # yfinance doesn't have a public search API, so we just return known tickers
    return _manual_search(query)


def format_tool_result(tool_name: str, result: Dict[str, Any]) -> str:
    """Format a tool result into a readable string for the LLM context."""

    if not result.get("success", False):
        return f"Error calling {tool_name}: {result.get('error', 'unknown error')}"

    if tool_name == "get_stock_price":
        return (f"{result['ticker']}: ${result['price']} {result['currency']} "
                f"(as of {result['timestamp']})")

    if tool_name == "get_financial_metrics":
        parts = [f"{result['ticker']} financial metrics:"]
        if result.get("pe_ratio") != "N/A":
            parts.append(f"P/E: {result['pe_ratio']}")
        if result.get("market_cap") != "N/A":
            parts.append(f"Market Cap: {result['market_cap']}")
        if result.get("revenue") != "N/A":
            parts.append(f"Revenue: {result['revenue']}")
        if result.get("profit_margin") != "N/A":
            pm = result["profit_margin"]
            parts.append(f"Profit Margin: {pm*100:.1f}%" if isinstance(pm, (int, float)) else f"Profit Margin: {pm}")
        if result.get("eps") != "N/A":
            parts.append(f"EPS: {result['eps']}")
        if result.get("dividend_yield") != "N/A":
            dy = result["dividend_yield"]
            parts.append(f"Dividend Yield: {dy*100:.1f}%" if isinstance(dy, (int, float)) else f"Dividend Yield: {dy}")
        return "\n".join(parts)

    return str(result)


# Tool definitions for the LLM system
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_metrics",
            "description": "Get key financial metrics/ratios for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_tickers",
            "description": "Search for ticker symbols matching a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for company name or ticker"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def get_tool_definitions() -> List[Dict]:
    """Return the tool definitions for the LLM system."""
    return TOOL_DEFINITIONS


if __name__ == "__main__":
    # Quick test
    print("Testing tool calling module...\n")

    # Test get_stock_price
    print("1. get_stock_price(AAPL):")
    result = get_stock_price("AAPL")
    print(f"   {format_tool_result('get_stock_price', result)}\n")

    # Test get_financial_metrics
    print("2. get_financial_metrics(AAPL):")
    result = get_financial_metrics("AAPL")
    print(f"   {format_tool_result('get_financial_metrics', result)}\n")

    # Test search_tickers
    print("3. search_tickers(apple):")
    result = search_tickers("apple")
    print(f"   {result}\n")

    # Test tool definitions
    print("4. Tool definitions:")
    for td in get_tool_definitions():
        print(f"   - {td['function']['name']}: {td['function']['description']}")