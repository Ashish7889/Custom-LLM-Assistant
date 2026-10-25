"""
tools/market.py -- Real-time market data retrieval via yfinance.
Strict Non-Negotiables:
- NO simulated or fake fallback data.
- Explicit MarketDataError when a ticker is invalid or network fails.
- In-memory TTL caching to prevent rate-limit throttling.
- Indian exchange support (NSE/BSE tickers formatted with .NS or .BO).
"""

import time
import logging
from typing import Dict, Any, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger("finsage.market")

# In-memory TTL cache: {ticker: (timestamp, data)}
_MARKET_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


class MarketDataError(Exception):
    """Raised when market data cannot be fetched or ticker is invalid."""
    pass


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol and handle NSE/BSE suffixes if specified."""
    cleaned = ticker.strip().upper()
    # Normalize Indian exchange patterns: "TCS NSE" -> "TCS.NS", "RELIANCE BSE" -> "RELIANCE.BO"
    if cleaned.endswith(" NSE"):
        cleaned = cleaned[:-4].strip() + ".NS"
    elif cleaned.endswith(" BSE"):
        cleaned = cleaned[:-4].strip() + ".BO"
    return cleaned


def get_real_stock_price(ticker: str) -> Dict[str, Any]:
    """Fetch real-time stock price from yfinance with caching.
    Strictly raises MarketDataError if lookup fails (NO SIMULATED DATA).
    """
    if not YFINANCE_AVAILABLE:
        raise MarketDataError("yfinance package is not installed on this server.")

    norm_ticker = normalize_ticker(ticker)
    now = time.time()

    # Check cache
    if norm_ticker in _MARKET_CACHE:
        cached_time, cached_data = _MARKET_CACHE[norm_ticker]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_data

    try:
        stock = yf.Ticker(norm_ticker)
        info = stock.info

        # Extract current price
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            # Fall back to fast_info if info is rate-limited
            fast = getattr(stock, "fast_info", None)
            if fast and hasattr(fast, "last_price"):
                price = fast.last_price

        if price is None or price == 0.0:
            raise MarketDataError(
                f"No real-time market price available for symbol '{norm_ticker}'. "
                "Verify ticker spelling or exchange suffix (.NS for NSE, .BO for BSE)."
            )

        currency = info.get("currency") or "USD"
        short_name = info.get("shortName") or norm_ticker

        data = {
            "ticker": norm_ticker,
            "company_name": short_name,
            "current_price": round(float(price), 2),
            "currency": currency,
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "previous_close": info.get("regularMarketPreviousClose"),
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)),
            "source": "Yahoo Finance (Verified)",
        }

        # Cache result
        _MARKET_CACHE[norm_ticker] = (now, data)
        return data

    except Exception as e:
        if isinstance(e, MarketDataError):
            raise
        logger.error(f"yfinance error for ticker {norm_ticker}: {str(e)}")
        raise MarketDataError(
            f"Failed to fetch market data for '{norm_ticker}': {str(e)}"
        )


def get_real_financial_metrics(ticker: str) -> Dict[str, Any]:
    """Fetch verified valuation metrics (P/E, Market Cap, Revenue, Margins)."""
    if not YFINANCE_AVAILABLE:
        raise MarketDataError("yfinance package is not installed.")

    norm_ticker = normalize_ticker(ticker)
    try:
        stock = yf.Ticker(norm_ticker)
        info = stock.info

        pe = info.get("trailingPE") or info.get("forwardPE")
        market_cap = info.get("marketCap")
        revenue = info.get("totalRevenue")
        profit_margin = info.get("profitMargins")

        if all(v is None for v in [pe, market_cap, revenue, profit_margin]):
            raise MarketDataError(f"No fundamental valuation metrics found for symbol '{norm_ticker}'.")

        return {
            "ticker": norm_ticker,
            "company_name": info.get("shortName", norm_ticker),
            "pe_ratio": round(float(pe), 2) if pe else "N/A",
            "market_cap": market_cap,
            "total_revenue": revenue,
            "profit_margin_pct": round(float(profit_margin) * 100, 2) if profit_margin else "N/A",
            "dividend_yield_pct": round(float(info.get("dividendYield", 0)) * 100, 2) if info.get("dividendYield") else "N/A",
            "source": "Yahoo Finance Fundamentals (Verified)",
        }
    except Exception as e:
        if isinstance(e, MarketDataError):
            raise
        raise MarketDataError(f"Valuation metrics lookup failed for '{norm_ticker}': {str(e)}")
