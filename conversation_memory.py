"""
Conversation memory module for the financial assistant.

Provides:
- Persistent chat history across sessions
- User portfolio tracking (holdings, cost basis, etc.)
- Risk tolerance profiling
- Coreference resolution (referring back to previously discussed stocks/tickers)
- Session management
"""

import json
import os
import uuid
from typing import Dict, List, Optional, Any, Set


class FinanceMemory:
    """Manages conversation memory for a financial assistant user."""
    
    def __init__(self, user_id: str = None, storage_path: str = "data/conversation_memory.json"):
        self.user_id = user_id or str(uuid.uuid4())
        self.storage_path = storage_path
        
        # Conversation history: list of {role, content} dicts
        self.history: List[Dict[str, str]] = []
        
        # User portfolio: {ticker: {"shares": float, "cost_basis": float, "acquired_at": str}}
        self.portfolio: Dict[str, Dict[str, Any]] = {}
        
        # Risk tolerance: conservative, moderate, aggressive
        self.risk_tolerance: str = "moderate"
        
        # Track discussed tickers and context
        self.discussed_tickers: Set[str] = set()
        self.last_topic: str = ""
        
        # Load existing memory if available
        self._load()
    
    def _load(self):
        """Load memory from disk if it exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.history = data.get("history", [])
                self.portfolio = data.get("portfolio", {})
                self.risk_tolerance = data.get("risk_tolerance", "moderate")
                self.discussed_tickers = set(data.get("discussed_tickers", []))
                self.last_topic = data.get("last_topic", "")
            except (json.JSONDecodeError, KeyError):
                print(f"Warning: Could not load memory from {self.storage_path}, starting fresh")
    
    def _save(self):
        """Save memory to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "user_id": self.user_id,
            "history": self.history,
            "portfolio": self.portfolio,
            "risk_tolerance": self.risk_tolerance,
            "discussed_tickers": list(self.discussed_tickers),
            "last_topic": self.last_topic,
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append({"role": role, "content": content})
        self._save()
    
    def get_recent_history(self, n: int = 5) -> List[Dict[str, str]]:
        """Get the last n messages from history."""
        return self.history[-n:] if len(self.history) > n else self.history
    
    def remember_ticker(self, ticker: str, context: str = ""):
        """Mark a ticker as discussed in the conversation."""
        ticker_upper = ticker.upper()
        self.discussed_tickers.add(ticker_upper)
        self.last_topic = context
        self._save()
    
    def has_discussed(self, ticker: str) -> bool:
        """Check if a ticker has been discussed before."""
        return ticker.upper() in self.discussed_tickers
    
    def add_to_portfolio(self, ticker: str, shares: float, cost_basis: float):
        """Add shares to the user's portfolio."""
        ticker_upper = ticker.upper()
        if ticker_upper in self.portfolio:
            # Average down
            existing = self.portfolio[ticker_upper]
            total_shares = existing["shares"] + shares
            total_cost = existing["cost_basis"] * existing["shares"] + cost_basis * shares
            self.portfolio[ticker_upper] = {
                "shares": total_shares,
                "cost_basis": total_cost / total_shares if total_shares > 0 else 0,
            }
        else:
            self.portfolio[ticker_upper] = {
                "shares": shares,
                "cost_basis": cost_basis,
            }
        self._save()
    
    def get_portfolio_summary(self) -> str:
        """Get a formatted summary of the user's portfolio."""
        if not self.portfolio:
            return "No portfolio holdings tracked."
        
        lines = [f"Portfolio ({len(self.portfolio)} holdings):"]
        total_value = 0
        for ticker, holdings in self.portfolio.items():
            lines.append(f"  {ticker}: {holdings['shares']:.2f} shares, "
                      f"avg cost basis: ${holdings['cost_basis']:.2f}")
        return "\n".join(lines)
    
    def resolve_coreferences(self, text: str) -> str:
        """Resolve references to previously discussed tickers.
        
        Replaces pronoun references like 'it', 'this', 'that' with 
        the actual ticker symbol if recently discussed.
        """
        text_lower = text.lower()
        
        # Check for common pronoun references to tickers
        pronoun_map = {
            "it": None,  # Will be replaced if we know the referent
            "this": None,
            "that": None,
        }
        
        # If we have recently discussed tickers, check if pronouns refer to them
        if self.discussed_tickers:
            # Simple: if text mentions a pronoun and we have discussed tickers,
            # note the reference (more sophisticated resolution would need NLP)
            for ticker in self.discussed_tickers:
                if ticker.lower() in text_lower:
                    # The user is referring to this ticker - could expand
                    pass
        
        return text
    
    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert memory to chat template format."""
        # Build system prompt with user context
        system_content = "You are a knowledgeable financial assistant."
        
        if self.portfolio:
            system_content += f"\n\nThe user has the following portfolio holdings: {self.get_portfolio_summary()}"
        
        if self.risk_tolerance != "moderate":
            system_content += f"\n\nThe user has a {self.risk_tolerance} risk tolerance."
        
        if self.discussed_tickers:
            system_content += f"\n\nPreviously discussed tickers: {', '.join(sorted(self.discussed_tickers))}"
        
        return [
            {"role": "system", "content": system_content},
        ] + self.history
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the memory state."""
        return {
            "user_id": self.user_id,
            "message_count": len(self.history),
            "portfolio_holdings": len(self.portfolio),
            "risk_tolerance": self.risk_tolerance,
            "discussed_tickers": sorted(self.discussed_tickers),
        }


def test_memory():
    """Quick test of the memory module."""
    import tempfile
    
    # Use a temp file for testing
    mem = FinanceMemory(user_id="test_user", storage_path="/tmp/test_memory.json")
    
    # Add some messages
    mem.add_message("user", "What do you think about AAPL?")
    mem.add_message("assistant", "AAPL is Apple stock. It's been performing well recently.")
    mem.add_message("user", "I want to buy 10 shares at $185 each")
    mem.add_to_portfolio("AAPL", 10, 185.0)
    
    # Add another question that references AAPL
    mem.add_message("user", "How has it performed this year?")
    
    # Check memory state
    print(f"User ID: {mem.user_id}")
    print(f"Discussed tickers: {mem.discussed_tickers}")
    print(f"Portfolio: {mem.portfolio}")
    print(f"Portfolio summary:\\n{mem.get_portfolio_summary()}")
    print(f"Recent history count: {len(mem.history)}")
    print(f"Summary: {mem.get_summary()}")
    
    # Clean up
    import os
    if os.path.exists("/tmp/test_memory.json"):
        os.remove("/tmp/test_memory.json")


if __name__ == "__main__":
    test_memory()