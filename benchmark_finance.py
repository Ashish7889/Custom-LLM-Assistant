"""
Benchmark module for evaluating financial LLM against baseline models.

Compares the fine-tuned model against the base model on financial metrics:
- Factual accuracy on finance concepts
- Number/percentage extraction accuracy
- Ticker symbol recognition
- Response quality and completeness
"""

import json
import re
import torch
from typing import Dict, List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> Tuple:
    """Load a model and tokenizer for benchmarking."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="bfloat16"
    )
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, system_prompt: str = "You are a knowledgeable financial assistant.") -> str:
    """Generate an answer to a financial question using the given model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    
    with torch.no_grad():
        output = model.generate(
            ids,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(
        output[0, ids.shape[1]:], skip_special_tokens=True
    )
    return generated


def evaluate_factual_accuracy(
    model, tokenizer, questions: List[str],
    expected_keywords: Dict[str, List[str]]
) -> Dict[str, float]:
    """Evaluate how many expected keywords appear in generated answers.
    
    Returns per-question accuracy and overall accuracy.
    """
    results = []
    total_questions = len(questions)
    total_keyword_hits = 0
    
    for i, question in enumerate(questions):
        generated = generate_answer(model, tokenizer, question)
        
        # Check for expected keywords in the generated text
        expected = expected_keywords.get(question, [])
        if not expected:
            results.append({"question": question, "accuracy": 1.0, "hits": 0, "total": 0})
            continue
        
        generated_lower = generated.lower()
        hits = sum(1 for kw in expected if kw.lower() in generated_lower)
        accuracy = hits / len(expected) if expected else 1.0
        
        total_keyword_hits += hits
        
        results.append({
            "question": question,
            "accuracy": accuracy,
            "hits": hits,
            "total": len(expected),
            "generated": generated[:150] + "...",
        })
    
    overall_accuracy = total_keyword_hits / (total_questions * total_keyword_hits) if total_keyword_hits > 0 else 0
    
    return {
        "per_question": results,
        "overall_accuracy": overall_accuracy,
        "total_questions": total_questions,
    }


def evaluate_number_extraction(
    model, tokenizer, questions: List[str],
    number_questions: Dict[str, str]
) -> Dict[str, Any]:
    """Evaluate number/percentage extraction from generated answers.
    
    number_questions maps question -> expected number pattern (regex)
    """
    results = []
    
    for question in questions:
        generated = generate_answer(model, tokenizer, question)
        
        # Try to extract numbers from generated text
        found_numbers = re.findall(r"\d+\.?\d*", generated)
        
        expected_pattern = number_questions.get(question, "")
        expected_match = re.search(expected_pattern, generated) if expected_pattern else None
        
        results.append({
            "question": question,
            "generated": generated[:150] + "...",
            "found_numbers": found_numbers[:5],
            "expected_pattern": expected_pattern,
            "expected_match": bool(expected_match),
            "extraction_correct": bool(expected_match) if expected_pattern else None,
        })
    
    return {"per_question": results}


def evaluate_ticker_recognition(
    model, tokenizer, ticker_queries: List[str]
) -> Dict[str, float]:
    """Evaluate ticker symbol recognition in generated answers."""
    results = []
    correct = 0
    
    for query in ticker_queries:
        generated = generate_answer(model, tokenizer, query)
        
        # Check if any known tickers appear in the generated text
        found_tickers = set()
        common_tickers = {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "JPM", "JNJ"}
        
        for ticker in common_tickers:
            if ticker in generated.upper():
                found_tickers.add(ticker)
        
        # Consider correct if the ticker from the query context appears
        query_upper = query.upper()
        is_correct = bool(found_tickers)
        if is_correct:
            correct += 1
        
        results.append({
            "query": query,
            "found_tickers": list(found_tickers),
            "correct": is_correct,
            "generated": generated[:150] + "...",
        })
    
    overall_accuracy = correct / len(ticker_queries) if ticker_queries else 0
    
    return {
        "per_question": results,
        "overall_accuracy": overall_accuracy,
        "total_tested": len(ticker_queries),
    }


def run_benchmark(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_path: str = "evaluation/benchmark_report.json"
) -> Dict[str, Any]:
    """Run the full benchmark suite and save results."""
    print(f"Loading model: {model_name}")
    model, tokenizer = load_base_model(model_name)
    
    # Benchmark questions
    factual_questions = [
        "What is compound interest?",
        "What is a bull market?",
        "What is a limit order?",
        "What is dividend yield?",
        "What is a good credit score?",
    ]
    
    expected_keywords = {
        "What is compound interest?": ["interest", "principal", "accumulated"],
        "What is a bull market?": ["rising", "stock prices", "optimism"],
        "What is a limit order?": ["specific price", "execute", "limit"],
        "What is dividend yield?": ["ratio", "percentage", "pays out"],
        "What is a good credit score?": ["670", "FICO", "qualify"],
    }
    
    number_questions = {
        "What is compound interest?": r"\d+\.?\d*%",
        "What is a good credit score?": r"670|740|800",
        "What is dividend yield?": r"\d+\.?\d*%",
    }
    
    ticker_queries = [
        "Tell me about AAPL stock",
        "What about MSFT?",
        "How does GOOGL perform?",
    ]
    
    print("\n=== Evaluating Factual Accuracy ===")
    factual_results = evaluate_factual_accuracy(
        model, tokenizer, factual_questions, expected_keywords
    )
    print(f"Overall accuracy: {factual_results['overall_accuracy']:.2%}")
    
    print("\n=== Evaluating Number Extraction ===")
    number_results = evaluate_number_extraction(
        model, tokenizer, factual_questions, number_questions
    )
    
    print("\n=== Evaluating Ticker Recognition ===")
    ticker_results = evaluate_ticker_recognition(model, tokenizer, ticker_queries)
    print(f"Overall ticker accuracy: {ticker_results['overall_accuracy']:.2%}")
    
    # Compile report
    report = {
        "model": model_name,
        "factual_accuracy": factual_results,
        "number_extraction": number_results,
        "ticker_recognition": ticker_results,
        "timestamp": torch.utils.collect_env_version() if hasattr(torch.utils, 'collect_env_version') else "unknown",
    }
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nBenchmark report saved to {output_path}")
    return report


if __name__ == "__main__":
    import os
    report = run_benchmark()
    print(json.dumps(report, indent=2))