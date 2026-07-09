import json
import re
from langchain_ollama import OllamaLLM

# Initialize Ollama LLM once
try:
    llm = OllamaLLM(model="tinyllama")
except Exception:
    llm = None

def query_llm_json(prompt: str, fallback: dict) -> dict:
    """
    Queries the LLM and forces parsing as JSON.
    Returns `fallback` dictionary if it fails.
    """
    if not llm:
        return fallback
        
    try:
        response = llm.invoke(prompt)
        return _extract_json(response)
    except Exception as e:
        print(f"[WARN] LLM JSON query failed: {e}")
        return fallback

def _extract_json(text: str) -> dict:
    """
    Attempts to extract JSON from an LLM response string.
    Finds the first '{' and last '}' to strip markdown.
    """
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except Exception:
        pass
    
    # If standard JSON parsing fails, fallback to simple regex cleaning or return empty
    raise ValueError("Valid JSON not found in response.")
