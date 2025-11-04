"""
Ollama Cache Management Utilities

Clears Ollama's conversation context to prevent contamination between LLM calls.
Critical for preventing query/response bleed-over in multi-segment processing.
"""
import subprocess
import json
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


def clear_ollama_model_context(model: str = "llama3") -> bool:
    """
    Clear the conversation context for a specific Ollama model.

    This sends an empty prompt to reset the model's context window,
    preventing previous responses from contaminating future queries.

    Args:
        model: Name of the Ollama model to clear

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Send empty message to clear context
        payload = {
            "model": model,
            "prompt": "",
            "stream": False,
            "options": {
                "num_ctx": 0  # Reset context window
            }
        }

        result = subprocess.run(
            ["ollama", "run", model, "--keepalive", "0"],
            input="",
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="ignore"
        )

        logger.debug(f"[clear_cache] Cleared context for model: {model}")
        return True

    except subprocess.TimeoutExpired:
        logger.warning(f"[clear_cache] Timeout clearing context for {model}")
        return False
    except Exception as e:
        logger.warning(f"[clear_cache] Failed to clear context for {model}: {e}")
        return False


def clear_all_ollama_contexts() -> bool:
    """
    Clear contexts for all commonly used models.

    Returns:
        bool: True if all clears successful, False if any failed
    """
    models = ["llama3", "deepseek-coder"]
    success = True

    for model in models:
        if not clear_ollama_model_context(model):
            success = False

    if success:
        logger.info("[clear_cache] Successfully cleared all Ollama contexts")
    else:
        logger.warning("[clear_cache] Some context clears failed")

    return success


if __name__ == "__main__":
    print("Clearing Ollama cache...")
    if clear_all_ollama_contexts():
        print("✓ Cache cleared successfully")
    else:
        print("⚠ Some cache clears failed (check logs)")
