"""
LLM utilities for calling Ollama models.
"""

import subprocess
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Default Ollama model (can be overridden in function call)
DEFAULT_OLLAMA_MODEL = "llama3"


async def ollama_chat(system_prompt: str, user_prompt: str, model: str = None, timeout: int = 60) -> str:
    """
    Call Ollama LLM with system and user prompts.

    Args:
        system_prompt: System/role prompt
        user_prompt: User query/task
        model: Model name (defaults to OLLAMA_MODEL from config)
        timeout: Timeout in seconds

    Returns:
        LLM response text

    Raises:
        Exception: If LLM call fails or times out
    """
    if model is None:
        model = DEFAULT_OLLAMA_MODEL

    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        logger.debug(f"[llm] Calling {model} with {len(combined_prompt)} char prompt")

        result = subprocess.run(
            ["ollama", "run", model],
            input=combined_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout
        )

        # Check for errors
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"[llm] Ollama failed with return code {result.returncode}")
            logger.error(f"[llm] Error output: {error_msg[:500]}")
            raise Exception(f"Ollama process failed: {error_msg[:200]}")

        response = result.stdout.strip()

        if not response:
            stderr_preview = result.stderr[:200] if result.stderr else "No stderr"
            logger.warning(f"[llm] Empty response from {model}")
            logger.warning(f"[llm] Stderr: {stderr_preview}")
            raise Exception(f"Empty LLM response (stderr: {stderr_preview})")

        logger.debug(f"[llm] Received {len(response)} char response from {model}")
        return response

    except subprocess.TimeoutExpired:
        logger.error(f"[llm] Timeout calling {model} after {timeout}s")
        raise Exception(f"LLM timeout after {timeout}s")

    except FileNotFoundError:
        logger.error(f"[llm] Ollama command not found. Is Ollama installed?")
        logger.error(f"[llm] Install from: https://ollama.com/")
        raise Exception("Ollama not installed or not in PATH")

    except Exception as e:
        logger.error(f"[llm] Error calling {model}: {e}")
        raise
