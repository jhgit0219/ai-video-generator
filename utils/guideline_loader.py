"""
Guideline Loader - Dynamic injection of content guidelines for LLM prompts.

Provides modular, context-aware guideline loading to keep prompts concise
while maintaining comprehensive rules.
"""

from pathlib import Path
from typing import List, Optional
import hashlib

# Guideline directory
GUIDELINES_DIR = Path(__file__).parent.parent / "guidelines"

# Version tracking
GUIDELINES_VERSION = "v1.0"

def get_guideline_hash() -> str:
    """
    Get hash of all guideline files for versioning.

    Returns:
        SHA256 hash of concatenated guideline contents
    """
    all_content = []
    for guideline_file in sorted(GUIDELINES_DIR.glob("*.md")):
        all_content.append(guideline_file.read_text(encoding="utf-8"))

    combined = "\n".join(all_content)
    return hashlib.sha256(combined.encode()).hexdigest()[:8]


def load_guideline(section: str) -> str:
    """
    Load a specific guideline section.

    Args:
        section: Section name without .md extension
                 Options: "subject_rules", "genre_rules", "forbidden_mods", "priority_rules"

    Returns:
        Guideline content as string

    Raises:
        FileNotFoundError: If guideline section doesn't exist
    """
    guideline_path = GUIDELINES_DIR / f"{section}.md"

    if not guideline_path.exists():
        raise FileNotFoundError(
            f"Guideline section '{section}' not found at {guideline_path}"
        )

    return guideline_path.read_text(encoding="utf-8")


def load_guidelines(sections: List[str]) -> str:
    """
    Load multiple guideline sections and combine them.

    Args:
        sections: List of section names to load

    Returns:
        Combined guideline content

    Example:
        >>> load_guidelines(["subject_rules", "genre_rules"])
    """
    guidelines = []

    for section in sections:
        try:
            content = load_guideline(section)
            guidelines.append(content)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    return "\n\n".join(guidelines)


def get_content_analyzer_guidelines() -> str:
    """
    Get guidelines for ContentAnalyzer agent.

    Returns:
        Subject rules + brief genre guidance (~800 tokens)
    """
    return load_guideline("subject_rules")


def get_query_refinement_guidelines() -> str:
    """
    Get guidelines for query refinement (Director Agent).

    Returns:
        Subject rules + forbidden modifications (~1000 tokens)
    """
    return load_guidelines(["subject_rules", "forbidden_mods"])


def get_ranking_guidelines() -> str:
    """
    Get guidelines for image ranking/selection.

    Returns:
        Priority rules + brief subject preservation (~600 tokens)
    """
    return load_guideline("priority_rules")


def get_genre_classification_guidelines() -> str:
    """
    Get guidelines for genre classification tasks.

    Returns:
        Genre rules only (~700 tokens)
    """
    return load_guideline("genre_rules")


def get_comprehensive_guidelines() -> str:
    """
    Get all guidelines (for reference/documentation only, NOT for LLM injection).

    Returns:
        All guideline sections combined (~2500 tokens)
    """
    return load_guidelines([
        "subject_rules",
        "genre_rules",
        "forbidden_mods",
        "priority_rules"
    ])


def get_guideline_summary() -> str:
    """
    Get 3-line summary for system prompts (minimal injection).

    Returns:
        Ultra-concise guideline summary (~50 tokens)
    """
    return """
CRITICAL RULES:
1. Always preserve gender, age, and species in all query refinements
2. Never genericize or swap subjects (girl→child, cat→pet are FORBIDDEN)
3. Match visual style to genre (fantasy→illustrated, historical→photorealistic)
    """.strip()


# Pre-load common combinations for performance
_CACHE = {}

def get_cached_guidelines(guideline_type: str) -> str:
    """
    Get guidelines with caching for performance.

    Args:
        guideline_type: One of "content_analyzer", "query_refinement", "ranking", "genre", "summary"

    Returns:
        Cached or freshly loaded guidelines
    """
    if guideline_type in _CACHE:
        return _CACHE[guideline_type]

    loaders = {
        "content_analyzer": get_content_analyzer_guidelines,
        "query_refinement": get_query_refinement_guidelines,
        "ranking": get_ranking_guidelines,
        "genre": get_genre_classification_guidelines,
        "summary": get_guideline_summary,
        "comprehensive": get_comprehensive_guidelines
    }

    if guideline_type not in loaders:
        raise ValueError(f"Unknown guideline type: {guideline_type}")

    content = loaders[guideline_type]()
    _CACHE[guideline_type] = content

    return content


# Guideline metadata
GUIDELINE_SIZES = {
    "subject_rules": "~600 tokens",
    "genre_rules": "~700 tokens",
    "forbidden_mods": "~800 tokens",
    "priority_rules": "~600 tokens",
    "summary": "~50 tokens",
    "content_analyzer": "~600 tokens (subject_rules only)",
    "query_refinement": "~1400 tokens (subject_rules + forbidden_mods)",
    "ranking": "~600 tokens (priority_rules only)",
    "comprehensive": "~2500 tokens (ALL - not recommended for LLM injection)",
}


def print_guideline_info():
    """Print information about available guidelines."""
    print("=" * 80)
    print(f"Content Guidelines {GUIDELINES_VERSION} (hash: {get_guideline_hash()})")
    print("=" * 80)
    print("\nAvailable sections:")
    for name, size in GUIDELINE_SIZES.items():
        print(f"  - {name:25s} {size}")
    print("\nUsage:")
    print("  from utils.guideline_loader import get_cached_guidelines")
    print("  guidelines = get_cached_guidelines('content_analyzer')")
    print("=" * 80)


if __name__ == "__main__":
    # Test guideline loading
    print_guideline_info()

    print("\nTesting guideline loading...")
    for guideline_type in ["content_analyzer", "query_refinement", "ranking", "summary"]:
        try:
            content = get_cached_guidelines(guideline_type)
            print(f"[OK] {guideline_type}: {len(content)} chars loaded")
        except Exception as e:
            print(f"[ERROR] {guideline_type}: {e}")
