"""
config.py

Central configuration + environment loading for the Subject Chatbot project.

- Knows where the project root is
- Loads .env
- Exposes API keys, model names, index name, etc.

Usage:
    from src.config import settings

    print(settings.PROJECT_ROOT)
    print(settings.PINECONE_API_KEY[:4])
"""

import os
import pathlib
from dataclasses import dataclass

from dotenv import load_dotenv, dotenv_values


# --------------------------------------------------------------------
# 1. Helper: find project root and load .env
# --------------------------------------------------------------------
def _init_project_root() -> pathlib.Path:
    """
    Ensure we are at the project root (folder that contains .env, data/, src/).

    Works even if you run from:
    - the root itself
    - the 'research' folder (where trials.ipynb lives)
    - inside src/
    """
    cwd = pathlib.Path.cwd()

    # If you're running from "research", go up one level.
    if cwd.name == "research":
        cwd = cwd.parent

    # If you're running from src/, go up one level.
    if cwd.name == "src":
        cwd = cwd.parent

    # Now cwd should be the project root.
    return cwd


PROJECT_ROOT = _init_project_root()

# Load .env from project root, overriding any old env values.
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Optional: sanity print if you want to debug env issues.
_env_vals = dotenv_values(PROJECT_ROOT / ".env")
print(f"[config] Project root: {PROJECT_ROOT}")
print(f"[config] Keys in .env: {list(_env_vals.keys())}")


# --------------------------------------------------------------------
# 2. Dataclass with all important settings
# --------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    # Paths
    PROJECT_ROOT: pathlib.Path = PROJECT_ROOT
    DATA_DIR: pathlib.Path = PROJECT_ROOT / "data"

    # API keys (read from env)
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # Pinecone config
    INDEX_NAME: str = "subject-chatbot"
    PINECONE_DIMENSION: int = 384
    PINECONE_METRIC: str = "cosine"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # Embedding model (HuggingFace)
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM config (OpenRouter DeepSeek)
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "tngtech/deepseek-r1t2-chimera:free"
    LLM_TEMPERATURE: float = 0.0


# Create a singleton instance to import everywhere
settings = Settings()


# --------------------------------------------------------------------
# 3. Basic validation (fail fast if keys are missing)
# --------------------------------------------------------------------
def validate_settings() -> None:
    """
    Check that critical settings are present.
    Raise clear errors if something is missing.
    """
    if not settings.PINECONE_API_KEY:
        raise RuntimeError(
            "[config] Missing PINECONE_API_KEY in .env. "
            "Add it and re-run."
        )

    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError(
            "[config] Missing OPENROUTER_API_KEY in .env. "
            "Add it and re-run."
        )

    if not settings.DATA_DIR.exists():
        raise RuntimeError(
            f"[config] DATA_DIR does not exist: {settings.DATA_DIR}. "
            "Create the 'data' folder and put your PDFs there."
        )


# If you import config, we can optionally validate immediately.
# Comment out if it becomes annoying.
validate_settings()