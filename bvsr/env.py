from __future__ import annotations

import os


def get(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def require(name: str, hint: str = "") -> str:
    v = get(name)
    if v is None:
        msg = f"environment variable {name} is not set"
        raise RuntimeError(msg + (f" ({hint})" if hint else ""))
    return v


def together_api_key() -> str:
    v = get("TOGETHER_API_KEY")
    if v is None:
        raise RuntimeError("environment variable TOGETHER_API_KEY is not set "
                           "(needed for the proposer models and the agentic literature search)")
    return v


def openai_api_key() -> str:
    return require("OPENAI_API_KEY", "needed for the OpenAI labelling models")


def openalex_api_key() -> str | None:
    return get("OPENALEX_API_KEY")


def openalex_mailto() -> str | None:
    return get("OPENALEX_MAILTO")
