from __future__ import annotations

import ast
import re

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def parse_search(text: str) -> tuple[str, str]:
    paper = (re.search(r"PAPER:\s*(.+)", text) or [None, ""])[1].strip()
    mech = (re.search(r"MECHANISM:\s*(.+)", text) or [None, ""])[1].strip()
    return paper, mech


def _parses(block: str) -> bool:
    try:
        ast.parse(block)
        return True
    except SyntaxError:
        return False


def parse_code(text: str, marker: str | None = None) -> str:
    body = text.split("</think>")[-1] if "</think>" in text else text
    blocks = _FENCE.findall(body)
    if not blocks:
        blocks = _FENCE.findall(text)
    blocks = [b.strip() for b in blocks if b.strip()]
    if not blocks:
        return ""
    if marker:
        good = [b for b in blocks if marker in b and _parses(b)]
        if good:
            return good[-1]
    good = [b for b in blocks if _parses(b)]
    return good[-1] if good else blocks[-1]
