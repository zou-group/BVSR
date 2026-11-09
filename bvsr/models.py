from __future__ import annotations

PROPOSERS = {
    "kimi":     dict(id="moonshotai/Kimi-K3",                 reasoning_effort="low"),
    "glm":      dict(id="zai-org/GLM-5.2",                    reasoning_effort=None),
    "glmflash": dict(id="zai-org/GLM-5.3-Flash",              reasoning_effort="low"),
    "dsflash":  dict(id="deepseek-ai/DeepSeek-V4-Flash-0731", reasoning_effort=None),
    "dsv4":     dict(id="deepseek-ai/DeepSeek-V4-Pro-0813",   reasoning_effort=None),
}
PROPOSER_ORDER = ["kimi", "glm", "glmflash", "dsflash", "dsv4"]

LABEL_MODELS = {
    "grouping":       dict(id="gpt-5.6-sol",  reasoning_effort="high", max_tokens=128000),
    "nonobviousness": dict(id="gpt-5.6-luna", reasoning_effort="low",  max_tokens=4000),
}

INFO_BITS_MODELS = ["Qwen/Qwen3-8B", "google/gemma-3-4b-it", "meta-llama/Llama-3.1-8B-Instruct"]


def resolve(name_or_id: str) -> tuple[str, str | None]:
    if name_or_id in PROPOSERS:
        e = PROPOSERS[name_or_id]
        return e["id"], e["reasoning_effort"]
    short = short_name(name_or_id)
    if short in PROPOSERS:
        return name_or_id, PROPOSERS[short]["reasoning_effort"]
    return name_or_id, None


def short_name(model_id: str) -> str:
    if model_id in PROPOSERS:
        return model_id
    for k, e in PROPOSERS.items():
        if e["id"] == model_id:
            return k
    return model_id
