"""The Language Fidelity Contract: detect -> constrain -> validate -> repair.

Constraining generation via the system prompt (llm_groq.py's stream()) does
NOT guarantee compliance — models drift, especially mid-answer when citing
English source text. So the output script is validated after generation and
repaired with one bounded re-prompt if it drifted, rather than just hoped for.
"""

from __future__ import annotations

from server.config import get_settings
from server.lang.detect import get_detector, script_histogram
from server.providers.base import LangContract, LLMProvider

import re

from collections.abc import Awaitable, Callable

MAX_REPAIR_ATTEMPTS = 2
_CITATION_RE = re.compile(r"\[[a-zA-Z0-9_:\.\s-]+\]|【[a-zA-Z0-9_:\.\s-]+】")
_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+|\n+")


async def build_contract(user_text: str, register: str = "neutral") -> LangContract:
    """detect + constrain: the reply must mirror the user's language/script."""
    det = await get_detector().detect(user_text)
    return LangContract(lang=det.lang, script=det.script, register=register)


def foreign_script_ratio(text: str, contract: LangContract) -> float:
    """What fraction of the reply's lettered characters are NOT in the
    contract's required script — validate()'s core check.
    
    Citations (e.g. [ashoka::0001] or 【maurya_empire::0000】) are stripped first
    so their Latin chunk IDs do not falsely penalize Hindi/Devanagari answers.
    """
    clean_text = _CITATION_RE.sub("", text)
    hist = script_histogram(clean_text)
    total = sum(hist.values())
    if total == 0:
        return 0.0
    foreign = sum(n for script, n in hist.items() if script != contract.script)
    return foreign / total


def validate(text: str, contract: LangContract) -> bool:
    return foreign_script_ratio(text, contract) <= get_settings().fidelity_script_threshold


async def _repair_once(text: str, contract: LangContract, llm: LLMProvider,
                       messages: list[dict], context: list[dict]) -> str:
    correction = {
        "role": "user",
        "content": (
            f"Your previous reply used the wrong script. Rewrite your ENTIRE "
            f"reply in language '{contract.lang}', script '{contract.script}' only. "
            f"Do not mix scripts."
        ),
    }
    chunks = []
    async for tok in llm.stream([*messages, {"role": "assistant", "content": text}, correction],
                                contract, context):
        chunks.append(tok)
    return "".join(chunks)


async def generate_with_contract(
    llm: LLMProvider,
    messages: list[dict],
    contract: LangContract,
    context: list[dict],
    on_sentence: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, bool, int]:
    """Streams a first reply (yielding sentences via on_sentence if provided),
    then repairs (bounded, never an unbounded loop — C6) until it passes
    validate() or the attempt budget runs out.

    Returns (final_text, passed_validation, repair_attempts_used).
    """
    chunks = []
    buffer = ""
    async for tok in llm.stream(messages, contract, context):
        chunks.append(tok)
        if on_sentence is not None:
            buffer += tok
            match = _SPLIT_RE.search(buffer)
            while match:
                sent = buffer[:match.start()].strip()
                buffer = buffer[match.end():]
                if sent:
                    await on_sentence(sent)
                match = _SPLIT_RE.search(buffer)

    if on_sentence is not None and buffer.strip():
        await on_sentence(buffer.strip())

    text = "".join(chunks)

    attempts = 0
    while not validate(text, contract) and attempts < MAX_REPAIR_ATTEMPTS:
        text = await _repair_once(text, contract, llm, messages, context)
        attempts += 1

    return text, validate(text, contract), attempts
