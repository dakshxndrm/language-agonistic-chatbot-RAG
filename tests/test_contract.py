"""validate/repair are the part of the fidelity contract that must be exactly
right — a threshold bug here means script drift ships silently. Uses a fake
LLM (no network) so this is deterministic.
"""

from server.lang.contract import (
    MAX_REPAIR_ATTEMPTS,
    foreign_script_ratio,
    generate_with_contract,
    validate,
)
from server.providers.base import LangContract


class FakeLLM:
    """Yields `replies` in order, one per stream() call, ignoring its args."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls = 0

    async def stream(self, messages, contract, context):
        self.calls += 1
        for ch in self._replies.pop(0):
            yield ch


def test_foreign_script_ratio_pure_match_is_zero():
    contract = LangContract(lang="en", script="Latn")
    assert foreign_script_ratio("Ashoka ruled the Maurya empire.", contract) == 0.0


def test_foreign_script_ratio_full_mismatch_is_one():
    contract = LangContract(lang="hi", script="Deva")
    assert foreign_script_ratio("Ashoka ruled the empire", contract) == 1.0


def test_foreign_script_ratio_ignores_latin_citation_tags():
    contract = LangContract(lang="hi", script="Deva")
    # Pure Hindi answer with Latin chunk citation: should have 0.0 foreign script ratio
    text = "मौर्य साम्राज्य की स्थापना चन्द्रगुप्त मौर्य ने की थी [maurya_empire::0000] 【ashoka::0002】"
    assert foreign_script_ratio(text, contract) == 0.0


def test_validate_respects_configured_threshold(monkeypatch):
    contract = LangContract(lang="en", script="Latn")
    # "चाय" mixed into an otherwise-English reply: small foreign fraction.
    text = "Ashoka loved चाय and ruled well over many many many years indeed."
    monkeypatch.setattr("server.lang.contract.get_settings",
                        lambda: type("S", (), {"fidelity_script_threshold": 0.5})())
    assert validate(text, contract)
    monkeypatch.setattr("server.lang.contract.get_settings",
                        lambda: type("S", (), {"fidelity_script_threshold": 0.01})())
    assert not validate(text, contract)


async def test_generate_with_contract_passes_first_try(monkeypatch):
    monkeypatch.setattr("server.lang.contract.get_settings",
                        lambda: type("S", (), {"fidelity_script_threshold": 0.15})())
    llm = FakeLLM(["Ashoka ruled well."])
    contract = LangContract(lang="en", script="Latn")
    text, passed, attempts = await generate_with_contract(llm, [], contract, [])
    assert text == "Ashoka ruled well."
    assert passed
    assert attempts == 0
    assert llm.calls == 1


async def test_generate_with_contract_repairs_once_then_passes(monkeypatch):
    monkeypatch.setattr("server.lang.contract.get_settings",
                        lambda: type("S", (), {"fidelity_script_threshold": 0.15})())
    llm = FakeLLM(["अशोक ने राज किया", "Ashoka ruled well."])
    contract = LangContract(lang="en", script="Latn")
    text, passed, attempts = await generate_with_contract(llm, [], contract, [])
    assert text == "Ashoka ruled well."
    assert passed
    assert attempts == 1
    assert llm.calls == 2


async def test_generate_with_contract_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("server.lang.contract.get_settings",
                        lambda: type("S", (), {"fidelity_script_threshold": 0.15})())
    llm = FakeLLM(["अशोक एक"] * (MAX_REPAIR_ATTEMPTS + 1))
    contract = LangContract(lang="en", script="Latn")
    text, passed, attempts = await generate_with_contract(llm, [], contract, [])
    assert not passed
    assert attempts == MAX_REPAIR_ATTEMPTS
    assert llm.calls == MAX_REPAIR_ATTEMPTS + 1
