#!/usr/bin/env python3
"""eval_fidelity.py — runs the full RAG+contract pipeline over the golden set
and measures:

  - LFR (Language Fidelity Rate), per flavour: fraction of answers whose final
    script (after contract.py's repair loop) matches the question's language.
  - False-premise handling (type=negative, 4 questions): does the model reject
    the false premise instead of confabulating an answer?
  - Out-of-scope refusal (type=out_of_scope, 2 questions): does the model
    refuse cleanly, in-language, instead of answering from parametric memory?

The false-premise/out-of-scope checks use the LLM itself as a judge (a second
call per question) — there is no gold "correctly refused" label to string-match
against, and this is the same LLM-as-judge pattern M5's RAGAS faithfulness/
relevance ablation will use, not a new approach invented for this script.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.lang.contract import generate_with_contract  # noqa: E402
from server.lang.detect import get_detector  # noqa: E402
from server.providers.base import LangContract  # noqa: E402
from server.providers.llm_groq import GroqLLM  # noqa: E402
from server.rag.orchestrate import orchestrate  # noqa: E402
from server.rag.prompt import build_context, build_messages  # noqa: E402

JUDGE_PROMPT = """You are grading whether an AI assistant handled a tricky
question correctly. Question: {question}
Gold behaviour: {gold}
Assistant's answer: {answer}

Reply with strictly one word: "yes" if the assistant's answer matches the gold
behaviour (rejects the false premise / refuses the out-of-scope question,
rather than confabulating or answering anyway), or "no" otherwise."""


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


async def answer_question(q: dict) -> tuple[str, bool, int]:
    result = await orchestrate(q["question"], question_type=q["type"])
    messages = build_messages(q["question"])
    context = build_context(result)
    det = await get_detector().detect(q["question"])
    contract = LangContract(lang=det.lang, script=det.script)
    llm = GroqLLM()
    text, passed, attempts = await generate_with_contract(llm, messages, contract, context)
    return text, passed, attempts


async def judge(llm: GroqLLM, question: str, gold: str, answer: str) -> bool:
    reply = await llm.complete(
        [{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold,
                                                          answer=answer)}]
    )
    return reply.strip().lower().startswith("yes")


async def main() -> int:
    golden = load_golden(Path("data/golden_set.jsonl"))
    print(f"[info] {len(golden)} questions\n")

    llm = GroqLLM()
    per_flavour_lfr: dict[str, list[bool]] = defaultdict(list)
    false_premise_results: list[tuple[str, bool]] = []
    out_of_scope_results: list[tuple[str, bool]] = []

    for i, q in enumerate(golden, start=1):
        text, passed, attempts = await answer_question(q)
        per_flavour_lfr[q["flavour"]].append(passed)
        print(f"[{i}/{len(golden)}] {q['id']} [{q['flavour']}] "
             f"fidelity={'PASS' if passed else 'FAIL'} repairs={attempts}")

        if q["type"] == "negative":
            ok = await judge(llm, q["question"], q["gold_answer"], text)
            false_premise_results.append((q["id"], ok))
        elif q["type"] == "out_of_scope":
            ok = await judge(llm, q["question"], q["gold_answer"], text)
            out_of_scope_results.append((q["id"], ok))

    width = max(len(f) for f in per_flavour_lfr)
    print(f"\n| {'flavour':<{width}} | n  |   LFR |")
    print(f"|{'-' * (width + 2)}|----|-------|")
    for flavour in sorted(per_flavour_lfr):
        vals = per_flavour_lfr[flavour]
        print(f"| {flavour:<{width}} | {len(vals):<2} | {sum(vals) / len(vals):>5.3f} |")
    allv = [v for vals in per_flavour_lfr.values() for v in vals]
    print(f"| {'ALL':<{width}} | {len(allv):<2} | {sum(allv) / len(allv):>5.3f} |")

    fp_pass = sum(ok for _, ok in false_premise_results)
    print(f"\n[false-premise] {fp_pass}/{len(false_premise_results)}: "
         f"{false_premise_results}")
    oos_pass = sum(ok for _, ok in out_of_scope_results)
    print(f"[out-of-scope]  {oos_pass}/{len(out_of_scope_results)}: "
         f"{out_of_scope_results}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
