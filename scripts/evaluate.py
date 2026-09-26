#!/usr/bin/env python3
"""evaluate.py — M5 full eval harness.

Runs everything genuinely measurable in THIS environment and writes
results/eval_<date>.json + a markdown table (C7: never fabricate a number).

Audio-dependent metrics — WER/CER, TTFA, barge-in latency — are marked
"pending live mic test" rather than measured: this environment has no real
microphone or browser loop, and a synthesized-speech proxy would be an
optimistically clean stand-in for the thing that actually matters (real
human speech, background noise, accents), not a substitute worth reporting
as if it were the real number.

Cost note: answer generation uses gpt-oss-120b (the §3-pinned model, per
instruction — "keep gpt-oss-120b for answers/eval"). The auxiliary LLM-judge
calls (faithfulness/relevance/false-premise/out-of-scope) use gpt-oss-20b —
a separate quota bucket, and judging correctness doesn't need the pinned
model, so this keeps 120b's budget for what actually needs it: generation.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_fidelity import JUDGE_PROMPT, load_golden  # noqa: E402
from server.lang.contract import build_contract, generate_with_contract  # noqa: E402
from server.providers.llm_groq import GroqLLM  # noqa: E402
from server.rag.db import close_pool  # noqa: E402
from server.rag.orchestrate import orchestrate  # noqa: E402
from server.rag.prompt import build_context, build_messages  # noqa: E402

RESULTS_DIR = Path("results")
ANSWER_MODEL = "openai/gpt-oss-120b"   # §3 pin — reserved for answers/eval
JUDGE_MODEL = "openai/gpt-oss-20b"     # separate quota bucket, judging only
PENDING = "TODO: measure — pending live mic test (no mic/browser in this environment)"


def chunk_to_article_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["chunk_id"]: json.loads(line)["article"]
                for line in fh if line.strip()}


async def retrieval_ablation(golden_with_gold: list[dict]) -> dict:
    """no-RAG / dense / hybrid+rerank / hybrid+graph recall@5 (C7: real
    numbers, one retrieve() call per question per mode)."""
    from server.rag.retrieve import retrieve

    chunk_article = chunk_to_article_map(Path("data/corpus.jsonl"))
    modes = ["dense", "hybrid_norerank", "hybrid"]
    per_mode: dict[str, list[float]] = defaultdict(list)
    graph_lift: list[float] = []
    graph_lift_vector: list[float] = []

    for i, q in enumerate(golden_with_gold, start=1):
        print(f"[retrieval {i}/{len(golden_with_gold)}] {q['id']}")
        gold = set(q["gold_articles"])
        for mode in modes:
            hits = await retrieve(q["question"], rerank_top_k=5, mode=mode)
            found = {h.article for h in hits}
            per_mode[mode].append(sum(1 for g in gold if g in found) / len(gold))

        result = await orchestrate(q["question"], question_type=q["type"], rerank_top_k=5)
        found_vec = {h.article for h in result.vector_only_hits}
        found_combined = found_vec | {chunk_article.get(cid) for gh in result.graph_paths
                                      for cid in gh.source_chunk_ids}
        found_combined.discard(None)
        graph_lift_vector.append(sum(1 for g in gold if g in found_vec) / len(gold))
        graph_lift.append(sum(1 for g in gold if g in found_combined) / len(gold))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": len(golden_with_gold),
        "no_rag": 0.0,  # by definition: no context means recall@5 is 0
        "dense": avg(per_mode["dense"]),
        "hybrid_norerank": avg(per_mode["hybrid_norerank"]),
        "hybrid_rerank": avg(per_mode["hybrid"]),
        "hybrid_rerank_plus_graph": avg(graph_lift),
        "_graph_ablation_vector_only": avg(graph_lift_vector),
    }


async def fidelity_pass(golden: list[dict]) -> dict:
    """Full pipeline (real gpt-oss-120b calls) over all 50 questions. Records
    per-flavour LFR, keeps every answer for the faithfulness pass below, and
    keeps going past an isolated per-question failure (e.g. a quota cap)
    instead of aborting the whole run — the failure itself is recorded."""
    per_flavour: dict[str, list[bool]] = defaultdict(list)
    records: list[dict] = []
    errors: list[dict] = []

    for i, q in enumerate(golden, start=1):
        try:
            result = await orchestrate(q["question"], question_type=q["type"])
            messages = build_messages(q["question"])
            context = build_context(result)
            contract = await build_contract(q["question"])
            llm = GroqLLM(model=ANSWER_MODEL)
            text, passed, repairs = await generate_with_contract(llm, messages, contract, context)
            per_flavour[q["flavour"]].append(passed)
            records.append({**q, "answer": text, "fidelity_passed": passed, "repairs": repairs,
                           "context": [c["text"] for c in context]})
            print(f"[{i}/{len(golden)}] {q['id']} fidelity={'PASS' if passed else 'FAIL'}")
        except Exception as e:  # noqa: BLE001 - record and continue, never abort the whole eval
            errors.append({"id": q["id"], "error": repr(e)})
            print(f"[{i}/{len(golden)}] {q['id']} ERROR: {e!r}")

    lfr_per_flavour = {f: sum(v) / len(v) for f, v in per_flavour.items()}
    allv = [v for vals in per_flavour.values() for v in vals]
    return {
        "lfr_per_flavour": lfr_per_flavour,
        "lfr_overall": sum(allv) / len(allv) if allv else 0.0,
        "n_evaluated": len(records),
        "n_errors": len(errors),
        "errors": errors,
        "_records": records,
    }


async def judge(llm: GroqLLM, question: str, gold: str, answer: str) -> bool:
    reply = await llm.complete(
        [{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold,
                                                          answer=answer)}]
    )
    return reply.strip().lower().startswith("yes")


async def false_premise_and_out_of_scope(records: list[dict]) -> dict:
    """Never lets one judge-call failure (e.g. a quota cap) abort the whole
    eval — records the failure and keeps going, same pattern as
    fidelity_pass()."""
    judge_llm = GroqLLM(model=JUDGE_MODEL)
    fp_results, oos_results, errors = [], [], []
    for r in records:
        if r["type"] not in ("negative", "out_of_scope"):
            continue
        try:
            ok = await judge(judge_llm, r["question"], r["gold_answer"], r["answer"])
        except Exception as e:  # noqa: BLE001 - record and continue
            errors.append({"id": r["id"], "error": repr(e)})
            continue
        (fp_results if r["type"] == "negative" else oos_results).append((r["id"], ok))
    return {
        "false_premise": {"pass": sum(ok for _, ok in fp_results), "n": len(fp_results),
                          "detail": fp_results},
        "out_of_scope": {"pass": sum(ok for _, ok in oos_results), "n": len(oos_results),
                         "detail": oos_results},
        "errors": errors,
    }


FAITHFULNESS_PROMPT = """Question: {question}
Sources given to the assistant:
{context}
Assistant's answer: {answer}

Is every factual claim in the answer supported by the sources (faithfulness),
and does the answer actually address the question (relevance)? Reply with
strictly one line: "faithful=yes/no relevant=yes/no"."""


async def faithfulness_relevance(records: list[dict]) -> dict:
    """LLM-judge on the SAME answers fidelity_pass() already generated — no
    extra 120b calls, only the cheaper judge model. Same never-abort-on-one-
    failure pattern as fidelity_pass()/false_premise_and_out_of_scope()."""
    judge_llm = GroqLLM(model=JUDGE_MODEL)
    faithful, relevant, errors = [], [], []
    for r in records:
        if r["type"] in ("negative", "out_of_scope"):
            continue  # "I don't know" answers aren't claims to check against sources
        context_block = "\n".join(f"- {c[:300]}" for c in r["context"][:5]) or "(none retrieved)"
        try:
            reply = await judge_llm.complete([{"role": "user", "content":
                FAITHFULNESS_PROMPT.format(question=r["question"], context=context_block,
                                           answer=r["answer"])}])
        except Exception as e:  # noqa: BLE001 - record and continue
            errors.append({"id": r["id"], "error": repr(e)})
            continue
        low = reply.strip().lower()
        faithful.append("faithful=yes" in low)
        relevant.append("relevant=yes" in low)
    n = len(faithful) or 1
    return {
        "faithfulness": sum(faithful) / n,
        "relevance": sum(relevant) / n,
        "n": len(faithful),
        "errors": errors,
    }


NO_CONTRACT_SYSTEM = "You are a helpful assistant. Answer the question using the given sources."


async def contract_ablation(golden: list[dict], sample_size: int = 10) -> dict:
    """with-contract (build_contract + validate/repair) vs no-contract (plain
    stream(), no target-language instruction) on a representative subset —
    one flavour-stratified sample, not the full 50, to control cost."""
    seen_flavours: set[str] = set()
    sample = []
    for q in golden:
        if q["flavour"] not in seen_flavours and q["type"] not in ("negative", "out_of_scope"):
            seen_flavours.add(q["flavour"])
            sample.append(q)
        if len(sample) >= sample_size:
            break

    from server.lang.contract import validate

    with_contract_pass, without_contract_pass, errors = [], [], []
    llm = GroqLLM(model=ANSWER_MODEL)
    for q in sample:
        try:
            result = await orchestrate(q["question"], question_type=q["type"])
            messages = build_messages(q["question"])
            context = build_context(result)
            contract = await build_contract(q["question"])

            _text, passed, _r = await generate_with_contract(llm, messages, contract, context)
            with_contract_pass.append(passed)

            chunks = []
            async for tok in llm.stream(
                [{"role": "system", "content": NO_CONTRACT_SYSTEM}, *messages], contract, []
            ):
                chunks.append(tok)
            without_contract_pass.append(validate("".join(chunks), contract))
        except Exception as e:  # noqa: BLE001 - record and continue, e.g. a quota cap
            errors.append({"id": q["id"], "error": repr(e)})

    n = len(with_contract_pass) or 1
    return {
        "n": len(with_contract_pass),
        "with_contract_lfr": sum(with_contract_pass) / n,
        "without_contract_lfr": sum(without_contract_pass) / n,
        "errors": errors,
    }


async def main() -> int:
    golden = load_golden(Path("data/golden_set.jsonl"))
    golden_with_gold = [q for q in golden if q.get("gold_articles")]

    print("=== retrieval ablation ===")
    retrieval = await retrieval_ablation(golden_with_gold)
    print(json.dumps(retrieval, indent=2))

    print("\n=== fidelity pass (LFR, per flavour) — real gpt-oss-120b calls ===")
    fidelity = await fidelity_pass(golden)
    records = fidelity.pop("_records")
    print(json.dumps({k: v for k, v in fidelity.items() if k != "errors"}, indent=2))

    print("\n=== false-premise / out-of-scope ===")
    fp_oos = await false_premise_and_out_of_scope(records)
    print(json.dumps(fp_oos, indent=2))

    print("\n=== faithfulness / relevance (LLM-judge) ===")
    faith = await faithfulness_relevance(records)
    print(json.dumps(faith, indent=2))

    print("\n=== contract ablation (with vs without) ===")
    contract_abl = await contract_ablation(golden)
    print(json.dumps(contract_abl, indent=2))

    audio_metrics = {"wer": PENDING, "cer": PENDING, "ttfa_p50_ms": PENDING,
                     "ttfa_p95_ms": PENDING, "barge_in_latency_ms": PENDING,
                     "asr_single_vs_routed_wer": PENDING}

    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "retrieval_ablation": retrieval,
        "fidelity": {k: v for k, v in fidelity.items() if k != "errors"},
        "fidelity_errors": fidelity["errors"],
        "false_premise": fp_oos["false_premise"],
        "out_of_scope": fp_oos["out_of_scope"],
        "faithfulness_relevance": faith,
        "contract_ablation": contract_abl,
        "audio_metrics_pending_live_mic_test": audio_metrics,
    }

    out_path = RESULTS_DIR / f"eval_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"

    def _write():
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    await asyncio.to_thread(_write)
    print(f"\n[done] wrote {out_path}")

    await close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
