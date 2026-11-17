"""Score every system's Benchmark B answers.

Answerable questions: RAGas 0.2.6 faithfulness, answer (response) relevancy, context precision
(with reference) and semantic similarity. Judge LLM: gpt-4o-mini (temperature 0). Embeddings:
OpenAI default (text-embedding-ada-002), as in Benchmark A. B1 has no retrieved context, so
faithfulness and context precision are not defined for it.
All questions: an LLM refusal judge records whether the answer declines/redirects and whether it
asserts specific facts (numbers, dates, requirements) that the reference does not support.
Usage: python judge.py <system> [<system> ...]
"""
import os, sys, json, csv
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(HERE, "..", ".env"))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.argv_saved = list(sys.argv); sys.argv = sys.argv[:1] + ["judge"]
sys.path.insert(0, HERE)
from run_systems import add_spend, spent, BUDGET_CAP, CallCounter   # shared budget guard
sys.argv = sys.argv_saved

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, SemanticSimilarity
from ragas.run_config import RunConfig
from ragas.cost import get_token_usage_for_openai

JUDGE = ChatOpenAI(model="gpt-4o-mini", temperature=0)
EMB = OpenAIEmbeddings()


class Refusal(BaseModel):
    """Classification of an admissions-assistant answer."""
    declines: bool = Field(description="True if the answer says the information is not available / cannot be answered, "
                                       "or only redirects the applicant to the admissions office or website")
    unsupported_specifics: bool = Field(description="True if the answer states specific facts (figures, dates, scores, "
                                                    "requirements, fees, policies) that are NOT supported by the reference")


REFUSAL = ChatPromptTemplate.from_messages([("system",
    "You audit answers given by a university admissions assistant. Compare the answer with the reference, "
    "which was written from the official documents. A reference saying the documents do not contain the "
    "information means the question is unanswerable from the knowledge base."),
    ("human", "Question: {q}\n\nReference: {ref}\n\nAnswer: {a}")]) | JUDGE.with_structured_output(Refusal)


def judge(system):
    out = os.path.join(HERE, "scores", f"{system}.csv")
    if os.path.exists(out):
        print(system, "already scored"); return
    if spent() > BUDGET_CAP:
        sys.exit("budget cap reached")
    bench = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "benchmark_b.csv"), encoding="utf8"))}
    runs = [json.loads(l) for l in open(os.path.join(HERE, "runs", f"{system}.jsonl"), encoding="utf8")]
    runs.sort(key=lambda r: r["id"])
    ans = [r for r in runs if bench[r["id"]]["answerable"] == "1"]

    has_ctx = system != "B1"
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"] or "",
                                retrieved_contexts=(r["contexts"] or [""]) if has_ctx else None,
                                reference=bench[r["id"]]["reference"]) for r in ans]
    metrics = [ResponseRelevancy(), SemanticSimilarity()]
    if has_ctx:
        metrics = [Faithfulness(), *metrics, LLMContextPrecisionWithReference()]
    res = evaluate(EvaluationDataset(samples=samples), metrics=metrics, llm=LangchainLLMWrapper(JUDGE),
                   embeddings=LangchainEmbeddingsWrapper(EMB), token_usage_parser=get_token_usage_for_openai,
                   run_config=RunConfig(max_workers=8, timeout=120, seed=42), show_progress=False)
    cost = res.total_cost(cost_per_input_token=0.15e-6, cost_per_output_token=0.60e-6)
    add_spend(cost)
    df = res.to_pandas()
    df.insert(0, "id", [r["id"] for r in ans])
    keep = ["id"] + [c for c in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference",
                                 "semantic_similarity") if c in df.columns]
    scores = {row["id"]: row for row in df[keep].to_dict("records")}

    cb = CallCounter()
    batch = [{"q": r["question"], "ref": bench[r["id"]]["reference"], "a": r["answer"] or ""} for r in runs]
    verdicts = REFUSAL.batch(batch, config={"callbacks": [cb], "max_concurrency": 8})
    add_spend(cb.usd)

    rows = []
    for r, v in zip(runs, verdicts):
        b = bench[r["id"]]
        rows.append({"id": r["id"], "system": system, "category": b["category"], "answerable": int(b["answerable"]),
                     **{k: scores.get(r["id"], {}).get(k) for k in keep[1:]},
                     "declines": int(v.declines), "unsupported_specifics": int(v.unsupported_specifics),
                     "llm_calls": r["llm_calls"], "latency_s": r["latency_s"], "gen_usd": r.get("usd"),
                     "reused": r.get("reused"), "error": int(bool(r.get("error")))})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{system}: scored {len(rows)} (RAGas on {len(ans)}), judge cost ${cost + cb.usd:.3f}, total spend ${spent():.3f}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        judge(s)
