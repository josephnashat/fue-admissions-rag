"""Policy-change simulation for the learned-questions index (Section 5.6, Table 11).

Phase 1: a stream of 24 Benchmark B questions (12 fee questions, 12 others) is answered by the
proposed model with an empty learned index, which fills the index as in normal operation.
Change: every figure in the tuition-fee document is raised by about 10% (controlled edit of a copy;
the institutional documents are untouched) and the documents index is rebuilt.
Phase 2: the same stream is replayed against the updated index under four configurations:
  deployed     learned index carried over, no invalidation
  provenance   safeguard (a): entries built from the re-uploaded fee document are invalidated
  expiry       safeguard (c): phase 2 runs in a new admission cycle, so phase-1 entries expire
  disabled     learned index switched off
A fee answer is stale if it quotes a superseded figure and correct if it quotes only current ones
(string match on the figures; no LLM judge).
Usage: python policy_sim.py phase1 | python policy_sim.py phase2 <config>
"""
import os, sys, json, csv, re, shutil, time
sys.argv, ARGS = sys.argv[:1] + ["sim"], sys.argv[1:]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_systems as R
from langchain_community.vectorstores import Chroma
from utilities._admin_utils import load_document, chunk_data_semantic, load_text_question_answer

SIM = os.path.join(HERE, "sim"); os.makedirs(SIM, exist_ok=True)
FEES = {"100,000": "110,000", "60,000": "66,000", "65,000": "71,500", "43,500": "47,850", "44,000": "48,400",
        "38,500": "42,350", "36,000": "39,600", "850": "935", "7,525": "8,280", "6,025": "6,630",
        "5,125": "5,640", "5,175": "5,690", "4,255": "4,680"}
FEE_DOC = "Tuition Fees.docx"


def stream():
    rows = list(csv.DictReader(open(os.path.join(HERE, "benchmark_b.csv"), encoding="utf8")))
    fees = [r for r in rows if r["category"] == "fees" and r["answerable"] == "1"][:12]
    other = [r for r in rows if r["category"] != "fees" and r["answerable"] == "1"][:12]
    return fees + other


def fee_edit(text):
    return re.sub(r"(?<![\d,])(" + "|".join(re.escape(k) for k in FEES) + r")(?![\d,])", lambda m: FEES[m.group(1)], text)


def domain_store(version):
    """Domain-evaluation index, built like chroma_db_for_domain_rag (all documents consolidated, semantic chunks)."""
    from utilities._admin_utils import semantic_chunking_with_images
    store = Chroma(f"domain-{version}", R.GV.Open_AI_Embedding, persist_directory=os.path.join(SIM, f"domain_{version}"))
    if store._collection.count() == 0:
        texts = []
        for f in sorted(os.listdir(os.path.join(R.PROJ, "documents"))):
            if f.endswith(".docx"):
                t = " ".join(d.page_content for d in load_document(os.path.join(R.PROJ, "documents", f)))
                texts.append(fee_edit(t) if version == "v2" and f == FEE_DOC else t)
        store.add_documents(semantic_chunking_with_images("\n\n ".join(texts), [], [], R.GV.Open_AI_Embedding, True, "consolidated_docs"))
    R.GV.Chroma_Retriever = store.as_retriever()
    return store


def docs_store(version):
    store = Chroma(f"docs-{version}", R.GV.Open_AI_Embedding, persist_directory=os.path.join(SIM, f"docs_{version}"),
                   collection_metadata=R.COSINE)
    if store._collection.count() == 0:
        chunks = []
        for f in sorted(os.listdir(os.path.join(R.PROJ, "documents"))):
            if not f.endswith(".docx"):
                continue
            docs = load_document(os.path.join(R.PROJ, "documents", f))
            if version == "v2" and f == FEE_DOC:
                for d in docs:
                    d.page_content = re.sub(r"\b(" + "|".join(re.escape(k) for k in FEES) + r")\b",
                                            lambda m: FEES[m.group(1)], d.page_content)
            chunks += chunk_data_semantic(docs, R.GV.Open_AI_Embedding)
        store.add_documents(chunks)
    R.GV.Pinecone_KB_Retriever = store.as_retriever(search_type="similarity_score_threshold",
                                                    search_kwargs={"score_threshold": 0.6, "k": 4})
    return store


def learned_store(path, writable=True):
    store = Chroma("learned", R.GV.Open_AI_Embedding, persist_directory=path, collection_metadata=R.COSINE)
    R.GV.Pinecone_KB_AI_Retriever = store.as_retriever(search_type="similarity_score_threshold",
                                                       search_kwargs={"score_threshold": 0.9, "k": 4})

    async def write(question, source, answer, doc_sources=None):
        if writable:
            store.add_documents([load_text_question_answer(question, source, answer, doc_sources)])
    R.E.write_to_pinecone = write
    return store


def fee_status(answer):
    a = answer.replace(" ", " ")
    old = [k for k in FEES if re.search(r"(?<![\d,])" + re.escape(k) + r"(?![\d,])", a)]
    new = [v for v in FEES.values() if re.search(r"(?<![\d,])" + re.escape(v) + r"(?![\d,])", a)]
    return {"stale": int(bool(old)), "current": int(bool(new) and not old), "old_figures": old}


def run(variant, out, no_reuse=False):
    app = R.build("P-noR" if no_reuse else "P")
    recs = []
    for r in stream():
        st, err = {}, None
        for attempt in range(3):
            cb = R.CallCounter(); t0 = time.perf_counter()
            try:
                st = app.invoke({"messages": [R.HumanMessage(content=r["question"], metadata={"mode": "Advanced"})],
                                 "applicant_profile": R.PROFILE}, config={"callbacks": [cb], "recursion_limit": 50})
                err = None; break
            except Exception as e:
                err = repr(e)[:200]
            finally:
                R.add_spend(cb.usd)
        ans = st.get("generation", "")
        rec = {"id": r["id"], "category": r["category"], "reused": bool(st.get("answered_before")),
               "llm_calls": cb.n, "latency_s": round(time.perf_counter() - t0, 2), "answer": ans, "error": err}
        if r["category"] == "fees":
            rec.update(fee_status(ans))
        recs.append(rec)
    json.dump(recs, open(out, "w", encoding="utf8"), indent=1, ensure_ascii=False)
    fees = [x for x in recs if x["category"] == "fees"]
    summ = {"variant": variant, "hit_rate": sum(x["reused"] for x in recs) / len(recs),
            "n_errors": sum(bool(x["error"]) for x in recs), "stale_rate_fees": sum(x["stale"] for x in fees) / len(fees),
            "current_rate_fees": sum(x["current"] for x in fees) / len(fees),
            "mean_llm_calls": sum(x["llm_calls"] for x in recs) / len(recs),
            "mean_latency_s": sum(x["latency_s"] for x in recs) / len(recs)}
    print(json.dumps(summ))
    return summ


if __name__ == "__main__":
    phase = ARGS[0]
    if phase == "phase1":
        shutil.rmtree(os.path.join(SIM, "learned_phase1"), ignore_errors=True)
        docs_store("v2"); domain_store("v2")      # pre-build the updated indexes used in phase 2
        docs_store("v1"); domain_store("v1"); learned_store(os.path.join(SIM, "learned_phase1"))
        run("phase1", os.path.join(SIM, "phase1.json"))
    else:
        cfg = ARGS[1]
        docs_store("v2"); domain_store("v2")      # the re-upload refreshes both indexes
        path = os.path.join(SIM, f"learned_{cfg}")
        shutil.rmtree(path, ignore_errors=True)
        shutil.copytree(os.path.join(SIM, "learned_phase1"), path)
        store = learned_store(path)
        if cfg == "provenance":
            # safeguard (a): the fee document was re-uploaded, so entries built from it are removed
            got = store._collection.get(include=["metadatas"])
            stale = [i for i, m in zip(got["ids"], got["metadatas"]) if FEE_DOC in (m.get("doc_sources") or "")]
            if stale:
                store._collection.delete(ids=stale)
            print("invalidated", len(stale), "of", len(got["ids"]), "entries")
        if cfg == "expiry":
            os.environ["ADMISSION_CYCLE"] = "next-cycle"   # safeguard (c): phase 2 belongs to a new cycle
        summ = run(cfg, os.path.join(SIM, f"phase2_{cfg}.json"), no_reuse=(cfg == "disabled"))
        json.dump(summ, open(os.path.join(SIM, f"summary_{cfg}.json"), "w"))
