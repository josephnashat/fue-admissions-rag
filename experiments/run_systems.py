"""Run every system on Benchmark B and store answers, contexts, LLM-call counts and latency.

Systems share the LLM (gpt-4o-mini), embeddings (text-embedding-3-small), Pinecone index and
generation prompt (Appendix A.4), so differences come from control flow only.
  B1  LLM-only            no retrieval
  B2  Naive RAG           one top-4 retrieval (similarity floor 0.6) + generation
  B3  CRAG (prompted)     retrieval evaluator per chunk + decompose-then-recompose, web search disabled
  P   Proposed            the deployed LangGraph state machine (advanced mode)
  P-noT / P-noG / P-noA / P-noR   ablations: no query transformation / no document grading /
                          no answer assessment / no learned-question reuse
Writes to the live learned-questions index are disabled, so the evaluation leaves production data unchanged.
Usage: python run_systems.py <system> [limit]
"""
import os, sys, json, time, csv, threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROJ = os.path.join(ROOT, "fue_research_project_admin_part")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, PROJ)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from utilities._globals import GLOBAL_VARIABLES as GV
from utilities.self_rag.graph_state import GraphState
from utilities.self_rag.answer_generation_chain import get_rag_chain
import utilities.self_rag.nodes as N
import utilities.self_rag.edges as E


from langchain_community.vectorstores import Chroma
from utilities._admin_utils import load_document, chunk_data_semantic, load_text_question_answer

# ---------------- local rebuild of the two Pinecone indexes (same embeddings, chunking, k and thresholds)
INDEX_DIR = os.path.join(HERE, "index")
COSINE = {"hnsw:space": "cosine"}


def documents_store():
    store = Chroma("future-university-in-egypt", GV.Open_AI_Embedding, persist_directory=INDEX_DIR,
                   collection_metadata=COSINE)
    if store._collection.count() == 0:
        chunks = []
        for f in sorted(os.listdir(os.path.join(PROJ, "documents"))):
            if f.endswith(".docx"):
                chunks += chunk_data_semantic(load_document(os.path.join(PROJ, "documents", f)), GV.Open_AI_Embedding)
        store.add_documents(chunks)
    return store


DOCS = documents_store()
GV.Pinecone_KB_Retriever = DOCS.as_retriever(search_type="similarity_score_threshold",
                                             search_kwargs={"score_threshold": 0.6, "k": 4})   # deployed setting
BASE_RETRIEVER = DOCS.as_retriever(search_kwargs={"k": 4})   # baselines: plain top-4, no similarity floor


def use_learned_store(name, writable=True):
    """Fresh, empty learned-questions index per run (the deployed index starts empty in a new installation)."""
    store = Chroma(f"learned-{name}", GV.Open_AI_Embedding, persist_directory=os.path.join(INDEX_DIR, f"learned_{name}"), collection_metadata=COSINE)
    store.delete_collection()
    store = Chroma(f"learned-{name}", GV.Open_AI_Embedding, persist_directory=os.path.join(INDEX_DIR, f"learned_{name}"), collection_metadata=COSINE)
    GV.Pinecone_KB_AI_Retriever = store.as_retriever(search_type="similarity_score_threshold",
                                                     search_kwargs={"score_threshold": 0.9, "k": 4})

    async def write(question, source, answer):
        if writable:
            store.add_documents([load_text_question_answer(question, source, answer)])
    E.write_to_pinecone = write
    return store

PROFILE = {"name": "Test Applicant", "email": "applicant@example.com", "mobile": "+200000000000",
           "faculty": "not yet decided"}
_, PERSONA = N.format_user_data(PROFILE)


PRICE = {"in": 0.15e-6, "out": 0.60e-6}   # gpt-4o-mini, USD per token
BUDGET_CAP = 4.50
_spend_lock = threading.Lock()
SPEND_DIR = os.path.join(HERE, "runs", "spend")


def add_spend(usd):
    """Each process keeps its own spend file; the cap is checked against the sum of all files."""
    os.makedirs(SPEND_DIR, exist_ok=True)
    f = os.path.join(SPEND_DIR, f"{os.getpid()}_{sys.argv[1] if len(sys.argv) > 1 else 'x'}.json")
    with _spend_lock:
        cur = json.load(open(f))["usd"] if os.path.exists(f) else 0.0
        json.dump({"usd": cur + usd}, open(f, "w"))


def spent():
    if not os.path.isdir(SPEND_DIR):
        return 0.0
    total = 0.0
    for f in os.listdir(SPEND_DIR):
        try:
            total += json.load(open(os.path.join(SPEND_DIR, f)))["usd"]
        except (ValueError, OSError):
            pass
    return total


class CallCounter(BaseCallbackHandler):
    def __init__(self):
        self.n = 0; self.tin = 0; self.tout = 0; self.lock = threading.Lock()
    def on_chat_model_start(self, *a, **k):
        with self.lock: self.n += 1
    def on_llm_start(self, *a, **k):
        with self.lock: self.n += 1
    def on_llm_end(self, response, **k):
        u = (response.llm_output or {}).get("token_usage") or {}
        with self.lock:
            self.tin += u.get("prompt_tokens", 0); self.tout += u.get("completion_tokens", 0)
    @property
    def usd(self):
        return self.tin * PRICE["in"] + self.tout * PRICE["out"]


def ctx_text(docs):
    return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]


# ---------------- baselines
LLM_ONLY = ChatPromptTemplate.from_messages([("system",
    "You are an online admissions assistant for Future University in Egypt (FUE), helping prospective "
    "undergraduate applicants. Answer the applicant's question clearly and politely, using bullet points "
    "where helpful.\n\nUser Persona: {who_you_are_talking_to}\nUser Question: {question}\n"
    "You should provide your answer in English.")]) | GV.Chat_OpenAI_temp_7 | StrOutputParser()


def generate_with(docs, q):
    return get_rag_chain().invoke({"context": docs, "question": q, "answer_language": "English",
                                   "who_you_are_talking_to": PERSONA})


def b1(q, cb):
    return LLM_ONLY.invoke({"question": q, "who_you_are_talking_to": PERSONA}, config={"callbacks": [cb]}), []


def b2(q, cb):
    docs = BASE_RETRIEVER.invoke(q)
    ans = get_rag_chain().invoke({"context": docs, "question": q, "answer_language": "English",
                                  "who_you_are_talking_to": PERSONA}, config={"callbacks": [cb]})
    return ans, ctx_text(docs)


class Eval(BaseModel):
    """Relevance of a retrieved document to the question."""
    verdict: str = Field(description="'correct' if the document contains information that answers the question, "
                                     "'ambiguous' if it is related but insufficient, 'incorrect' if it is irrelevant")

class Strips(BaseModel):
    """Knowledge strips kept after refinement."""
    relevant_sentences: list[str] = Field(description="sentences copied verbatim from the document that help answer the question; empty if none")

CRAG_EVAL = ChatPromptTemplate.from_messages([("system", "You are a retrieval evaluator. Judge whether the document "
    "helps answer the question."), ("human", "Question: {question}\n\nDocument:\n{document}")]) \
    | GV.Chat_OpenAI_temp_0.with_structured_output(Eval)
CRAG_STRIP = ChatPromptTemplate.from_messages([("system", "Decompose the document into sentences and keep only the "
    "sentences that are relevant to the question. Copy them verbatim."), ("human", "Question: {question}\n\nDocument:\n{document}")]) \
    | GV.Chat_OpenAI_temp_0.with_structured_output(Strips)
CRAG_REWRITE = ChatPromptTemplate.from_messages([("system", "Rewrite the question as a concise search query for a "
    "university admissions knowledge base. Return only the query."), ("human", "{question}")]) \
    | GV.Chat_OpenAI_temp_0 | StrOutputParser()


def b3(q, cb):
    cfg = {"callbacks": [cb]}

    def evaluate(docs):
        return [(d, CRAG_EVAL.invoke({"question": q, "document": d.page_content}, config=cfg).verdict.lower()) for d in docs]

    def refine(docs):
        strips = []
        for d in docs:
            strips += CRAG_STRIP.invoke({"question": q, "document": d.page_content}, config=cfg).relevant_sentences
        return [Document(page_content=" ".join(strips))] if strips else []

    graded = evaluate(BASE_RETRIEVER.invoke(q))
    keep = [d for d, v in graded if v.startswith("correct")] or [d for d, v in graded if v.startswith("ambiguous")]
    if not keep:   # "incorrect": web search is disabled, so rewrite once and search the same index again
        q2 = CRAG_REWRITE.invoke({"question": q}, config=cfg)
        graded = evaluate(BASE_RETRIEVER.invoke(q2))
        keep = [d for d, v in graded if not v.startswith("incorrect")]
    knowledge = refine(keep)
    ans = get_rag_chain().invoke({"context": knowledge, "question": q, "answer_language": "English",
                                  "who_you_are_talking_to": PERSONA}, config=cfg)
    return ans, ctx_text(knowledge)


# ---------------- proposed model and ablations
def transform_identity(state):
    q = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    if state["check_kb"]:
        docs = GV.Pinecone_KB_AI_Retriever.invoke(q)
        docs = [d for d in docs if "source" in d.metadata and "Naive" not in d.metadata["origin"]]
        if docs:
            return {"question": q, "documents": [Document(page_content=d.metadata["answer"]) for d in docs],
                    "answered_before": True, "check_kb": False}
    return {"question": q, "generate_counter": state["generate_counter"] + 1, "answered_before": False, "check_kb": False}


def grade_passthrough(state):
    return {"documents": state["documents"], "question": state["question"]}


def domain_no_reuse(state):
    return {**N.domain_agent(state), "check_kb": False}


def no_assessment(state):
    return "answered before" if state["answered_before"] else "perfect answer"


def build(variant):
    nodes = {"kyc": N.kyc, "domain_knowledge": N.domain_agent, "transform_query": N.transform_query,
             "retrieve": N.retrieve, "grade_documents": N.grade_documents, "generate": N.generate}
    assess = E.grade_generation_v_documents_and_question
    if variant == "P-noT": nodes["transform_query"] = transform_identity
    if variant == "P-noG": nodes["grade_documents"] = grade_passthrough
    if variant == "P-noR": nodes["domain_knowledge"] = domain_no_reuse
    if variant == "P-noA": assess = no_assessment
    wf = StateGraph(GraphState)
    for k, f in nodes.items():
        wf.add_node(k, f)
    # edges identical to utilities/self_rag/graph.py
    wf.set_entry_point("kyc")
    wf.add_conditional_edges("transform_query", E.check_ai_generated_db,
                             {"question_answerd_before": "generate", "new_question": "retrieve"})
    wf.add_conditional_edges("domain_knowledge", E.decide_to_answer_question,
                             {"irrelevant_domain_knowledge": END, "relevant_domain_knowledge": "transform_query",
                              "conversation": END, "simple mode": END})
    wf.add_conditional_edges("kyc", E.decide_to_continue_kyc, {"kyc_done": "domain_knowledge", "kyc_collecting": END})
    wf.add_edge("retrieve", "grade_documents")
    wf.add_conditional_edges("grade_documents", E.decide_to_generate,
                             {"all_docs_are_not_relevant": "transform_query", "relevant_doc_exist": "generate",
                              "best possible transformation": "generate"})
    wf.add_conditional_edges("generate", assess,
                             {"hallucination-poor answer": "generate", "perfect answer": END,
                              "best possible answer": END, "answered before": END})
    return wf.compile()


def proposed(variant):
    app = build(variant)

    def run(q, cb):
        st = app.invoke({"messages": [HumanMessage(content=q, metadata={"mode": "Advanced"})], "applicant_profile": PROFILE},
                        config={"callbacks": [cb], "recursion_limit": 50})
        docs = st.get("documents") or []
        if not docs:   # answer fell back to (or ended at) the domain-evaluation step, whose context is the Chroma retrieval
            docs = GV.Chroma_Retriever.invoke(q)
        meta = {"category": st.get("domain_relevant_question"), "reused": bool(st.get("answered_before")),
                "counter": st.get("generate_counter")}
        return st.get("generation", ""), ctx_text(docs), meta
    return run


SYSTEMS = {"B1": b1, "B2": b2, "B3": b3}


def main():
    name = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fn = SYSTEMS.get(name)
    if fn is None:
        use_learned_store(name)
        fn = proposed(name)
        if os.path.exists(os.path.join(HERE, "runs", f"{name}.jsonl")):
            sys.exit(f"{name}.jsonl exists; delete it to rerun (the learned index restarts empty)")
    rows = list(csv.DictReader(open(os.path.join(HERE, "benchmark_b.csv"), encoding="utf8")))[:limit]
    out = os.path.join(HERE, "runs", f"{name}.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out, encoding="utf8")}
    lock = threading.Lock()

    def work(r):
        if r["id"] in done:
            return
        if spent() > BUDGET_CAP:
            print("BUDGET CAP REACHED, skipping", r["id"]); return
        for attempt in range(3):
            cb = CallCounter(); t0 = time.perf_counter()
            try:
                res = fn(r["question"], cb)
                break
            except Exception as e:
                err = repr(e)[:300]; time.sleep(5)
        else:
            print("FAILED", r["id"], err); return
        ans, ctx = res[0], res[1]
        add_spend(cb.usd)
        rec = {"id": r["id"], "system": name, "question": r["question"], "answer": ans, "contexts": ctx,
               "llm_calls": cb.n, "latency_s": round(time.perf_counter() - t0, 3),
               "tokens_in": cb.tin, "tokens_out": cb.tout, "usd": round(cb.usd, 5), **(res[2] if len(res) > 2 else {})}
        with lock:
            with open(out, "a", encoding="utf8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # the proposed variants run in benchmark order so learned-index reuse is reproducible
    with ThreadPoolExecutor(4 if name in SYSTEMS else 1) as ex:
        list(ex.map(work, rows))
    n = sum(1 for _ in open(out, encoding="utf8"))
    print(f"{name}: {n}/{len(rows)} answers in {out}; total spend so far ${spent():.3f}")


if __name__ == "__main__":
    main()
