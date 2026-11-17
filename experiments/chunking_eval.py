"""Chunking-strategy comparison (Section 3.2, Table 5a).

Each strategy chunks the same admissions documents, embeds the chunks with text-embedding-3-small and
answers the 102 answerable Benchmark B questions with top-k cosine retrieval (raw question, no rewriting).
A retrieval is a hit when a retrieved chunk comes from the document that holds the reference answer
(recorded in benchmark_b.csv), so no LLM judge is involved. Reported: chunk count and length,
document hit rate at k = 1 and 4, mean reciprocal rank (MRR@4), and the mean token count passed to the
generator at k = 4 (context cost).
"""
import os, sys, csv, json
import numpy as np

sys.argv, ARGS = sys.argv[:1] + ["chunking"], sys.argv[1:]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_systems as R   # project path, API key, embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utilities._admin_utils import load_document

EMB = R.GV.Open_AI_Embedding
DOCS = []
for f in sorted(os.listdir(os.path.join(R.PROJ, "documents"))):
    if f.endswith(".docx"):
        for d in load_document(os.path.join(R.PROJ, "documents", f)):
            d.metadata["doc"] = f
            DOCS.append(d)

STRATEGIES = {
    "Whole document (no chunking)": lambda docs: docs,
    "Fixed-size, 500 / 100 overlap": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents,
    "Fixed-size, 1,000 / 200 overlap": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents,
    "Semantic, min 600 characters": SemanticChunker(EMB, min_chunk_size=600).split_documents,
    "Semantic, min 1,500 characters (deployed)": SemanticChunker(EMB, min_chunk_size=1500).split_documents,
}

rows = [r for r in csv.DictReader(open(os.path.join(HERE, "benchmark_b.csv"), encoding="utf8")) if r["answerable"] == "1"]
Q = np.array(EMB.embed_documents([r["question"] for r in rows]))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
gold = [r["source_doc"] for r in rows]

out = {}
for name, split in STRATEGIES.items():
    chunks = split(DOCS)
    C = np.array(EMB.embed_documents([c.page_content for c in chunks]))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    top = np.argsort(-(Q @ C.T), axis=1)[:, :4]
    docs_of = [[chunks[j].metadata["doc"] for j in t] for t in top]
    hit1 = np.mean([g == d[0] for g, d in zip(gold, docs_of)])
    hit4 = np.mean([g in d for g, d in zip(gold, docs_of)])
    mrr = np.mean([next((1 / (i + 1) for i, x in enumerate(d) if x == g), 0) for g, d in zip(gold, docs_of)])
    lens = np.array([len(c.page_content) for c in chunks])
    ctx_tokens = np.mean([sum(len(chunks[j].page_content) for j in t) / 4 for t in top])   # ~4 characters per token
    out[name] = {"chunks": len(chunks), "len_mean": float(lens.mean()), "len_sd": float(lens.std(ddof=1)) if len(lens) > 1 else 0.0,
                 "hit1": float(hit1), "hit4": float(hit4), "mrr4": float(mrr), "ctx_tokens_k4": float(ctx_tokens)}
    print(f"{name:45s} chunks={len(chunks):3d} len={lens.mean():6.0f}±{out[name]['len_sd']:5.0f} "
          f"hit@1={hit1:.3f} hit@4={hit4:.3f} MRR@4={mrr:.3f} ctx_tokens@4={ctx_tokens:6.0f}")
json.dump({"n_questions": len(rows), "results": out}, open(os.path.join(HERE, "chunking_results.json"), "w"), indent=2)
