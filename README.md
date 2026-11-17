# State-machine-controlled RAG for university admissions

Source code for the paper

> J. Nashat and A. S. Salama, "A State-Machine-Controlled Retrieval-Augmented Generation System for Centralized Admissions Document Management and Intelligent Query Handling", *International Journal of Information Technology and Computer Science (IJITCS)*.

The system answers prospective students' questions from a university's admissions documents. A LangGraph state machine profiles the applicant, checks whether a question is in scope, rewrites it, retrieves and grades document chunks, generates an answer, checks it for grounding and completeness, and reuses previously validated answers through a provenance-tagged semantic cache.

## Repository layout

| Path | Contents |
|---|---|
| `fue_research_project_admin_part/utilities/self_rag/graph.py` | The state machine (nodes and conditional transitions of Algorithm 1) |
| `fue_research_project_admin_part/utilities/self_rag/nodes.py`, `edges.py` | Node logic and routing decisions, including learned-answer reuse and its safeguards |
| `fue_research_project_admin_part/utilities/self_rag/*_chain.py` | Prompt templates (reproduced in Appendix A of the paper) |
| `fue_research_project_admin_part/utilities/_admin_utils.py` | Document ingestion, semantic chunking, indexing, provenance-based invalidation, applicant-willingness classifier |
| `fue_research_project_admin_part/utilities/_globals.py` | Models, embeddings, retrievers and thresholds |
| `fue_research_project_admin_part/home.py`, `pages/` | Streamlit applicant chat and administrator interfaces |
| `experiments/run_systems.py` | Runs the baselines (LLM-only, naive RAG, prompted corrective RAG), the proposed model and its ablations |
| `experiments/judge.py` | RAGas metrics and the refusal judge |
| `experiments/stats.py` | Mean ± SD, 95% CI, Wilcoxon signed-rank with Holm correction, effect sizes, McNemar tests and Wilson intervals |
| `experiments/chunking_eval.py` | Retrieval comparison of chunking strategies |
| `experiments/policy_sim.py` | Policy-change simulation for the learned-questions index |

## Data

The institutional documents of Future University in Egypt are not redistributed. The evaluation benchmarks, the classifier data and the evaluation outputs are available from the authors on reasonable request. To run the system on your own institution, place its documents in `fue_research_project_admin_part/documents/`; the experiment scripts expect a benchmark file `experiments/benchmark_b.csv` with the columns `id, category, question, answerable, reference, source_doc`.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install -r fue_research_project_admin_part/requirements.txt
pip install sentence-transformers chromadb langchain-experimental
cp .env.example .env                                  # then fill in the keys
```

The application uses Pinecone for the two vector indexes and PostgreSQL for LangGraph checkpoints and interaction logs:

```bash
docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
streamlit run fue_research_project_admin_part/home.py
```

## Running the evaluation

The experiment scripts rebuild both indexes in a local Chroma store with the deployed settings, so they need only an OpenAI key.

```bash
cd experiments
python run_systems.py P          # also B1, B2, B3, P-noT, P-noG, P-noA, P-noR
python judge.py P B1 B2 B3
python stats.py
python chunking_eval.py
python policy_sim.py phase1 && python policy_sim.py phase2 deployed
```

## Licence

MIT; see `LICENSE`.
