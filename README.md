## Mini Agentic Pipeline

An end-to-end minimal agentic workflow that retrieves context from a small knowledge base (KB), reasons with an LLM, optionally calls a CSV-based tool for prices, and produces an answer with a clear step-by-step trace.

### Features
- **Retriever**: Loads 8–20 KB docs, builds embeddings (OpenAI) or TF‑IDF fallback.
- **Reasoner**: Uses OpenAI Chat for tool decision and answer synthesis; heuristic fallback if no API key.
- **Actor (Tool)**: Local CSV price lookup with fuzzy matching and latency tracking.
- **Controller**: Orchestrates components with a shared state and structured logs.
- **Eval Harness**: Runs 8–12 test queries, collects latency for tool calls.

### Repo Layout
- `src/agentic_pipeline/`: Core components
- `data/kb/`: Knowledge base docs (8–20 small docs)
- `data/prices.csv`: CSV tool data
- `data/test_queries.json`: Evaluation queries (10)
- `results/`: Run artifacts and eval outputs

### Setup
1) Python 3.10+ recommended.
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Copy env template and set your key (optional):
```bash
cp .env.example .env
# set OPENAI_API_KEY in .env (or export in shell)
```

If `OPENAI_API_KEY` is set, the pipeline uses OpenAI `gpt-4o-mini` and `text-embedding-3-small`. Otherwise, it falls back to deterministic heuristics and TF‑IDF.

### Run (single query)
```bash
python -m src.main --query "What is the warranty for AlphaWidget Pro?"
```

### Evaluate
```bash
python -m src.eval.evaluate
```
Results are written to `results/`.

### Sample Queries
- **What's the return policy for accessories?**
- **Do you offer international shipping and how long does it take?**
- **How long is the warranty for AlphaWidget Pro?**
- **What is the price of BetaGadget Plus?**
- **What are the supported payment methods?**
- **Is there any bulk discount policy?**
- **What are the dimensions of DeltaDevice 2?**
- **How much does the OmegaAccessory Cable cost?**
- **Explain the difference between AlphaWidget Pro and AlphaWidget Mini**

### Design Choices
- Simple JSON traces with timestamps for each step (retrieval, reasoning, tool calls, synthesis).
- Modular prompts stored under `prompts/` with versioning (`v1`, `v2`, ...).
- Minimal dependencies; CSV tool runs offline; web/API tools can be added similarly.

### Known Limitations
- Heuristic fallback is simplistic; OpenAI improves both decisions and fluency.
- KB and CSV are small and illustrative only.

### Demo Video
https://youtu.be/JVxaDvAfyKU

### License
MIT


