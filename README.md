# UdaPlay — AI Research Agent for the Video Game Industry

RAG + tool-using agent that answers questions about games from a curated catalog, evaluates retrieval quality, and can fall back to web search.

## Streamlit demo (recruiter showcase)

### Local

```bash
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY (optional: TAVILY_API_KEY)
streamlit run streamlit_app.py
```

### Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with **Main file path**: `streamlit_app.py`
4. In **Secrets**, add:

```toml
OPENAI_API_KEY = "sk-..."
# optional web-search fallback
TAVILY_API_KEY = "tvly-..."
# only if using Vocareum / custom gateway
# OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
```

Also works on **Render** (Web Service → `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`).

## Project layout

- `streamlit_app.py` — interactive demo UI
- `udaplay_runtime.py` — agent + Chroma indexing
- `games/` — game catalog JSON documents
- `lib/` — agent, RAG, LLM, and tooling libraries
- notebooks — original course walkthrough

## Notebooks

See `Udaplay_01_starter_project.ipynb` and `Udaplay_02_starter_project.ipynb` for the full pipeline.
