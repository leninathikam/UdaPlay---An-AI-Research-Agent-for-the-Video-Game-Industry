"""Runtime helpers for the UdaPlay Streamlit demo."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lib.agents import Agent
from lib.llm import LLM
from lib.parsers import PydanticOutputParser
from lib.tooling import tool

REPO_ROOT = Path(__file__).resolve().parent
GAMES_DIR = REPO_ROOT / "games"
CHROMA_PATH = REPO_ROOT / "chromadb"
COLLECTION_NAME = "udaplay"

AGENT_INSTRUCTIONS = """
You are UdaPlay, an AI research agent for the video game industry.

Workflow for every user question:
1. Call retrieve_game with the user's question.
2. Call evaluate_retrieval with the question and retrieved documents.
3. If evaluation says documents are NOT useful, and game_web_search is available, call it.
4. Produce a final answer using the best available source.

Rules:
- Prefer internal database results when they are sufficient.
- Cite sources clearly (game name/platform/year for internal docs, URLs for web results).
- Be concise but complete.
- If using web search, mention that internal knowledge was insufficient.
""".strip()


class EvaluationReport(BaseModel):
    useful: bool = Field(description="Whether the documents are useful to answer the question")
    description: str = Field(description="Explanation of the evaluation result")


def load_env() -> Dict[str, Optional[str]]:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("UDACITY_OPENAI_API_KEY")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or ("https://openai.vocareum.com/v1" if api_key and api_key.startswith("voc-") else None)
    )
    chroma_key = os.getenv("CHROMA_OPENAI_API_KEY", api_key)
    tavily_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if chroma_key:
        os.environ["CHROMA_OPENAI_API_KEY"] = chroma_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url

    return {
        "api_key": api_key,
        "base_url": base_url,
        "chroma_key": chroma_key,
        "tavily_key": tavily_key,
    }


def index_games(collection) -> int:
    ids, documents, metadatas = [], [], []
    for file_name in sorted(os.listdir(GAMES_DIR)):
        if not file_name.endswith(".json"):
            continue
        with open(GAMES_DIR / file_name, "r", encoding="utf-8") as f:
            game = json.load(f)

        content = (
            f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) - "
            f"Genre: {game['Genre']}. Publisher: {game['Publisher']}. "
            f"{game['Description']}"
        )
        ids.append(os.path.splitext(file_name)[0])
        documents.append(content)
        metadatas.append(game)

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def build_agent(session_id: str = "default") -> Agent:
    env = load_env()
    if not env["api_key"]:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in Streamlit secrets or a local .env file."
        )

    embed_kwargs: Dict[str, Any] = {"api_key": env["chroma_key"] or env["api_key"]}
    if env["base_url"]:
        embed_kwargs["api_base"] = env["base_url"]

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(**embed_kwargs)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if collection.count() == 0:
        index_games(collection)

    judge_llm = LLM(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=env["api_key"],
        base_url=env["base_url"],
    )

    @tool
    def retrieve_game(query: str) -> List[Dict[str, Any]]:
        """Semantic search over the UdaPlay game knowledge base."""
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["metadatas", "distances"],
        )
        retrieved = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            retrieved.append({**metadata, "similarity": round(1 - distance, 3)})
        return retrieved

    @tool
    def evaluate_retrieval(question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Judge whether retrieved documents are enough to answer the question."""
        prompt = (
            "Your task is to evaluate if the documents are enough to respond the query. "
            "Give a detailed explanation, so it's possible to take an action to accept it or not.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved documents:\n{json.dumps(retrieved_docs, indent=2)}"
        )
        response = judge_llm.invoke(prompt, response_format=EvaluationReport)
        parser = PydanticOutputParser(model_class=EvaluationReport)
        report = parser.parse(response)
        return report.model_dump()

    tools = [retrieve_game, evaluate_retrieval]

    if env["tavily_key"]:
        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=env["tavily_key"])

        @tool
        def game_web_search(question: str) -> Dict[str, Any]:
            """Search the web when internal game knowledge is insufficient."""
            search_result = tavily_client.search(
                query=question,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                include_images=False,
            )
            return {
                "answer": search_result.get("answer", ""),
                "results": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "content": item.get("content"),
                        "score": item.get("score"),
                    }
                    for item in search_result.get("results", [])
                ],
                "search_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query": question,
                },
            }

        tools.append(game_web_search)

    agent = Agent(
        model_name="gpt-4o-mini",
        instructions=AGENT_INSTRUCTIONS,
        tools=tools,
        temperature=0.2,
    )
    # Keep a stable session for chat continuity inside Streamlit.
    agent.memory.create_session(session_id)
    return agent


def extract_answer(run) -> str:
    state = run.get_final_state() or {}
    messages = state.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        if content and not tool_calls:
            return content
    return "I couldn't produce an answer. Please try again."
