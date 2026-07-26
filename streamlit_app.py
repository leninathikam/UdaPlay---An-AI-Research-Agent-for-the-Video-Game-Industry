"""UdaPlay — recruiter-ready Streamlit demo."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# Streamlit Cloud / some Linux hosts ship an older SQLite; Chroma needs a newer one.
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st

from udaplay_runtime import build_agent, extract_answer, load_env

st.set_page_config(
    page_title="UdaPlay | AI Game Research Agent",
    page_icon="🎮",
    layout="wide",
)

GAMES_DIR = Path(__file__).resolve().parent / "games"


def load_catalog():
    games = []
    for path in sorted(GAMES_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            games.append(json.load(f))
    return games


@st.cache_resource(show_spinner="Indexing game catalog and warming agent...")
def get_agent():
    return build_agent(session_id="streamlit")


def main():
    env = load_env()

    st.title("UdaPlay")
    st.caption("AI research agent for the video game industry — RAG + tool-using agent")

    with st.sidebar:
        st.header("About")
        st.write(
            "Ask questions about classic and modern games. UdaPlay retrieves from an "
            "embedded game catalog, evaluates retrieval quality, and can fall back to web search."
        )
        st.markdown(
            "[GitHub repo](https://github.com/leninathikam/UdaPlay---An-AI-Research-Agent-for-the-Video-Game-Industry)"
        )
        st.divider()
        st.subheader("Status")
        st.write("OpenAI key:", "ready" if env["api_key"] else "missing")
        st.write("Web search (Tavily):", "ready" if env["tavily_key"] else "optional / off")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Try asking")
        for q in [
            "When was Pokémon Gold and Silver released?",
            "Recommend racing games on PlayStation",
            "What was the first 3D Mario platformer?",
        ]:
            if st.button(q, use_container_width=True):
                st.session_state.pending_prompt = q

    catalog = load_catalog()
    col_chat, col_catalog = st.columns([1.6, 1], gap="large")

    with col_catalog:
        st.subheader("Catalog snapshot")
        st.write(f"{len(catalog)} games indexed for retrieval")
        for game in catalog[:8]:
            st.markdown(
                f"**{game['Name']}** · {game['Platform']} · {game['YearOfRelease']}\n\n"
                f"{game['Genre']} — {game['Publisher']}"
            )
            st.divider()

    with col_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hi — I'm UdaPlay. Ask me about games in the catalog "
                        "(release years, platforms, genres) or broader industry questions."
                    ),
                }
            ]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.session_state.pop("pending_prompt", None) or st.chat_input(
            "Ask about a game, platform, or genre..."
        )

        if prompt:
            if not env["api_key"]:
                st.error("Add OPENAI_API_KEY in Streamlit secrets or a local .env file.")
                st.stop()

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Researching..."):
                    try:
                        agent = get_agent()
                        run = agent.invoke(prompt, session_id="streamlit")
                        answer = extract_answer(run)
                    except Exception as exc:  # noqa: BLE001 - show demo-friendly errors
                        answer = f"Something went wrong while running the agent:\n\n`{exc}`"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
