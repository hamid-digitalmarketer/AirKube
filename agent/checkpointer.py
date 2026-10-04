"""
LangGraph checkpointer setup (#12).

Gives the agent persistent, per-conversation memory addressed by a
`thread_id`, so chat history (`AgentState.messages`) survives across
dashboard reloads and CLI restarts instead of living only in the caller's
process memory (Streamlit session state / run_agent.py's in-memory loop).

Backed by SQLite by default (via the `langgraph-checkpoint-sqlite` package)
so state is durable across restarts with no external services required.
The database path is configurable via the `AIRKUBE_CHECKPOINT_DB` env var.
Set `AIRKUBE_CHECKPOINT_DB=:memory:` (or leave the sqlite package
uninstalled) to fall back to LangGraph's built-in in-memory `MemorySaver`,
which still supports per-thread state within a single process but does not
persist across restarts.
"""

import logging
import os
import sqlite3

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("agent_checkpointer")

DEFAULT_CHECKPOINT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checkpoints.sqlite"
)


def get_checkpointer():
    """
    Returns a LangGraph checkpointer instance.

    Tries `SqliteSaver` first (persists across restarts); falls back to
    `MemorySaver` (in-process only) if the sqlite checkpoint package isn't
    installed or the db path can't be opened, or if
    `AIRKUBE_CHECKPOINT_DB=:memory:` is explicitly requested.
    """
    db_path = os.getenv("AIRKUBE_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB)

    if db_path == ":memory:":
        logger.info("AIRKUBE_CHECKPOINT_DB=:memory: - using in-process MemorySaver.")
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        logger.info("Using SqliteSaver checkpointer at %s", db_path)
        return saver
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed - falling back to "
            "in-process MemorySaver. Conversation state will NOT survive "
            "restarts. Install with `pip install langgraph-checkpoint-sqlite` "
            "for persistence."
        )
        return MemorySaver()
    except Exception as e:
        logger.error("Failed to initialize SqliteSaver at %s: %s. Falling back "
                     "to in-process MemorySaver.", db_path, e)
        return MemorySaver()
