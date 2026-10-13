import logging
import os
from ml.env import load_env

load_env()

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import (
    trigger_ml_pipeline,
    trigger_news_data_pipeline,
    get_pipeline_status,
    query_knowledge_graph,
    record_deployment,
    check_system_health,
    get_kg_schema,
)
from agent.checkpointer import get_checkpointer

logger = logging.getLogger("agent_graph")

SYSTEM_PROMPT = """You are AirKube, an intelligent MLOps assistant.
Your goal is to help users manage ML pipelines, query the Knowledge Graph, and ensure system health.

## Knowledge Graph Schema (Neo4j)
Node labels and properties:
- Model      (id, name, version, framework, description)
- Experiment (id, name, status, created_at)
- Run        (id, name, status, metrics, parameters)
- Deployment (id, name, cluster, image, replicas, status)

Relationships — use EXACTLY these names:
- (:Run)-[:BELONGS_TO]->(:Experiment)
- (:Model)-[:PRODUCED_BY]->(:Run)
- (:Deployment)-[:SERVES]->(:Model)

Example queries:
- All deployed models:  MATCH (d:Deployment)-[:SERVES]->(m:Model) RETURN d.name, m.name, m.version
- Experiments & runs:   MATCH (r:Run)-[:BELONGS_TO]->(e:Experiment) RETURN e.name, r.name, r.status
- Full pipeline trace:  MATCH (d)-[:SERVES]->(m)-[:PRODUCED_BY]->(r)-[:BELONGS_TO]->(e) RETURN d.name,m.name,r.name,e.name

## Guidelines
1. Always use `query_knowledge_graph` with correct Cypher when asked about models, runs, experiments, or deployments. This tool is strictly read-only.
2. Use `check_system_health` to verify API and component status.
3. Use `trigger_ml_pipeline` or `trigger_news_data_pipeline` to start pipelines.
4. After triggering a pipeline, offer to check its status with `get_pipeline_status` using the pipeline name and the execution ID the trigger returned.
5. Use `record_deployment` to record that a model has been deployed - it is the only sanctioned write path into the Knowledge Graph. Never attempt a CREATE/MERGE/DELETE via `query_knowledge_graph`.
6. Be concise. Show results clearly.
"""

tools = [
    trigger_ml_pipeline,
    trigger_news_data_pipeline,
    get_pipeline_status,
    query_knowledge_graph,
    record_deployment,
    check_system_health,
    get_kg_schema,
]

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    logger.warning("OPENROUTER_API_KEY not set — agent will fail on invocation.")

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.6"),
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)

# Cap how many recent messages from the (now checkpointer-persisted, #12)
# conversation are sent to the LLM on each turn, so per-turn token cost
# stays bounded even as a thread's full history grows across restarts.
# Callers (dashboard.py, run_agent.py) should rely on the checkpointer to
# retain full history and only pass the new message(s) for this turn into
# `app.invoke`/`app.stream` - this is where the sliding window is applied.
MAX_HISTORY_MESSAGES = 15


def _safe_history_slice(messages, max_messages):
    """
    Slices to the last `max_messages` messages without splitting an
    AIMessage(tool_calls=...) from its paired ToolMessage(s) (#19).

    A naive `[-max_messages:]` slice can start the window right in the
    middle of a tool-call/tool-response pair - e.g. keeping the
    ToolMessage but cutting off the preceding AIMessage that issued the
    tool_calls it responds to. OpenAI-compatible APIs (including
    OpenRouter) reject that shape, since every `tool` message must be
    immediately preceded in context by the assistant message that
    requested it. Walk the boundary back past any leading ToolMessage(s)
    so the window always starts on a safe boundary.
    """
    messages = list(messages)
    if len(messages) <= max_messages:
        return messages

    start = len(messages) - max_messages
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    return messages[start:]


def call_model(state: AgentState):
    recent_messages = _safe_history_slice(state["messages"], MAX_HISTORY_MESSAGES)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent_messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=get_checkpointer())
