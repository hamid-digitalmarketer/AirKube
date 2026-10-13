import os
import sys
import uuid
import asyncio
from ml.env import load_env

load_env()

from langchain_core.messages import HumanMessage
from agent.graph import app
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def _print_stream_event(event):
    kind = event["event"]

    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            console.print(content, end="", style="cyan")
    elif kind == "on_tool_start":
        console.print(f"\n[bold magenta]Using Tool: {event['name']}[/bold magenta]")
    elif kind == "on_tool_end":
        console.print(f"[dim]Tool Output: {event['data'].get('output')}[/dim]\n")


async def _run_turn(new_messages, config):
    console.print("[dim]Agent is thinking...[/dim]")

    # Run the graph exactly once per turn via astream_events (#17) - a
    # second ainvoke() here would re-run the whole graph (including any
    # tool calls) a second time, duplicating side effects like pipeline
    # triggers. The checkpointer (#12) persists state after the stream
    # completes, so we read the final messages back from there instead of
    # invoking again.
    async for event in app.astream_events({"messages": new_messages}, version="v1", config=config):
        _print_stream_event(event)

    final_state = await app.aget_state(config)
    return final_state.values["messages"]

def _resolve_thread_id():
    """
    Resolve the LangGraph checkpointer thread_id (#12) for this CLI run.

    `--thread <id>` resumes a prior conversation (its full history is
    restored from the checkpointer's persisted storage); otherwise a new
    thread id is generated for this process.
    """
    if "--thread" in sys.argv:
        idx = sys.argv.index("--thread")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return str(uuid.uuid4())


async def main():
    console.print(Panel.fit("[bold blue]AirKube Agentic CLI[/bold blue]\n[green]Powered by LangGraph & OpenRouter (GLM-4.6)[/green]"))

    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[bold red]WARNING: OPENROUTER_API_KEY not found in environment.[/bold red]")
        console.print("Please set it via `set OPENROUTER_API_KEY=...` or `.env` file.")
        # We don't exit, just let it fail naturally if they try to use it, or they might set it now.

    thread_id = _resolve_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]Conversation thread: {thread_id} (resume later with --thread {thread_id})[/dim]")

    while True:
        try:
            user_input = console.input("[bold yellow]User > [/bold yellow]")
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            # Only the new human message is sent per turn - the checkpointer
            # persists full history for this thread_id server-side, and
            # agent/graph.py's call_model applies its own sliding window
            # before calling the LLM.
            await _run_turn([HumanMessage(content=user_input)], config)

            # Print final response if not streamed (or just a separator)
            console.print("\n" + "-"*30 + "\n")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    asyncio.run(main())
