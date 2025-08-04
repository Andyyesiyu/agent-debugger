# Agent Debugger

Agent Debugger provides a visual interface for debugging AI agent workflows. It offers both Flask and FastAPI backends with WebSocket support and integrates with a pluggable tool system and optional LLM-driven execution.

## Features
- Task planning with multiple strategies (speed, cost, accuracy, balanced)
- Real-time monitoring via WebSocket events
- LLM function calling with token and cost tracking
- Extensible tool architecture

## Development
Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run the application:

```bash
uv run python app.py                 # Flask version
uv run uvicorn app_fastapi:app --reload --port 8000  # FastAPI version
```

## License
Distributed under the MIT License. See `LICENSE` for more information.

