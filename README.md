# Agent Debugger

Agent Debugger provides a visual interface for planning and observing AI agent task execution.  
Both Flask and FastAPI frontends are available and share a common execution engine.

## Features
- Task planning and cost estimation
- Real-time WebSocket updates during execution
- Pluggable tool architecture
- Dedicated debug pages (`/debug`) to inspect reasoning steps and function calls

## Development
```bash
# Run Flask app
uv run python app.py

# Run FastAPI app
uv run python run_fastapi.py
```
