
## Project Overview

This is an Agent Debugger application that provides a visual interface for debugging AI agent workflows. It uses Flask with WebSocket support to create a real-time monitoring system for agent task execution.

## Key Architecture Components

### Core System
- **Agent Engine** (`agent_engine/core.py`): Main decision engine that creates tasks, plans execution steps, and estimates costs. Uses strategy patterns (speed, cost, accuracy, balanced) to optimize execution.
- **Tool System** (`tools/`): Pluggable tool architecture with a base interface (`base.py`) and manager (`manager.py`). Tools include web_search, text_analysis, and content_generation.
- **WebSocket Communication**: Real-time updates during task execution using Flask-SocketIO

### Data Models
- **Pydantic Schemas** (`models/schemas.py`): Type-safe data validation for all API requests/responses and WebSocket messages
- **Task System**: Tasks are broken down into steps, each using specific tools with token/cost tracking

## Common Development Commands

```bash
# Run the Flask application
uv run python app.py --reload

# Run the FastAPI application (with better type safety)
uv run python run_fastapi.py --reload
# or
uv run uvicorn app_fastapi:app --reload --port 5000

# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Update lock file
uv lock

# type check
uvx ty check .
```