# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Key Patterns and Conventions

1. **Tool Development**: New tools should inherit from `BaseTool` and implement:
   - `name`: Tool identifier
   - `description`: Human-readable description
   - `execute()`: Main logic returning `ToolResult`
   - `estimate_cost()`: Cost prediction logic

2. **Task Execution Flow**:
   - Task created via REST API → Agent plans steps → Cost estimated → WebSocket execution with real-time updates
   - Each step tracks tokens, costs, and execution time

3. **Error Handling**: Tools return structured `ToolResult` with success/error fields. WebSocket emits error events for client handling.

4. **Token/Cost Tracking**: Built into every tool execution with configurable rates (default: $2/1M tokens)

## API Endpoints

- `POST /api/tasks`: Create new task with description and strategy
- `GET /api/tasks/<task_id>`: Get task details and status
- WebSocket events: `task_started`, `step_started`, `step_completed`, `task_completed`, `error`

## Running the Application
