"""Shared task execution helpers for web frameworks."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable


# Type aliases for callbacks
SyncSend = Callable[[str, dict], None]
AsyncSend = Callable[[str, dict], Awaitable[None]]


def _update_session(session: dict, result: dict) -> None:
    session["total_tokens"] += result.get("input_tokens", 0) + result.get(
        "output_tokens", 0
    )
    session["total_cost"] += result.get("cost", 0)
    session["last_output"] = str(result.get("data", ""))[:500]


def execute_agent_task(
    agent_engine: Any,
    tool_manager: Any,
    task_id: str,
    session: dict,
    send: SyncSend,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Execute a task synchronously.

    Parameters
    ----------
    agent_engine: AgentDecisionEngine
        Engine managing tasks and planning.
    tool_manager: ToolManager
        Manager used to execute tools.
    task_id: str
        Identifier for the task.
    session: dict
        Session dictionary storing progress.
    send: callable
        Callback used to emit events.
    sleep: callable, optional
        Sleep function allowing injection for testing.
    """

    if sleep is None:
        import time

        sleep = time.sleep

    task = agent_engine.get_task(task_id)
    if not task:
        send("error", {"message": "任务不存在"})
        return

    task_plan = task.steps
    for i, step in enumerate(task_plan):
        step_data = {
            "step": i + 1,
            "total_steps": len(task_plan),
            "tool": step.tool,
            "description": step.description,
        }
        send("step_started", step_data)

        input_data = task.description if i == 0 else session.get("last_output", "")
        result = tool_manager.execute_tool(step.tool, input_data)
        _update_session(session, result)

        completion_data = {
            "step": i + 1,
            "total_steps": len(task_plan),
            "total_tokens": result.get("input_tokens", 0)
            + result.get("output_tokens", 0),
            "cost": result.get("cost", 0),
            "result": str(result.get("data", ""))[:200],
            "success": result.get("success", False),
        }
        send("step_completed", completion_data)
        sleep(1.5)

    session["status"] = "completed"
    session["completed_at"] = datetime.now()
    send("task_completed", session)


async def execute_agent_task_async(
    agent_engine: Any,
    tool_manager: Any,
    task_id: str,
    session: dict,
    send: AsyncSend,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Asynchronous task execution used by FastAPI."""

    if sleep is None:
        sleep = asyncio.sleep

    task = agent_engine.get_task(task_id)
    if not task:
        await send("error", {"message": "任务不存在"})
        return

    task_plan = task.steps
    for i, step in enumerate(task_plan):
        step_data = {
            "step": i + 1,
            "total_steps": len(task_plan),
            "tool": step.tool,
            "description": step.description,
        }
        await send("step_started", step_data)

        input_data = task.description if i == 0 else session.get("last_output", "")
        result = tool_manager.execute_tool(step.tool, input_data)
        _update_session(session, result)

        completion_data = {
            "step": i + 1,
            "total_steps": len(task_plan),
            "total_tokens": result.get("input_tokens", 0)
            + result.get("output_tokens", 0),
            "cost": result.get("cost", 0),
            "result": str(result.get("data", ""))[:200],
            "success": result.get("success", False),
        }
        await send("step_completed", completion_data)
        await sleep(1.5)

    session["status"] = "completed"
    session["completed_at"] = datetime.now().isoformat()
    await send("task_completed", session)
