"""Unit tests for the LLM integration layer."""

from pathlib import Path
import asyncio
import sys
from typing import Dict, List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent_engine.core import AgentDecisionEngine
from tools.manager import tool_manager


@pytest.fixture()
def decision_engine() -> AgentDecisionEngine:
    """Provide an agent engine configured with the default tool manager."""

    engine = AgentDecisionEngine()
    engine.set_tool_manager(tool_manager)
    return engine


def test_tool_function_schemas_are_well_formed() -> None:
    """Ensure each tool exposes a schema that contains required metadata."""

    schemas: List[Dict[str, Dict[str, object]]] = tool_manager.get_function_schemas()

    assert schemas, "Expected at least one tool schema to be registered"
    for schema in schemas:
        function = schema.get("function", {})
        assert "name" in function and function["name"], "Function schema missing name"
        assert "description" in function and function["description"], "Function schema missing description"
        parameters = function.get("parameters", {})
        assert parameters.get("type") == "object", "Function parameters must be an object"
        assert isinstance(parameters.get("properties", {}), dict)


def test_execute_with_llm_gracefully_handles_missing_openai(
    decision_engine: AgentDecisionEngine,
) -> None:
    """When the OpenAI client is unavailable the engine should fail with a helpful error."""

    task_id = decision_engine.create_task(
        "Collect and summarise new research on reinforcement learning"
    )
    decision_engine.plan_execution(task_id)

    result = asyncio.run(decision_engine.execute_with_llm(task_id))

    assert result["success"] is False
    assert "OpenAI client" in result["error"]
    task = decision_engine.get_task(task_id)
    assert task is not None
    assert task.status == "failed"
