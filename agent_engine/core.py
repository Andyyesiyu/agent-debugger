"""Core decision and execution logic for the Agent Debugger application."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Cost and token estimation helpers
# ---------------------------------------------------------------------------

DEFAULT_COST_PER_TOKEN = 0.000002
WEB_SEARCH_INPUT_TOKENS = 1000
WEB_SEARCH_MIN_OUTPUT_TOKENS = 500
WEB_SEARCH_MAX_OUTPUT_TOKENS = 5000
ANALYSIS_INPUT_TOKENS = 1500
ANALYSIS_OUTPUT_TOKENS = 800
GENERATION_INPUT_TOKENS = 1200
GENERATION_OUTPUT_TOKENS = 1000


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    """Return the ISO formatted representation of ``value`` when available."""

    return value.isoformat() if value else None


@dataclass
class TaskStep:
    """Represents a single unit of work executed by the agent."""

    step_id: str
    tool: str
    description: str
    input_data: Any = None
    output_data: Any = None
    status: str = "pending"
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class Task:
    """Container representing an agent task and its lifecycle information."""

    id: str
    description: str
    strategy: str = "balanced"  # "speed", "cost", "accuracy", "balanced"
    status: str = "created"
    steps: List[TaskStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    output: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the task's execution state."""

        duration = (
            (self.completed_at - self.started_at).total_seconds()
            if self.completed_at and self.started_at
            else None
        )

        return {
            "task_id": self.id,
            "description": self.description,
            "strategy": self.strategy,
            "status": self.status,
            "total_steps": len(self.steps),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "duration": duration,
            "created_at": _isoformat(self.created_at),
            "started_at": _isoformat(self.started_at),
            "completed_at": _isoformat(self.completed_at),
        }


class AgentDecisionEngine:
    """High level orchestration layer for agent task planning and execution."""

    strategies: Dict[str, Dict[str, Any]] = {
        "speed": {
            "max_concurrent": 3,
            "timeout": 10,
            "cache_enabled": True,
            "allow_cheaper_models": True,
        },
        "cost": {
            "max_concurrent": 1,
            "timeout": 60,
            "cache_enabled": True,
            "prefer_cache": True,
        },
        "accuracy": {
            "max_concurrent": 1,
            "timeout": 300,
            "cache_enabled": False,
            "prefer_quality_models": True,
        },
        "balanced": {
            "max_concurrent": 2,
            "timeout": 30,
            "cache_enabled": True,
            "balance_quality_cost": True,
        },
    }

    def __init__(self, llm_engine: Optional["LLMFunctionCallingEngine"] = None) -> None:
        self.tasks: Dict[str, Task] = {}
        self.llm_engine = llm_engine or LLMFunctionCallingEngine()
        self.tool_manager = None

    # ------------------------------------------------------------------
    # Task lifecycle helpers
    # ------------------------------------------------------------------
    def set_tool_manager(self, tool_manager: Any) -> None:
        """Attach a tool manager to both the decision and LLM engines."""

        self.tool_manager = tool_manager
        self.llm_engine.set_tool_manager(tool_manager)

    def create_task(self, description: str, strategy: str = "balanced") -> str:
        """Create a new task and return its identifier."""

        task_id = str(uuid.uuid4())
        self.tasks[task_id] = Task(id=task_id, description=description, strategy=strategy)
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Return a task when present."""

        return self.tasks.get(task_id)

    def plan_execution(self, task_id: str) -> List[TaskStep]:
        """Generate a naive execution plan based on the task description."""

        task = self._require_task(task_id)
        description = task.description.lower()
        plan: List[TaskStep] = []

        def add_step(tool: str, description_text: str) -> None:
            step_number = len(plan) + 1
            plan.append(
                TaskStep(
                    step_id=f"step_{step_number}",
                    tool=tool,
                    description=description_text,
                )
            )

        if any(keyword in description for keyword in ["搜索", "信息", "数据", "趋势", "研究"]):
            add_step(
                "web_search",
                f"搜索与'{task.description[:50]}...'相关的信息",
            )

        if plan or any(
            keyword in description for keyword in ["分析", "总结", "提取", "理解"]
        ):
            add_step("text_analysis", "分析收集的数据并提取关键点")

        add_step("content_generation", "生成最终回应或报告")

        task.steps = plan
        return plan

    def estimate_cost(self, task_id: str) -> Dict[str, Any]:
        """Estimate the overall cost footprint for a planned task."""

        task = self._require_task(task_id)

        total_tokens = 0
        total_cost = 0.0
        for step in task.steps:
            input_tokens, output_tokens = self._estimate_tokens_for_step(
                step.tool, task.description
            )
            step.input_tokens = input_tokens
            step.output_tokens = output_tokens
            step.cost = self._calculate_token_cost(input_tokens, output_tokens)
            total_tokens += input_tokens + output_tokens
            total_cost += step.cost

        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "steps": len(task.steps),
        }

    def get_execution_strategy(self, task_id: str) -> Dict[str, Any]:
        """Return the strategy configuration for the requested task."""

        task = self.get_task(task_id)
        if not task:
            return self.strategies["balanced"]
        return self.strategies.get(task.strategy, self.strategies["balanced"])

    async def execute_with_llm(
        self, task_id: str, progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """Run the task using the LLM function calling engine."""

        task = self._require_task(task_id)
        if not self.tool_manager:
            return {"success": False, "error": "Tool manager not set"}

        strategy_config = self.get_execution_strategy(task_id)
        max_iterations = strategy_config.get("max_concurrent", 5)

        self.llm_engine.reasoning_steps.clear()

        try:
            result = await self.llm_engine.reason_and_execute(
                task_description=task.description,
                strategy=task.strategy,
                max_iterations=max_iterations,
                progress_callback=progress_callback,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            task.status = "failed"
            return {"success": False, "error": str(exc)}

        task.status = "completed" if result.get("success") else "failed"
        task.total_tokens = result.get("total_tokens", 0)
        task.total_cost = result.get("total_cost", 0.0)

        for reasoning_step in self.llm_engine.reasoning_steps:
            for index, function_call in enumerate(reasoning_step.function_calls):
                output_data = (
                    reasoning_step.results[index]
                    if index < len(reasoning_step.results)
                    else None
                )
                task.steps.append(
                    TaskStep(
                        step_id=f"{reasoning_step.step_id}_{index}",
                        tool=function_call.function_name,
                        description=f"LLM reasoning: {reasoning_step.reasoning[:100]}...",
                        input_data=function_call.arguments,
                        output_data=output_data,
                        status="completed",
                        input_tokens=self._split_tokens(
                            reasoning_step.input_tokens, len(reasoning_step.function_calls)
                        ),
                        output_tokens=self._split_tokens(
                            reasoning_step.output_tokens, len(reasoning_step.function_calls)
                        ),
                        cost=self._split_value(
                            reasoning_step.cost, len(reasoning_step.function_calls)
                        ),
                        started_at=reasoning_step.timestamp,
                        completed_at=reasoning_step.timestamp,
                    )
                )

        return result

    def supports_llm_execution(self) -> bool:
        """Return ``True`` when an LLM engine and tool manager are configured."""

        return self.tool_manager is not None and self.llm_engine is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_task(self, task_id: str) -> Task:
        task = self.get_task(task_id)
        if not task:
            raise ValueError("Task not found")
        return task

    @staticmethod
    def _calculate_token_cost(input_tokens: int, output_tokens: int) -> float:
        return float((input_tokens + output_tokens) * DEFAULT_COST_PER_TOKEN)

    @staticmethod
    def _split_value(value: float, parts: int) -> float:
        if parts <= 1:
            return value
        return value / parts

    @staticmethod
    def _split_tokens(value: int, parts: int) -> int:
        if parts <= 1:
            return value
        return max(0, value // parts)

    def _estimate_tokens_for_step(self, tool: str, description: str) -> tuple[int, int]:
        if tool == "web_search":
            length = len(description)
            dynamic_output = min(
                WEB_SEARCH_MAX_OUTPUT_TOKENS,
                max(WEB_SEARCH_MIN_OUTPUT_TOKENS, length * 2),
            )
            return WEB_SEARCH_INPUT_TOKENS, dynamic_output
        if tool == "text_analysis":
            return ANALYSIS_INPUT_TOKENS, ANALYSIS_OUTPUT_TOKENS
        if tool == "content_generation":
            return GENERATION_INPUT_TOKENS, GENERATION_OUTPUT_TOKENS
        return 0, 0


@dataclass
class FunctionCall:
    """表示一个函数调用"""

    function_name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class LLMReasoningStep:
    """表示LLM推理步骤"""

    step_id: str
    reasoning: str
    function_calls: List[FunctionCall] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class LLMFunctionCallingEngine:
    """LLM function calling engine backed by the OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        if client is not None:
            self.client = client
        elif AsyncOpenAI is not None:
            self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        else:  # pragma: no cover - depends on optional dependency
            self.client = None

        self.tool_manager = None
        self.reasoning_steps: List[LLMReasoningStep] = []

    def set_tool_manager(self, tool_manager: Any) -> None:
        """Attach the shared tool manager."""

        self.tool_manager = tool_manager

    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Return OpenAI tool schemas for all registered tools."""

        if not self.tool_manager:
            return []
        return self.tool_manager.get_function_schemas()

    async def execute_function_call(self, function_call: FunctionCall) -> Dict[str, Any]:
        """Execute a tool call originating from the LLM."""

        if not self.tool_manager:
            return {"success": False, "error": "Tool manager not set"}

        tool = self.tool_manager.get_tool(function_call.function_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool {function_call.function_name} not found",
            }

        try:
            input_data = function_call.arguments.get("input_data", "")
            result = tool.execute(input_data)
        except Exception as exc:  # pragma: no cover - depends on tool implementation
            return {"success": False, "error": str(exc)}

        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "execution_time": result.execution_time,
            "cost": (result.input_tokens + result.output_tokens) * tool.cost_per_token,
        }

    async def reason_and_execute(
        self,
        task_description: str,
        strategy: str = "balanced",
        max_iterations: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Drive the reasoning loop using the OpenAI function calling API."""

        if self.client is None:
            return {
                "success": False,
                "error": "OpenAI client is not available. Install the 'openai' package to enable LLM execution.",
                "reasoning_steps": len(self.reasoning_steps),
            }

        self.reasoning_steps.clear()

        system_prompt = self._get_system_prompt(strategy)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task_description}"},
        ]

        final_result: Optional[Dict[str, Any]] = None

        for iteration in range(max_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self._get_model_for_strategy(strategy),
                    messages=messages,  # type: ignore[arg-type]
                    tools=self.get_available_functions(),  # type: ignore[arg-type]
                    tool_choice="auto",
                    temperature=0.1,
                )
            except Exception as exc:  # pragma: no cover - network/runtime errors
                return {
                    "success": False,
                    "error": str(exc),
                    "reasoning_steps": len(self.reasoning_steps),
                }

            message = response.choices[0].message
            reasoning_step = self._build_reasoning_step(message, response, strategy)
            messages.append(self._build_assistant_message(message))

            if message.tool_calls:
                await self._handle_tool_calls(
                    message.tool_calls,
                    reasoning_step,
                    messages,
                    iteration,
                    progress_callback,
                )

            self.reasoning_steps.append(reasoning_step)

            if progress_callback:
                await progress_callback(
                    {
                        "type": "reasoning_step",
                        "step": iteration + 1,
                        "reasoning": reasoning_step.reasoning,
                        "function_calls": len(reasoning_step.function_calls),
                        "tokens": reasoning_step.input_tokens + reasoning_step.output_tokens,
                        "cost": reasoning_step.cost,
                    }
                )

            if not message.tool_calls:
                final_result = self._build_success_result(message.content)
                break

        return final_result or self._build_timeout_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_reasoning_step(self, message: Any, response: Any, strategy: str) -> LLMReasoningStep:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return LLMReasoningStep(
            step_id=str(uuid.uuid4())[:8],
            reasoning=message.content or "",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=self._calculate_cost(usage, strategy) if usage else 0.0,
        )

    @staticmethod
    def _build_assistant_message(message: Any) -> Dict[str, Any]:
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if getattr(message, "tool_calls", None):
            assistant_message["tool_calls"] = message.tool_calls
        return assistant_message

    async def _handle_tool_calls(
        self,
        tool_calls: Any,
        reasoning_step: LLMReasoningStep,
        messages: List[Dict[str, Any]],
        iteration: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
    ) -> None:
        for tool_call in tool_calls:
            function_call = FunctionCall(
                function_name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
                call_id=tool_call.id,
            )
            reasoning_step.function_calls.append(function_call)

            result = await self.execute_function_call(function_call)
            reasoning_step.results.append(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

            if progress_callback:
                await progress_callback(
                    {
                        "type": "function_call",
                        "step": iteration + 1,
                        "function_name": function_call.function_name,
                        "arguments": function_call.arguments,
                        "result": result,
                    }
                )

    def _build_success_result(self, final_message: Optional[str]) -> Dict[str, Any]:
        return {
            "success": True,
            "result": final_message,
            "reasoning_steps": len(self.reasoning_steps),
            "total_tokens": self._total_tokens(),
            "total_cost": self._total_cost(),
        }

    def _build_timeout_result(self) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Maximum iterations reached without completion",
            "reasoning_steps": len(self.reasoning_steps),
            "total_tokens": self._total_tokens(),
            "total_cost": self._total_cost(),
        }

    def _total_tokens(self) -> int:
        return sum(step.input_tokens + step.output_tokens for step in self.reasoning_steps)

    def _total_cost(self) -> float:
        return sum(step.cost for step in self.reasoning_steps)

    def _get_system_prompt(self, strategy: str) -> str:
        """根据策略获取系统提示"""
        base_prompt = """You are an intelligent agent that can use various tools to complete tasks.

Available tools:
- web_search: Search the web for information
- text_analysis: Analyze text content for insights
- content_generation: Generate content based on input

Your goal is to complete the given task efficiently by:
1. Understanding the task requirements
2. Selecting appropriate tools to gather information or perform actions
3. Chaining tool calls when necessary to build upon previous results
4. Providing a comprehensive final answer

Always explain your reasoning before making tool calls."""

        strategy_additions = {
            "speed": " Focus on completing the task quickly with minimal tool calls.",
            "cost": " Minimize the number of tool calls and prefer simpler approaches to reduce costs.",
            "accuracy": " Use multiple tools and cross-reference information to ensure accuracy.",
            "balanced": " Balance speed, cost, and accuracy in your approach.",
        }

        addition = strategy_additions.get(strategy, strategy_additions["balanced"])
        return base_prompt + (addition or "")

    def _get_model_for_strategy(self, strategy: str) -> str:
        """根据策略选择模型"""
        model_mapping = {
            "speed": "gpt-3.5-turbo",
            "cost": "gpt-3.5-turbo",
            "accuracy": "gpt-4",
            "balanced": "gpt-4o-mini",
        }
        return model_mapping.get(strategy, "gpt-4o-mini")

    def _calculate_cost(self, usage: Any, strategy: str) -> float:
        """Estimate API call cost using simplified pricing tables."""

        if not usage:
            return 0.0

        model = self._get_model_for_strategy(strategy)

        cost_per_1k_tokens = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "gpt-4o-mini": 0.0015,
        }

        rate = cost_per_1k_tokens.get(model, 0.002)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        return (total_tokens / 1000) * rate
