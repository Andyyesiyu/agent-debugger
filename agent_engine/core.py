import uuid
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import openai
import os


@dataclass
class TaskStep:
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
    id: str
    description: str
    strategy: str = "balanced"  # "speed", "cost", "accuracy", "balanced"
    status: str = "created"
    steps: List[TaskStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0
    output: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "task_id": self.id,
            "description": self.description,
            "strategy": self.strategy,
            "status": self.status,
            "total_steps": len(self.steps),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "duration": (self.completed_at - self.started_at).total_seconds()
            if self.completed_at and self.started_at
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


class AgentDecisionEngine:
    """Agent决策引擎 - 核心逻辑"""

    strategies = {
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

    def __init__(self):
        self.tasks = {}
        self.llm_engine = LLMFunctionCallingEngine()
        self.tool_manager = None  # 将在设置时初始化

    def set_tool_manager(self, tool_manager):
        """设置工具管理器"""
        self.tool_manager = tool_manager
        self.llm_engine.set_tool_manager(tool_manager)

    def create_task(self, description: str, strategy: str = "balanced") -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        task = Task(id=task_id, description=description, strategy=strategy)
        self.tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)

    def plan_execution(self, task_id: str) -> List[TaskStep]:
        """根据任务描述生成执行计划"""
        task = self.get_task(task_id)
        if not task:
            raise ValueError("Task not found")

        # 简单规划逻辑 - 基于关键词识别工具使用
        description = task.description.lower()
        plan = []
        step_counter = 1

        # 搜索工具 - 判断是否涉及信息收集
        if any(
            keyword in description
            for keyword in ["搜索", "信息", "数据", "趋势", "研究"]
        ):
            plan.append(
                TaskStep(
                    step_id=f"step_{step_counter}",
                    tool="web_search",
                    description=f"搜索与'{task.description[:50]}...'相关的信息",
                )
            )
            step_counter += 1

        # 分析工具 - 判断是否涉及内容分析
        if (
            any(keyword in description for keyword in ["分析", "总结", "提取", "理解"])
            or len(plan) > 0
        ):
            plan.append(
                TaskStep(
                    step_id=f"step_{step_counter}",
                    tool="text_analysis",
                    description="分析收集的数据并提取关键点",
                )
            )
            step_counter += 1

        # 生成工具 - 判断是否涉及内容生成
        plan.append(
            TaskStep(
                step_id=f"step_{step_counter}",
                tool="content_generation",
                description="生成最终回应或报告",
            )
        )

        task.steps = plan
        return plan

    def estimate_cost(self, task_id: str) -> Dict[str, Any]:
        """估算任务成本"""
        task = self.get_task(task_id)
        if not task:
            raise ValueError("Task not found")

        total_tokens = 0
        for step in task.steps:
            # 简单的成本估算模型
            if step.tool == "web_search":
                length = len(task.description)
                tokens = min(5000, max(500, length * 2))
                length = len(task.description)
                tokens = min(5000, max(500, length * 2))
                step.input_tokens = int(WEB_SEARCH_INPUT_TOKENS)
                step.output_tokens = int(tokens)
                step.cost = float(tokens * 0.000002)  # ~$2 per 1M tokens

            elif step.tool == "text_analysis":
                step.input_tokens = 2000
                step.output_tokens = 800
                step.cost = float(2800 * 0.000002)

            elif step.tool == "content_generation":
                step.input_tokens = 1500
                step.output_tokens = 1000
                step.cost = float(2500 * 0.000002)

            total_tokens += step.input_tokens + step.output_tokens

        total_cost = 0
        for step in task.steps:
            total_cost += step.cost

        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "steps": len(task.steps),
        }

    def get_execution_strategy(self, task_id: str) -> Dict[str, Any]:
        """获取执行策略配置"""
        task = self.get_task(task_id)
        if not task:
            return self.strategies["balanced"]
        return self.strategies.get(task.strategy, self.strategies["balanced"])

    async def execute_with_llm(
        self, task_id: str, progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """使用LLM函数调用执行任务"""
        task = self.get_task(task_id)
        if not task:
            return {"success": False, "error": "Task not found"}

        if not self.tool_manager:
            return {"success": False, "error": "Tool manager not set"}

        # 清空之前的推理步骤
        self.llm_engine.reasoning_steps = []

        try:
            # 使用LLM进行推理和执行
            result = await self.llm_engine.reason_and_execute(
                task_description=task.description,
                strategy=task.strategy,
                max_iterations=self.strategies[task.strategy].get("max_concurrent", 5),
                progress_callback=progress_callback,
            )

            # 更新任务状态
            task.status = "completed" if result["success"] else "failed"
            task.total_tokens = result.get("total_tokens", 0)
            task.total_cost = result.get("total_cost", 0.0)

            # 将LLM推理步骤转换为任务步骤
            for reasoning_step in self.llm_engine.reasoning_steps:
                for j, function_call in enumerate(reasoning_step.function_calls):
                    step = TaskStep(
                        step_id=f"{reasoning_step.step_id}_{j}",
                        tool=function_call.function_name,
                        description=f"LLM reasoning: {reasoning_step.reasoning[:100]}...",
                        input_data=function_call.arguments,
                        output_data=reasoning_step.results[j]
                        if j < len(reasoning_step.results)
                        else None,
                        status="completed",
                        input_tokens=reasoning_step.input_tokens
                        // len(reasoning_step.function_calls)
                        if reasoning_step.function_calls
                        else reasoning_step.input_tokens,
                        output_tokens=reasoning_step.output_tokens
                        // len(reasoning_step.function_calls)
                        if reasoning_step.function_calls
                        else reasoning_step.output_tokens,
                        cost=reasoning_step.cost / len(reasoning_step.function_calls)
                        if reasoning_step.function_calls
                        else reasoning_step.cost,
                        started_at=reasoning_step.timestamp,
                        completed_at=reasoning_step.timestamp,
                    )
                    task.steps.append(step)

            return result

        except Exception as e:
            task.status = "failed"
            return {"success": False, "error": str(e)}

    def supports_llm_execution(self) -> bool:
        """检查是否支持LLM执行"""
        return self.tool_manager is not None and self.llm_engine is not None


# 配置常量
WEB_SEARCH_INPUT_TOKENS = 1000
WEB_SEARCH_OUTPUT_TOKENS = 2000
ANALYSIS_INPUT_TOKENS = 1500
ANALYSIS_OUTPUT_TOKENS = 800


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
    """LLM函数调用引擎 - 使用OpenAI的函数调用能力"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.tool_manager = None  # 将在初始化时设置
        self.reasoning_steps: List[LLMReasoningStep] = []

    def set_tool_manager(self, tool_manager):
        """设置工具管理器"""
        self.tool_manager = tool_manager

    def get_available_functions(self) -> List[Dict[str, Any]]:
        """获取可用函数的OpenAI函数模式"""
        if not self.tool_manager:
            return []
        return self.tool_manager.get_function_schemas()

    async def execute_function_call(
        self, function_call: FunctionCall
    ) -> Dict[str, Any]:
        """执行函数调用"""
        if not self.tool_manager:
            return {"success": False, "error": "Tool manager not set"}

        tool = self.tool_manager.get_tool(function_call.function_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool {function_call.function_name} not found",
            }

        try:
            # 执行工具
            input_data = function_call.arguments.get("input_data", "")
            result = tool.execute(input_data)

            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "execution_time": result.execution_time,
                "cost": (result.input_tokens + result.output_tokens)
                * tool.cost_per_token,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def reason_and_execute(
        self,
        task_description: str,
        strategy: str = "balanced",
        max_iterations: int = 5,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """使用LLM推理并执行函数调用"""

        system_prompt = self._get_system_prompt(strategy)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task_description}"},
        ]

        iteration = 0
        final_result = None

        while iteration < max_iterations:
            try:
                # 调用LLM进行推理
                response = await self.client.chat.completions.create(
                    model=self._get_model_for_strategy(strategy),
                    messages=messages,  # type: ignore
                    tools=self.get_available_functions(),  # type: ignore
                    tool_choice="auto",
                    temperature=0.1,
                )

                message = response.choices[0].message
                reasoning_step = LLMReasoningStep(
                    step_id=str(uuid.uuid4())[:8],
                    reasoning=message.content or "",
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens
                    if response.usage
                    else 0,
                    cost=self._calculate_cost(response.usage, strategy)
                    if response.usage
                    else 0.0,
                )

                # 添加LLM响应到对话历史
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.tool_calls:
                    assistant_message["tool_calls"] = message.tool_calls
                messages.append(assistant_message)

                # 如果有函数调用，执行它们
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_call = FunctionCall(
                            function_name=tool_call.function.name,
                            arguments=json.loads(tool_call.function.arguments),
                            call_id=tool_call.id,
                        )
                        reasoning_step.function_calls.append(function_call)

                        # 执行函数调用
                        result = await self.execute_function_call(function_call)
                        reasoning_step.results.append(result)

                        # 添加函数结果到对话历史
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result),
                            }
                        )

                        # 发送进度更新
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

                self.reasoning_steps.append(reasoning_step)

                # 发送推理步骤更新
                if progress_callback:
                    await progress_callback(
                        {
                            "type": "reasoning_step",
                            "step": iteration + 1,
                            "reasoning": reasoning_step.reasoning,
                            "function_calls": len(reasoning_step.function_calls),
                            "tokens": reasoning_step.input_tokens
                            + reasoning_step.output_tokens,
                            "cost": reasoning_step.cost,
                        }
                    )

                # 如果没有函数调用，说明任务完成
                if not message.tool_calls:
                    final_result = {
                        "success": True,
                        "result": message.content,
                        "reasoning_steps": len(self.reasoning_steps),
                        "total_tokens": sum(
                            step.input_tokens + step.output_tokens
                            for step in self.reasoning_steps
                        ),
                        "total_cost": sum(step.cost for step in self.reasoning_steps),
                    }
                    break

                iteration += 1

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "reasoning_steps": len(self.reasoning_steps),
                }

        if final_result is None:
            final_result = {
                "success": False,
                "error": "Maximum iterations reached without completion",
                "reasoning_steps": len(self.reasoning_steps),
                "total_tokens": sum(
                    step.input_tokens + step.output_tokens
                    for step in self.reasoning_steps
                ),
                "total_cost": sum(step.cost for step in self.reasoning_steps),
            }

        return final_result

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

    def _calculate_cost(self, usage, strategy: str) -> float:
        """计算API调用成本"""
        model = self._get_model_for_strategy(strategy)

        # 简化的成本计算 (实际应该根据具体模型定价)
        cost_per_1k_tokens = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "gpt-4o-mini": 0.0015,
        }

        rate = cost_per_1k_tokens.get(model, 0.002)
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        return (total_tokens / 1000) * rate


GENERATION_INPUT_TOKENS = 1200
GENERATION_OUTPUT_TOKENS = 1000
