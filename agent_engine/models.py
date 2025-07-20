from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List, Optional, Any, Dict
from enum import Enum
from uuid import uuid4


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStrategy(str, Enum):
    SPEED = "speed"
    COST = "cost"
    ACCURACY = "accuracy"
    BALANCED = "balanced"


class ExecutionMode(str, Enum):
    STATIC = "static"  # 传统的静态规划执行
    LLM_DRIVEN = "llm_driven"  # LLM驱动的动态执行


class FunctionCallModel(BaseModel):
    """函数调用模型"""

    function_name: str = Field(...)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    result: Optional[Dict[str, Any]] = None
    execution_time: float = Field(default=0.0, ge=0.0)
    success: bool = True
    error_message: Optional[str] = None


class LLMReasoningStepModel(BaseModel):
    """LLM推理步骤模型"""

    step_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    reasoning: str = Field(...)
    function_calls: List[FunctionCallModel] = Field(default_factory=list)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class TaskStepModel(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    tool: str = Field(...)
    description: str = Field(...)
    input_data: Any = None
    output_data: Any = None
    status: StepStatus = StepStatus.PENDING
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Function calling metadata
    function_call: Optional[FunctionCallModel] = None
    llm_reasoning: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.STATIC

    def mark_running(self):
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, output_data=None, output_tokens=0, cost=0.0):
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now()
        self.output_data = output_data
        self.output_tokens = output_tokens
        self.cost = cost

    def mark_failed(self, error_message: str):
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message


class TaskModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(..., min_length=1)
    strategy: TaskStrategy = TaskStrategy.BALANCED
    status: TaskStatus = TaskStatus.PENDING
    steps: List[TaskStepModel] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    output: Any = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # LLM execution metadata
    execution_mode: ExecutionMode = ExecutionMode.STATIC
    llm_reasoning_steps: List[LLMReasoningStepModel] = Field(default_factory=list)
    model_used: Optional[str] = None
    max_iterations: int = Field(default=5, ge=1, le=20)

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}

    def add_step(
        self, tool: str, description: str, input_data: Any = None
    ) -> TaskStepModel:
        step = TaskStepModel(tool=tool, description=description, input_data=input_data)
        self.steps.append(step)
        return step

    def start_execution(self):
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def complete_execution(self, output: Any = None):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.output = output

        # 计算总计
        self.total_tokens = sum(
            step.input_tokens + step.output_tokens for step in self.steps
        )
        self.total_cost = sum(step.cost for step in self.steps)

    def fail_execution(self, error_message: str):
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.steps[-1].mark_failed(error_message) if self.steps else None

    def add_llm_reasoning_step(self, reasoning_step: LLMReasoningStepModel):
        """添加LLM推理步骤"""
        self.llm_reasoning_steps.append(reasoning_step)
        self.total_tokens += reasoning_step.input_tokens + reasoning_step.output_tokens
        self.total_cost += reasoning_step.cost

    def set_llm_execution_mode(self, model_used: str, max_iterations: int = 5):
        """设置LLM执行模式"""
        self.execution_mode = ExecutionMode.LLM_DRIVEN
        self.model_used = model_used
        self.max_iterations = max_iterations

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "task_id": self.id,
            "description": self.description,
            "strategy": self.strategy,
            "execution_mode": self.execution_mode,
            "status": self.status,
            "total_steps": len(self.steps),
            "llm_reasoning_steps": len(self.llm_reasoning_steps),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model_used": self.model_used,
            "duration": (self.completed_at - self.started_at).total_seconds()
            if self.completed_at and self.started_at
            else None,
        }


class ExecutionStrategy(BaseModel):
    max_concurrent: int = 1
    timeout: int = 30
    cache_enabled: bool = True
    prefer_cache: bool = False
    budget_limit: Optional[float] = None

    class Config:
        extra = "allow"


class CostEstimate(BaseModel):
    total_tokens: int = Field(..., ge=0)
    total_cost: float = Field(..., ge=0.0)
    steps: int = Field(..., ge=0)
    strategy_config: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self):
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "steps": self.steps,
            **self.strategy_config,
        }
