from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TaskRequest(BaseModel):
    """Create task request."""

    description: str
    strategy: str = "balanced"
    execution_mode: str = "static"


class ExecutionRequest(BaseModel):
    """Trigger task execution."""

    execution_mode: str = "static"


class TaskStep(BaseModel):
    step_id: str
    tool: str
    description: str
    status: str = "pending"


class TaskResponse(BaseModel):
    id: str
    description: str
    strategy: str
    status: str
    steps: List[TaskStep]
    total_tokens: int = 0
    total_cost: float = 0.0
    created_at: datetime


class CostEstimate(BaseModel):
    total_tokens: int
    total_cost: float
    steps: int


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
