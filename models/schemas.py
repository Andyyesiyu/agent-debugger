from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TaskRequest(BaseModel):
    """Request model for creating a task."""

    description: str
    strategy: str = "balanced"
    execution_mode: str = "static"


class ExecutionRequest(BaseModel):
    """Request model for executing a task."""

    execution_mode: str = "static"


class TaskStep(BaseModel):
    """Step information in a task response."""

    step_id: str
    tool: str
    description: str
    status: str = "pending"


class TaskResponse(BaseModel):
    """Response model for task information."""

    id: str
    description: str
    strategy: str
    status: str
    steps: List[TaskStep]
    total_tokens: int = 0
    total_cost: float = 0.0
    created_at: datetime


class CostEstimate(BaseModel):
    """Estimated cost for executing a task."""

    total_tokens: int
    total_cost: float
    steps: int


class ApiResponse(BaseModel):
    """Generic API response wrapper."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

