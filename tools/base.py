from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    execution_time: float = 0.0


class BaseTool(ABC):
    """工具基础接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def cost_per_token(self) -> float:
        """每token的平均成本(美元)"""
        return 0.000002  # $2 per 1M tokens

    @abstractmethod
    def execute(self, input_data: Any) -> ToolResult:
        """执行工具的主要逻辑"""
        pass

    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据的合法性"""
        return True

    def estimate_cost(self, input_data: Any) -> Dict[str, float]:
        """预估执行成本"""
        return {
            "input_tokens": 1000,
            "output_tokens": 800,
            "estimated_cost": 1000 * self.cost_per_token,
        }

    def get_function_schema(self) -> Dict[str, Any]:
        """获取OpenAI函数调用模式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数模式 - 子类可以重写以提供更详细的参数定义"""
        return {
            "type": "object",
            "properties": {
                "input_data": {
                    "type": "string",
                    "description": "Input data for the tool",
                }
            },
            "required": ["input_data"],
        }

    @property
    def supports_streaming(self) -> bool:
        """是否支持流式处理"""
        return False

    @property
    def requires_api_key(self) -> bool:
        """是否需要API密钥"""
        return False

    def get_usage_examples(self) -> List[Dict[str, str]]:
        """获取使用示例"""
        return [
            {
                "description": f"Basic usage of {self.name}",
                "input": "sample input",
                "expected_output": "sample output",
            }
        ]
