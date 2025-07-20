from typing import Dict, List, Any
from .web_search import WebSearchTool, TextAnalysisTool, ContentGenerationTool
from .base import BaseTool


class ToolManager:
    """工具管理器 - 负责工具注册和调度"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        tools = [WebSearchTool(), TextAnalysisTool(), ContentGenerationTool()]

        for tool in tools:
            self.tools[tool.name] = tool

    def get_tool(self, tool_name: str) -> BaseTool:
        """获取指定工具"""
        return self.tools.get(tool_name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "estimated_cost_per_use": tool.estimate_cost("test data")[
                    "estimated_cost"
                ],
                "supports_streaming": tool.supports_streaming,
                "requires_api_key": tool.requires_api_key,
                "usage_examples": tool.get_usage_examples(),
            }
            for tool in self.tools.values()
        ]

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的OpenAI函数模式"""
        return [tool.get_function_schema() for tool in self.tools.values()]

    def get_tool_by_function_name(self, function_name: str) -> BaseTool:
        """根据函数名获取工具"""
        return self.tools.get(function_name)

    def execute_tool(self, tool_name: str, input_data: Any) -> Dict[str, Any]:
        """执行工具并返回结构化结果"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool {tool_name} not found"}

        try:
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


# 全局工具管理器实例
tool_manager = ToolManager()
