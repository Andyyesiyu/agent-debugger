import asyncio
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from agent_engine.core import get_agent_engine
from agent_engine.executor import execute_agent_task_async
from models.schemas import (
    ApiResponse,
    ExecutionRequest,
    TaskRequest,
    TaskResponse,
    TaskStep,
)
from tools.manager import tool_manager

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Agent Debugger API", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局Agent引擎和工具管理器 - 使用工厂函数
agent_engine = get_agent_engine()
agent_engine.set_tool_manager(tool_manager)  # 设置工具管理器
sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}


# == API 路由 ==
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index_fastapi.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/debug", response_class=HTMLResponse)
async def debug_page():
    """Serve the front-end debugger UI."""
    with open("templates/debug_fastapi.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/tasks", response_model=ApiResponse, status_code=201)
async def create_task(task_req: TaskRequest) -> ApiResponse:
    # 使用Agent引擎创建任务
    task_id = agent_engine.create_task(task_req.description, task_req.strategy)

    # 生成执行计划
    plan = agent_engine.plan_execution(task_id)

    try:
        agent_engine.estimate_cost(task_id)
    except ValueError:
        pass  # 忽略估算成本的错误

    # 构建响应数据
    task_steps = [
        TaskStep(
            step_id=step.step_id,
            tool=step.tool,
            description=step.description,
            status="pending",
        )
        for step in plan
    ]

    task_response = TaskResponse(
        id=task_id,
        description=task_req.description,
        strategy=task_req.strategy,
        status="created",
        steps=task_steps,
        total_tokens=0,
        total_cost=0.0,
        created_at=datetime.now(),
    )

    # 保存到sessions
    sessions[task_id] = task_response.model_dump()

    return ApiResponse(
        success=True, message="任务创建成功", data=task_response.model_dump()
    )


@app.get("/api/tasks/{task_id}", response_model=ApiResponse)
async def get_task(task_id: str) -> ApiResponse:
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="任务不存在")

    return ApiResponse(success=True, message="任务详情获取成功", data=sessions[task_id])


@app.get("/api/tools", response_model=ApiResponse)
async def list_tools() -> ApiResponse:
    tools = tool_manager.list_tools()
    return ApiResponse(success=True, message="工具列表获取成功", data={"tools": tools})


@app.get("/api/tools/schemas", response_model=ApiResponse)
async def get_function_schemas() -> ApiResponse:
    """获取所有工具的OpenAI函数模式"""
    schemas = tool_manager.get_function_schemas()
    return ApiResponse(
        success=True, message="函数模式获取成功", data={"schemas": schemas}
    )


@app.post("/api/tasks/{task_id}/execute", response_model=ApiResponse)
async def execute_task(task_id: str, request: ExecutionRequest) -> ApiResponse:
    """执行任务 - 支持静态和LLM驱动两种模式"""
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="任务不存在")

    execution_mode = request.execution_mode
    if execution_mode == "llm_driven":
        if not agent_engine.supports_llm_execution():
            raise HTTPException(status_code=400, detail="LLM执行引擎未配置")
        asyncio.create_task(execute_agent_task_with_llm(task_id))
    else:
        session = sessions[task_id]
        asyncio.create_task(
            execute_agent_task_async(
                agent_engine,
                tool_manager,
                task_id,
                session,
                lambda event, data: broadcast_to_task(
                    task_id, {"type": event, "data": data}
                ),
            )
        )

    return ApiResponse(success=True, message=f"任务执行已启动 ({execution_mode} 模式)")


@app.get("/api/tasks/{task_id}/summary", response_model=ApiResponse)
async def get_task_summary(task_id: str) -> ApiResponse:
    """获取任务执行摘要"""
    task = agent_engine.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 获取任务执行摘要
    summary = task.get_execution_summary()

    return ApiResponse(success=True, message="任务摘要获取成功", data=summary)


# == WebSocket 处理 ==
@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()

    # 添加到连接池
    if task_id not in websocket_connections:
        websocket_connections[task_id] = []
    websocket_connections[task_id].append(websocket)

    try:
        # 发送当前任务状态
        if task_id in sessions:
            await websocket.send_json(
                {"type": "task_status", "data": sessions[task_id]}
            )

        # 等待消息
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "start_execution":
                # 检查执行模式
                execution_mode = data.get("execution_mode", "static")
                if execution_mode == "llm_driven":
                    # 使用LLM驱动的执行
                    asyncio.create_task(execute_agent_task_with_llm(task_id))
                else:
                    session = sessions[task_id]
                    asyncio.create_task(
                        execute_agent_task_async(
                            agent_engine,
                            tool_manager,
                            task_id,
                            session,
                            lambda event, data: broadcast_to_task(
                                task_id, {"type": event, "data": data}
                            ),
                        )
                    )

    except WebSocketDisconnect:
        # 从连接池移除
        if task_id in websocket_connections:
            websocket_connections[task_id].remove(websocket)
            if not websocket_connections[task_id]:
                del websocket_connections[task_id]


async def broadcast_to_task(task_id: str, message: Dict[str, Any]):
    """向任务的所有WebSocket连接广播消息"""
    if task_id in websocket_connections:
        disconnected = []
        for websocket in websocket_connections[task_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # 清理断开的连接
        for ws in disconnected:
            websocket_connections[task_id].remove(ws)


async def execute_agent_task_with_llm(task_id: str):
    """使用LLM函数调用执行Agent任务"""
    if task_id not in sessions:
        await broadcast_to_task(
            task_id, {"type": "error", "data": {"message": "任务不存在"}}
        )
        return

    session = sessions[task_id]
    task = agent_engine.get_task(task_id)

    if not task:
        await broadcast_to_task(
            task_id, {"type": "error", "data": {"message": "任务不存在"}}
        )
        return

    # 检查是否支持LLM执行
    if not agent_engine.supports_llm_execution():
        await broadcast_to_task(
            task_id, {"type": "error", "data": {"message": "LLM执行引擎未配置"}}
        )
        return

    # 更新状态为运行中
    session["status"] = "running"
    session["execution_mode"] = "llm_driven"
    await broadcast_to_task(task_id, {"type": "task_started", "data": session})

    # 定义进度回调函数
    async def progress_callback(event_data):
        event_type = event_data.get("type")

        if event_type == "reasoning_step":
            # LLM推理步骤事件
            reasoning_data = {
                "step": event_data["step"],
                "reasoning": event_data["reasoning"],
                "function_calls": event_data["function_calls"],
                "tokens": event_data["tokens"],
                "cost": event_data["cost"],
            }
            await broadcast_to_task(
                task_id, {"type": "llm_reasoning", "data": reasoning_data}
            )

        elif event_type == "function_call":
            # 函数调用事件
            function_data = {
                "step": event_data["step"],
                "function_name": event_data["function_name"],
                "arguments": event_data["arguments"],
                "result": event_data["result"],
            }
            await broadcast_to_task(
                task_id, {"type": "function_call", "data": function_data}
            )

    try:
        # 使用LLM执行任务
        result = await agent_engine.execute_with_llm(task_id, progress_callback)

        # 更新会话状态
        session["total_tokens"] = result.get("total_tokens", 0)
        session["total_cost"] = result.get("total_cost", 0.0)
        session["reasoning_steps"] = result.get("reasoning_steps", 0)
        session["success"] = result.get("success", False)

        if result["success"]:
            session["status"] = "completed"
            session["result"] = result.get("result", "")
        else:
            session["status"] = "failed"
            session["error"] = result.get("error", "Unknown error")

        session["completed_at"] = datetime.now().isoformat()

        # 发送任务完成事件
        await broadcast_to_task(task_id, {"type": "task_completed", "data": session})

    except Exception as e:
        session["status"] = "failed"
        session["error"] = str(e)
        session["completed_at"] = datetime.now().isoformat()
        await broadcast_to_task(task_id, {"type": "error", "data": {"message": str(e)}})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
