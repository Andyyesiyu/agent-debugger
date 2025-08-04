import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit, join_room

from agent_engine.core import AgentDecisionEngine
from tools.manager import tool_manager
from models.schemas import ApiResponse, TaskRequest, TaskResponse, TaskStep

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局Agent引擎
agent_engine = AgentDecisionEngine()
sessions: Dict[str, Dict[str, Any]] = {}


# == API 路由 ==
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tasks", methods=["POST"])
def create_task():
    try:
        data = request.json or {}
        task_req = TaskRequest(**data)  # type: ignore
    except Exception as e:
        return jsonify(ApiResponse(success=False, message=str(e)).model_dump()), 400

    # 使用Agent引擎创建任务
    task_id = agent_engine.create_task(task_req.description, task_req.strategy)

    # 生成执行计划
    plan = agent_engine.plan_execution(task_id)
    agent_engine.estimate_cost(task_id)

    task_steps = [
        TaskStep(step_id=step.step_id, tool=step.tool, description=step.description)
        for step in plan
    ]
    task_response = TaskResponse(
        id=task_id,
        description=task_req.description,
        strategy=task_req.strategy,
        status="created",
        steps=task_steps,
        created_at=datetime.now(),
    )

    sessions[task_id] = task_response.model_dump()

    response = ApiResponse(
        success=True, message="任务创建成功", data=task_response.model_dump()
    )

    return jsonify(response.model_dump()), 201


@app.route("/api/tasks/<task_id>")
def get_task(task_id):
    task = sessions.get(task_id)
    if not task:
        response = ApiResponse(success=False, message="任务不存在")
        return jsonify(response.model_dump()), 404

    response = ApiResponse(
        success=True, message="任务详情获取成功", data=task
    )
    return jsonify(response.model_dump())


@app.route("/api/tools")
def list_tools():
    tools = tool_manager.list_tools()
    response = ApiResponse(
        success=True, message="工具列表获取成功", data={"tools": tools}
    )
    return jsonify(response.model_dump())


# == WebSocket 事件 ==
@socketio.on("join")
def on_join(data):
    task_id = data["task_id"]
    join_room(task_id)

    if task_id in sessions:
        emit("task_status", sessions[task_id], to=task_id)


@socketio.on("start_execution")
def start_execution(data):
    task_id = data["task_id"]

    if task_id not in sessions:
        emit("error", {"message": "任务不存在"})
        return

    def execute_task():
        session = sessions[task_id]
        session["status"] = "running"
        socketio.emit("task_started", session, to=task_id)

        # 执行真实的Agent任务
        execute_agent_task(task_id)

    thread = threading.Thread(target=execute_task)
    thread.start()


def execute_agent_task(task_id):
    """执行Agent任务"""
    session = sessions[task_id]
    task = agent_engine.get_task(task_id)

    if not task:
        socketio.emit("error", {"message": "任务不存在"}, to=task_id)
        return

    task_plan = task.steps

    for i, step in enumerate(task_plan):
        step_data = {
            "step": i + 1,
            "total_steps": len(task_plan),
            "tool": step.tool,
            "description": step.description,
        }
        socketio.emit("step_started", step_data, to=task_id)

        # 准备输入数据
        input_data = task.description if i == 0 else session.get("last_output", "")

        # 执行工具
        result = tool_manager.execute_tool(step.tool, input_data)

        # 更新状态
        session["total_tokens"] += result.get("input_tokens", 0) + result.get(
            "output_tokens", 0
        )
        session["total_cost"] += result.get("cost", 0)
        session["last_output"] = str(result.get("data", ""))[:500]

        completion_data = {
            "step": i + 1,
            "total_steps": len(task_plan),
            "total_tokens": result.get("input_tokens", 0)
            + result.get("output_tokens", 0),
            "cost": result.get("cost", 0),
            "result": str(result.get("data", ""))[:200],
            "success": result["success"],
        }

        socketio.emit("step_completed", completion_data, to=task_id)

        time.sleep(1.5)

    session["status"] = "completed"
    session["completed_at"] = datetime.now()
    socketio.emit("task_completed", session, to=task_id)


if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5000)
