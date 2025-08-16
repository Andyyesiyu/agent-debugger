from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room
from datetime import datetime
import threading
from agent_engine.core import AgentDecisionEngine
from agent_engine.executor import execute_agent_task
from models.schemas import ApiResponse, TaskRequest
from tools.manager import tool_manager

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局Agent引擎
agent_engine = AgentDecisionEngine()
sessions = {}


# == API 路由 ==
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/debug")
def debug_page():
    """Simple page for observing agent execution in real time."""
    return render_template("debug.html")


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

    task_data = {
        "id": task_id,
        "description": task_req.description,
        "strategy": task_req.strategy,
        "status": "created",
        "steps": [
            {
                "step_id": step.step_id,
                "tool": step.tool,
                "description": step.description,
                "status": "pending",
            }
            for step in plan
        ],
        "total_tokens": 0,
        "total_cost": 0.0,
        "created_at": datetime.now(),
    }

    sessions[task_id] = task_data

    response = ApiResponse(
        success=True, message="任务创建成功", data=task_data
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
        socketio.emit("task_status", sessions[task_id], to=task_id)


@socketio.on("start_execution")
def start_execution(data):
    task_id = data["task_id"]

    if task_id not in sessions:
        socketio.emit("error", {"message": "任务不存在"})
        return

    def execute_task():
        session = sessions[task_id]
        session["status"] = "running"
        socketio.emit("task_started", session, to=task_id)

        execute_agent_task(
            agent_engine,
            tool_manager,
            task_id,
            session,
            lambda event, data: socketio.emit(event, data, to=task_id),
        )

    thread = threading.Thread(target=execute_task)
    thread.start()


if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5000)
