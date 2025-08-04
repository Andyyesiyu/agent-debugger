#!/usr/bin/env python
"""运行FastAPI版本的Agent Debugger"""

import uvicorn


if __name__ == "__main__":
    # 运行FastAPI应用
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

