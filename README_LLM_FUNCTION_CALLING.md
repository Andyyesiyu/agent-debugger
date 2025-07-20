# LLM Function Calling Integration

This document describes the new LLM function calling capabilities integrated into the Agent Debugger project.

## Overview

The Agent Debugger now supports **real LLM-driven function calling** where an LLM (OpenAI GPT models) can:

1. **Dynamically select and invoke tools** from the existing tool system
2. **Parse function call responses** and use them to make decisions about next steps  
3. **Chain multiple function calls** together based on LLM reasoning
4. **Display the function calling process** in real-time through WebSocket interface
5. **Track token usage and costs** for both LLM reasoning and function executions

## Key Features

### 🧠 Intelligent Tool Selection
- LLM analyzes the task and automatically selects appropriate tools
- Supports chaining multiple tools based on previous results
- Adapts strategy based on execution mode (speed, cost, accuracy, balanced)

### ⚡ Real-time Monitoring
- Live WebSocket updates showing LLM reasoning steps
- Function call execution tracking with results
- Token usage and cost monitoring in real-time

### 🔧 Enhanced Tool System
- Tools now provide OpenAI function schemas automatically
- Better parameter validation and usage examples
- Support for streaming and API key requirements

### 📊 Comprehensive Tracking
- Detailed execution summaries with reasoning traces
- Cost breakdown by reasoning steps and function calls
- Performance metrics and execution time tracking

## Setup

### 1. Install Dependencies
```bash
uv sync
```

### 2. Set OpenAI API Key
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application
```bash
# FastAPI version (recommended)
uv run uvicorn app_fastapi:app --reload --port 5000

# Or Flask version
uv run python app.py
```

## Usage

### API Endpoints

#### Create Task with LLM Execution
```bash
curl -X POST "http://localhost:5000/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Search for Python FastAPI tutorials and create a summary",
    "strategy": "balanced",
    "execution_mode": "llm_driven"
  }'
```

#### Execute Task with LLM
```bash
curl -X POST "http://localhost:5000/api/tasks/{task_id}/execute" \
  -H "Content-Type: application/json" \
  -d '{"execution_mode": "llm_driven"}'
```

#### Get Function Schemas
```bash
curl "http://localhost:5000/api/tools/schemas"
```

### WebSocket Events

Connect to `ws://localhost:5000/ws/{task_id}` to receive real-time updates:

#### LLM Reasoning Events
```json
{
  "type": "llm_reasoning",
  "data": {
    "step": 1,
    "reasoning": "I need to search for information about FastAPI...",
    "function_calls": 1,
    "tokens": 150,
    "cost": 0.0003
  }
}
```

#### Function Call Events  
```json
{
  "type": "function_call",
  "data": {
    "step": 1,
    "function_name": "web_search",
    "arguments": {"input_data": "Python FastAPI tutorial"},
    "result": {"success": true, "data": "..."}
  }
}
```

### Execution Modes

#### Static Mode (Traditional)
- Pre-planned execution steps
- Fixed tool sequence
- Faster but less adaptive

#### LLM-Driven Mode (New)
- Dynamic tool selection by LLM
- Adaptive based on intermediate results
- More intelligent but uses more tokens

## Testing

Run the test script to verify LLM function calling:

```bash
# Set your OpenAI API key first
export OPENAI_API_KEY="your-api-key-here"

# Run the test
uv run python test_llm_function_calling.py
```

## Architecture

### Core Components

1. **LLMFunctionCallingEngine** - Handles OpenAI API integration and function calling
2. **Enhanced Tool System** - Provides function schemas and better metadata
3. **AgentDecisionEngine** - Orchestrates both static and LLM-driven execution
4. **WebSocket Communication** - Real-time progress updates
5. **Pydantic Models** - Type-safe data validation with function calling support

### Execution Flow

1. **Task Creation** - User creates task with description and strategy
2. **LLM Planning** - LLM analyzes task and plans initial approach
3. **Function Calling** - LLM selects and calls appropriate tools
4. **Result Processing** - LLM processes results and decides next steps
5. **Iteration** - Process repeats until task completion or max iterations
6. **Final Result** - LLM provides comprehensive final answer

## Cost Management

### Token Tracking
- Input/output tokens tracked per reasoning step
- Function call overhead included in calculations
- Real-time cost updates during execution

### Strategy-Based Optimization
- **Speed**: Uses faster models, minimal iterations
- **Cost**: Prefers cheaper models, caches results
- **Accuracy**: Uses best models, multiple verification steps
- **Balanced**: Optimizes for speed/cost/accuracy balance

## Available Tools

1. **web_search** - Search the web for information
2. **text_analysis** - Analyze text for insights and sentiment  
3. **content_generation** - Generate structured content and reports

Each tool provides:
- OpenAI function schema
- Parameter validation
- Usage examples
- Cost estimation

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   ```
   Error: Tool manager not set
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **LLM Execution Not Supported**
   ```
   Error: LLM execution engine not configured
   Solution: Ensure tool manager is properly initialized
   ```

3. **High Token Usage**
   ```
   Issue: Costs higher than expected
   Solution: Use "cost" strategy or limit max_iterations
   ```

### Debug Mode

Enable debug logging to see detailed execution traces:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

- Add more specialized tools (file operations, database queries, etc.)
- Implement tool result caching for cost optimization
- Add support for custom LLM models and providers
- Enhance error handling and retry mechanisms
- Add execution history and analytics dashboard
