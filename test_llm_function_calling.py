#!/usr/bin/env python3
"""
Test script for LLM Function Calling capabilities in Agent Debugger

This script demonstrates how to use the new LLM-driven function calling features.
"""

import asyncio
import os
from agent_engine.core import AgentDecisionEngine, LLMFunctionCallingEngine
from tools.manager import tool_manager

async def test_llm_function_calling():
    """Test the LLM function calling engine"""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🚀 Testing LLM Function Calling Engine...")
    
    # Initialize the engine
    agent_engine = AgentDecisionEngine()
    agent_engine.set_tool_manager(tool_manager)
    
    # Create a test task
    task_id = agent_engine.create_task(
        description="Search for information about Python FastAPI framework and then analyze the results to create a summary report",
        strategy="balanced"
    )
    
    print(f"📝 Created task: {task_id}")
    print(f"📋 Task description: Search for information about Python FastAPI framework and then analyze the results to create a summary report")
    
    # Define a progress callback to see real-time updates
    async def progress_callback(event_data):
        event_type = event_data.get("type")
        
        if event_type == "reasoning_step":
            print(f"\n🧠 LLM Reasoning Step {event_data['step']}:")
            print(f"   💭 Reasoning: {event_data['reasoning'][:100]}...")
            print(f"   🔧 Function calls: {event_data['function_calls']}")
            print(f"   🎯 Tokens: {event_data['tokens']}")
            print(f"   💰 Cost: ${event_data['cost']:.4f}")
            
        elif event_type == "function_call":
            print(f"\n⚡ Function Call in Step {event_data['step']}:")
            print(f"   🛠️  Function: {event_data['function_name']}")
            print(f"   📥 Arguments: {event_data['arguments']}")
            print(f"   📤 Success: {event_data['result']['success']}")
            if event_data['result']['success']:
                result_preview = str(event_data['result']['data'])[:100]
                print(f"   📊 Result preview: {result_preview}...")
    
    try:
        # Execute the task with LLM
        print("\n🎯 Starting LLM-driven execution...")
        result = await agent_engine.execute_with_llm(task_id, progress_callback)
        
        print("\n" + "="*60)
        print("📊 EXECUTION RESULTS")
        print("="*60)
        
        if result["success"]:
            print("✅ Task completed successfully!")
            print(f"📝 Final result: {result['result']}")
        else:
            print("❌ Task failed!")
            print(f"💥 Error: {result['error']}")
        
        print(f"\n📈 Execution Statistics:")
        print(f"   🔄 Reasoning steps: {result['reasoning_steps']}")
        print(f"   🎯 Total tokens: {result['total_tokens']}")
        print(f"   💰 Total cost: ${result['total_cost']:.4f}")
        
        # Get the task details
        task = agent_engine.get_task(task_id)
        print(f"\n📋 Task Details:")
        print(f"   🆔 ID: {task.id}")
        print(f"   📊 Status: {task.status}")
        print(f"   🔧 Total steps: {len(task.steps)}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

async def test_function_schemas():
    """Test function schema generation"""
    print("\n🔧 Testing Function Schema Generation...")
    
    schemas = tool_manager.get_function_schemas()
    print(f"📋 Generated {len(schemas)} function schemas:")
    
    for schema in schemas:
        function_info = schema["function"]
        print(f"\n🛠️  {function_info['name']}:")
        print(f"   📝 Description: {function_info['description']}")
        print(f"   📥 Parameters: {list(function_info['parameters']['properties'].keys())}")

def main():
    """Main test function"""
    print("🧪 Agent Debugger LLM Function Calling Test Suite")
    print("="*60)
    
    # Test function schemas
    asyncio.run(test_function_schemas())
    
    # Test LLM function calling
    asyncio.run(test_llm_function_calling())
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()
