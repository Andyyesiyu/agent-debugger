#!/usr/bin/env python3
"""
Demo script showing LLM Function Calling capabilities

This script demonstrates how to create and execute tasks using the new
LLM-driven function calling system.
"""

import asyncio
import json
import os
from datetime import datetime
from agent_engine.core import AgentDecisionEngine
from tools.manager import tool_manager


class AgentDemo:
    def __init__(self):
        self.agent_engine = AgentDecisionEngine()
        self.agent_engine.set_tool_manager(tool_manager)

    def print_header(self, title):
        """Print a formatted header"""
        print("\n" + "=" * 60)
        print(f"🤖 {title}")
        print("=" * 60)

    def print_step(self, step, description):
        """Print a formatted step"""
        print(f"\n📋 Step {step}: {description}")
        print("-" * 40)

    async def demo_basic_llm_execution(self):
        """Demonstrate basic LLM-driven task execution"""
        self.print_header("Basic LLM Function Calling Demo")

        # Create a simple task
        task_description = "Find information about artificial intelligence trends in 2024 and create a brief summary"

        self.print_step(1, "Creating Task")
        task_id = self.agent_engine.create_task(
            description=task_description, strategy="balanced"
        )
        print(f"✅ Task created with ID: {task_id}")
        print(f"📝 Description: {task_description}")

        # Define progress callback
        async def progress_callback(event_data):
            event_type = event_data.get("type")

            if event_type == "reasoning_step":
                print(f"\n🧠 LLM Reasoning (Step {event_data['step']}):")
                print(f"   💭 {event_data['reasoning'][:80]}...")
                print(f"   🔧 Function calls planned: {event_data['function_calls']}")
                print(f"   💰 Cost so far: ${event_data['cost']:.4f}")

            elif event_type == "function_call":
                print(f"\n⚡ Executing Function:")
                print(f"   🛠️  {event_data['function_name']}")
                print(f"   📥 Input: {str(event_data['arguments'])[:50]}...")
                success = event_data["result"]["success"]
                print(
                    f"   {'✅' if success else '❌'} Result: {'Success' if success else 'Failed'}"
                )

        self.print_step(2, "Executing with LLM")
        try:
            result = await self.agent_engine.execute_with_llm(
                task_id, progress_callback
            )

            self.print_step(3, "Results")
            if result["success"]:
                print("✅ Task completed successfully!")
                print(f"\n📊 Final Result:")
                print(f"{result['result']}")

                print(f"\n📈 Execution Statistics:")
                print(f"   🔄 Reasoning steps: {result['reasoning_steps']}")
                print(f"   🎯 Total tokens: {result['total_tokens']}")
                print(f"   💰 Total cost: ${result['total_cost']:.4f}")
            else:
                print("❌ Task failed!")
                print(f"💥 Error: {result['error']}")

        except Exception as e:
            print(f"❌ Execution error: {e}")

    async def demo_multi_step_reasoning(self):
        """Demonstrate multi-step reasoning with tool chaining"""
        self.print_header("Multi-Step Reasoning Demo")

        task_description = """
        Research the latest developments in Python web frameworks, 
        analyze the key features and benefits, 
        then generate a comparison report with recommendations
        """

        self.print_step(1, "Creating Complex Task")
        task_id = self.agent_engine.create_task(
            description=task_description,
            strategy="accuracy",  # Use accuracy strategy for thorough analysis
        )
        print(f"✅ Complex task created: {task_id}")

        step_counter = [0]  # Use list to allow modification in nested function

        async def detailed_progress_callback(event_data):
            event_type = event_data.get("type")

            if event_type == "reasoning_step":
                step_counter[0] += 1
                print(f"\n🧠 Reasoning Step {step_counter[0]}:")
                print(f"   💭 LLM is thinking: {event_data['reasoning'][:100]}...")
                if event_data["function_calls"] > 0:
                    print(
                        f"   🎯 Planning to call {event_data['function_calls']} function(s)"
                    )
                else:
                    print(f"   ✨ Providing final answer")

            elif event_type == "function_call":
                print(f"\n⚡ Function Call:")
                print(f"   🛠️  Tool: {event_data['function_name']}")
                args_preview = str(event_data["arguments"]).replace("\n", " ")[:60]
                print(f"   📥 Args: {args_preview}...")

                result = event_data["result"]
                if result["success"]:
                    data_preview = str(result["data"]).replace("\n", " ")[:80]
                    print(f"   ✅ Success: {data_preview}...")
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

        self.print_step(2, "Executing Multi-Step Analysis")
        try:
            result = await self.agent_engine.execute_with_llm(
                task_id, detailed_progress_callback
            )

            self.print_step(3, "Analysis Complete")
            if result["success"]:
                print("✅ Multi-step analysis completed!")
                print(f"\n📋 Generated Report:")
                print("-" * 40)
                print(result["result"])
                print("-" * 40)

                print(f"\n📊 Execution Metrics:")
                print(f"   🔄 Reasoning iterations: {result['reasoning_steps']}")
                print(f"   🎯 Total tokens used: {result['total_tokens']:,}")
                print(f"   💰 Total cost: ${result['total_cost']:.4f}")

                # Get task details
                task = self.agent_engine.get_task(task_id)
                if task is not None:
                    print(f"   🔧 Function calls made: {len(task.steps)}")

            else:
                print("❌ Analysis failed!")
                print(f"💥 Error: {result['error']}")

        except Exception as e:
            print(f"❌ Execution error: {e}")

    def demo_available_tools(self):
        """Show available tools and their capabilities"""
        self.print_header("Available Tools")

        tools = tool_manager.list_tools()

        for i, tool in enumerate(tools, 1):
            print(f"\n🛠️  Tool {i}: {tool['name']}")
            print(f"   📝 Description: {tool['description']}")
            print(f"   💰 Est. cost per use: ${tool['estimated_cost_per_use']:.6f}")
            print(f"   🔄 Supports streaming: {tool['supports_streaming']}")
            print(f"   🔑 Requires API key: {tool['requires_api_key']}")

            if tool["usage_examples"]:
                print(f"   📚 Example: {tool['usage_examples'][0]['description']}")

    async def run_demo(self):
        """Run the complete demo"""
        print("🚀 Agent Debugger LLM Function Calling Demo")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            print("\n❌ OPENAI_API_KEY not found!")
            print("Please set your OpenAI API key:")
            print("export OPENAI_API_KEY='your-api-key-here'")
            return

        # Show available tools
        self.demo_available_tools()

        # Run basic demo
        await self.demo_basic_llm_execution()

        # Ask user if they want to continue with complex demo
        print("\n" + "=" * 60)
        response = input("🤔 Run multi-step reasoning demo? (y/N): ").strip().lower()

        if response in ["y", "yes"]:
            await self.demo_multi_step_reasoning()
        else:
            print("⏭️  Skipping multi-step demo")

        print("\n✨ Demo completed!")
        print("🔗 Try the web interface at: http://localhost:5000")


def main():
    """Main entry point"""
    demo = AgentDemo()
    asyncio.run(demo.run_demo())


if __name__ == "__main__":
    main()
