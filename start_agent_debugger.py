#!/usr/bin/env python3
"""
Agent Debugger Startup Script

This script provides an easy way to start the Agent Debugger with LLM function calling capabilities.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("   You can still run the application, but LLM-driven execution will not work")
        print("   To enable LLM features, set your API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print()
    else:
        print("✅ OpenAI API key found")
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        print("✅ uv package manager found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv package manager not found")
        print("   Please install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    return True

def install_dependencies():
    """Install dependencies using uv"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def start_application(mode="fastapi", port=5000):
    """Start the application"""
    print(f"🚀 Starting Agent Debugger ({mode} mode) on port {port}...")
    
    try:
        if mode == "fastapi":
            # Start FastAPI application
            subprocess.run([
                "uv", "run", "uvicorn", 
                "app_fastapi:app", 
                "--reload", 
                "--port", str(port),
                "--host", "0.0.0.0"
            ], check=True)
        else:
            # Start Flask application
            subprocess.run([
                "uv", "run", "python", "app.py"
            ], check=True)
            
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")

def show_usage():
    """Show usage information"""
    print("🤖 Agent Debugger with LLM Function Calling")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python start_agent_debugger.py [OPTIONS]")
    print()
    print("Options:")
    print("  --mode fastapi|flask    Application mode (default: fastapi)")
    print("  --port PORT            Port to run on (default: 5000)")
    print("  --install              Install dependencies first")
    print("  --demo                 Run demo script instead")
    print("  --test                 Run test script instead")
    print("  --help                 Show this help message")
    print()
    print("Examples:")
    print("  python start_agent_debugger.py")
    print("  python start_agent_debugger.py --mode flask --port 8000")
    print("  python start_agent_debugger.py --install")
    print("  python start_agent_debugger.py --demo")
    print()
    print("Features:")
    print("  ✨ LLM-driven function calling with OpenAI")
    print("  🔧 Dynamic tool selection and chaining")
    print("  📊 Real-time WebSocket monitoring")
    print("  💰 Token usage and cost tracking")
    print("  🎯 Multiple execution strategies")
    print()

def run_demo():
    """Run the demo script"""
    print("🎬 Running LLM Function Calling Demo...")
    try:
        subprocess.run(["uv", "run", "python", "demo_llm_agent.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run demo: {e}")

def run_test():
    """Run the test script"""
    print("🧪 Running LLM Function Calling Tests...")
    try:
        subprocess.run(["uv", "run", "python", "test_llm_function_calling.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run tests: {e}")

def main():
    """Main entry point"""
    args = sys.argv[1:]
    
    # Parse arguments
    mode = "fastapi"
    port = 5000
    install_deps = False
    run_demo_mode = False
    run_test_mode = False
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg in ["--help", "-h"]:
            show_usage()
            return
        elif arg == "--mode":
            if i + 1 < len(args):
                mode = args[i + 1]
                i += 1
            else:
                print("❌ --mode requires a value (fastapi or flask)")
                return
        elif arg == "--port":
            if i + 1 < len(args):
                try:
                    port = int(args[i + 1])
                    i += 1
                except ValueError:
                    print("❌ --port requires a numeric value")
                    return
            else:
                print("❌ --port requires a value")
                return
        elif arg == "--install":
            install_deps = True
        elif arg == "--demo":
            run_demo_mode = True
        elif arg == "--test":
            run_test_mode = True
        else:
            print(f"❌ Unknown argument: {arg}")
            show_usage()
            return
        
        i += 1
    
    # Validate mode
    if mode not in ["fastapi", "flask"]:
        print("❌ Mode must be 'fastapi' or 'flask'")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Install dependencies if requested
    if install_deps:
        if not install_dependencies():
            return
    
    # Run demo or test if requested
    if run_demo_mode:
        run_demo()
        return
    
    if run_test_mode:
        run_test()
        return
    
    # Show startup information
    print("🤖 Agent Debugger with LLM Function Calling")
    print("=" * 50)
    print(f"📍 Mode: {mode}")
    print(f"🌐 Port: {port}")
    print(f"🔗 URL: http://localhost:{port}")
    print()
    
    if os.getenv("OPENAI_API_KEY"):
        print("✅ LLM function calling enabled")
    else:
        print("⚠️  LLM function calling disabled (no API key)")
    
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the application
    start_application(mode, port)

if __name__ == "__main__":
    main()
