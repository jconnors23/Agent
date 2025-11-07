from typing import Dict
try:
    from langchain.agents import create_agent  # New import (LangGraph V1.0+)
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent  # Fallback for older versions
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from config import Config
from bash import Bash

def main(config: Config):
    # Create the client
    llm = ChatOpenAI(
        model=config.llm_model_name,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,  # type: ignore
        temperature=config.llm_temperature,
        top_p=config.llm_top_p,
    )
    # Create the tool
    bash = Bash(config)
    
    # Wrapper to log command execution (allowlist validation happens in bash.exec_bash_command)
    def exec_bash_command_with_logging(cmd: str) -> Dict[str, str]:
        """Execute a bash command with logging (no user confirmation)."""
        print(f"    ▶️   Executing: {cmd}")
        return bash.exec_bash_command(cmd)
    
    # Create the agent
    agent = create_agent(
        model=llm,
        tools=[exec_bash_command_with_logging],
        prompt=config.system_prompt,
        checkpointer=InMemorySaver(),
    )
    print("[INFO] Type 'quit' at any time to exit the agent loop.\n")

    # The main loop
    while True:
        user = input(f"['{bash.cwd}' 🙂] ").strip()

        if user.lower() == "quit":
            print("\n[🤖] Shutting down. Bye!\n")
            break
        if not user:
            continue

        # Always tell the agent where the current working directory is to avoid confusions.
        user += f"\n Current working directory: `{bash.cwd}`"
        print("\n[🤖] Thinking...")

        # Run the agent's logic and get the response.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user}]},
            config={"configurable": {"thread_id": "cli"}},  # one ongoing conversation
        )
        # Show the response (without the thinking part, if any)
        if not result.get("messages") or not result["messages"][-1].content:
            print("[ERROR] No response from agent")
            print("-" * 80 + "\n")
            continue
        
        response = result["messages"][-1].content.strip()

        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        if response:
            print(response)
            print("-" * 80 + "\n")

if __name__ == "__main__":
    # Load the configuration
    config = Config()
    main(config)