import json

from config import Config
from bash import Bash
from helpers import Messages, LLM

def main(config: Config):
    bash = Bash(config)
    # The model
    llm = LLM(config)
    # The conversation history, with the system prompt
    messages = Messages(config.system_prompt)
    print("[INFO] Type 'quit' at any time to exit the agent loop.\n")

    # The main agent loop
    while True:
        # Get user message.
        user = input(f"['{bash.cwd}' 🙂] ").strip()
        if user.lower() == "quit":
            print("\n[🤖] Shutting down. Bye!\n")
            break
        if not user:
            continue
        # Always tell the agent where the current working directory is to avoid confusions.
        user += f"\n Current working directory: `{bash.cwd}`"
        messages.add_user_message(user)

        # The tool-call/response loop
        while True:
            print("\n[🤖] Thinking...")
            response, tool_calls = llm.query(messages, [bash.to_json_schema()])

            if response:
                response = response.strip()
                # Do not store the thinking part to save context space
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()

                # Add the (non-empty) response to the context
                if response:
                    messages.add_assistant_message(response)

            # Process tool calls
            if tool_calls:
                for tc in tool_calls:
                    function_name = tc.function.name
                    function_args = json.loads(tc.function.arguments)

                    # Ensure it's calling the right tool
                    if function_name != "exec_bash_command" or "cmd" not in function_args:
                        tool_call_result = {"error": "Incorrect tool or function argument"}
                    else:
                        command = function_args["cmd"]
                        # Execute command directly (allowlist validation happens in bash.exec_bash_command)
                        print(f"    ▶️   Executing: {command}")
                        tool_call_result = bash.exec_bash_command(command)

                    messages.add_tool_message(tool_call_result, tc.id)
            else:
                # Display the assistant's message to the user.
                if response:
                    print(response)
                    print("-" * 80 + "\n")
                break

if __name__ == "__main__":
    # Load the configuration
    config = Config()
    main(config)