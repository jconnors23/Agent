import json
import re
import sys
import select
import traceback

from config import Config
from bash import Bash
from helpers import Messages, LLM

def input_with_timeout(prompt: str, timeout: float = 30.0) -> str:
    """
    Get user input with a timeout. Returns empty string if timeout occurs.
    
    Args:
        prompt: The prompt to display to the user
        timeout: Timeout in seconds (default: 30.0)
    
    Returns:
        User input string, or empty string if timeout occurred
    """
    # Check if stdin is a TTY (terminal), otherwise fall back to regular input
    if not sys.stdin.isatty():
        # Not a TTY, use regular input (e.g., when piped or redirected)
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return ""
    
    print(prompt, end='', flush=True)
    
    # Use select to check if input is available (Unix/Linux/WSL)
    try:
        if select.select([sys.stdin], [], [], timeout)[0]:
            try:
                return sys.stdin.readline().strip()
            except (EOFError, KeyboardInterrupt):
                return ""
        else:
            # Timeout occurred
            print("\n[⏱️] No input received within timeout period.")
            return ""
    except (OSError, ValueError):
        # Fallback if select doesn't work (e.g., on Windows without WSL)
        # This shouldn't happen on WSL, but provides a safety net
        try:
            return input().strip()
        except (EOFError, KeyboardInterrupt):
            return ""

def fix_json_escaping(json_str: str) -> str:
    """
    Attempt to fix common JSON escaping issues, particularly with backslashes
    before semicolons in bash commands (e.g., backslash-semicolon should be double-backslash-semicolon in JSON).
    """
    # Fix common issue: \; should be \\; in JSON strings
    # Pattern: find backslash-semicolon sequences that aren't properly escaped
    # We'll look for patterns like: -exec ... {} \;
    # And change \; to \\; but only if it's not already escaped (not \\;)
    # Use a negative lookbehind to ensure we don't double-escape
    backslash_semicolon = re.compile(r'(?<![\\])\\(?=;)')
    fixed = backslash_semicolon.sub(r'\\\\', json_str)
    return fixed

def main(config: Config):
    bash = Bash(config)
    # The model
    llm = LLM(config)
    # The conversation history, with the system prompt
    messages = Messages(config.system_prompt)
    print("[INFO] Type 'quit' at any time to exit the agent loop.")
    print("[INFO] Agent will auto-shutdown after 30 seconds of inactivity.\n")

    # The main agent loop
    while True:
        # Get user message with timeout (30 seconds after agent output)
        user = input_with_timeout(f"['{bash.cwd}' 🙂] ", timeout=30.0).strip()
        
        # Check if timeout occurred (empty string returned)
        if not user:
            print("\n[🤖] Shutting down due to inactivity timeout (30 seconds).\n")
            break
            
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
                    try:
                        # Safely access tool call ID
                        tool_call_id = getattr(tc, 'id', None)
                        
                        # Safely access tool call attributes
                        function_name = getattr(tc.function, 'name', None)
                        if not function_name:
                            tool_call_result = {"error": "Tool call missing function name"}
                            messages.add_tool_message(tool_call_result, tool_call_id)
                            continue
                        
                        # Parse JSON arguments with error handling and auto-fix
                        function_args = None
                        arguments_str = getattr(tc.function, 'arguments', None)
                        if arguments_str is None:
                            tool_call_result = {"error": "Tool call missing arguments"}
                            messages.add_tool_message(tool_call_result, tool_call_id)
                            continue
                        
                        try:
                            function_args = json.loads(arguments_str)
                        except json.JSONDecodeError as e:
                            # Try to fix common escaping issues
                            print(f"    ⚠️   JSON parse failed, attempting to fix escaping...")
                            try:
                                fixed_arguments = fix_json_escaping(arguments_str)
                                function_args = json.loads(fixed_arguments)
                                print(f"    ⚠️   Fixed JSON escaping issue in tool arguments")
                            except (json.JSONDecodeError, Exception) as fix_error:
                                # If fixing didn't work, return a clear error
                                error_msg = (
                                    f"Failed to parse tool arguments as JSON: {str(e)}. "
                                    f"Attempted fix also failed: {str(fix_error)}. "
                                    f"This usually happens when the command contains special characters. "
                                    f"Arguments (first 200 chars): {arguments_str[:200]}"
                                )
                                print(f"    ✗   {error_msg}")
                                tool_call_result = {"error": error_msg}
                                messages.add_tool_message(tool_call_result, tool_call_id)
                                continue

                        # Ensure it's calling the right tool
                        if function_args is None:
                            tool_call_result = {"error": "Failed to parse function arguments"}
                            messages.add_tool_message(tool_call_result, tool_call_id)
                        elif function_name != "exec_bash_command" or "cmd" not in function_args:
                            tool_call_result = {"error": "Incorrect tool or function argument"}
                            messages.add_tool_message(tool_call_result, tool_call_id)
                        else:
                            command = function_args["cmd"]
                            # Execute command directly (allowlist validation happens in bash.exec_bash_command)
                            print(f"    ▶️   Executing: {command}")
                            tool_call_result = bash.exec_bash_command(command)
                            messages.add_tool_message(tool_call_result, tool_call_id)
                            
                            # Print a summary if there was an error
                            if "error" in tool_call_result:
                                print(f"    ✗   Command failed: {tool_call_result['error']}")
                            elif tool_call_result.get("stderr"):
                                print(f"    ⚠️   Command produced stderr: {tool_call_result['stderr'][:100]}")
                            else:
                                # Show a brief success message for long outputs
                                stdout_len = len(tool_call_result.get("stdout", ""))
                                if stdout_len > 1000:
                                    print(f"    ✓   Command executed successfully ({stdout_len} chars of output)")
                    except Exception as e:
                        # Catch any unexpected errors during tool call processing
                        error_msg = f"Unexpected error processing tool call: {type(e).__name__}: {str(e)}"
                        print(f"    ✗   {error_msg}")
                        traceback.print_exc()
                        tool_call_result = {"error": error_msg}
                        # Safely get tool call ID for error message
                        tool_call_id = getattr(tc, 'id', None) if 'tc' in locals() else None
                        messages.add_tool_message(tool_call_result, tool_call_id)
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