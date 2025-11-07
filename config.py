import os
from dataclasses import dataclass, field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

@dataclass
class Config:
    """
    Configuration class for the application.
    """

    # -------------------------------------
    # LLM configuration
    #--------------------------------------

    # NVIDIA NIM API (hosted service - no local GPU needed)
    # Nemotron-Nano-9B-v2 supports function calling (tool use)
    # Get your API key at: https://build.nvidia.com/ (NVIDIA API Catalog)
    # API key should be set in .env file as NVIDIA_API_KEY=your-key-here
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model_name: str = "nvidia/nvidia-nemotron-nano-9b-v2"
    llm_api_key: str = os.getenv("NVIDIA_API_KEY", "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")
    # Sampling parameters (we've reduced the temperature to make the model more deterministic)
    llm_temperature: float = 0.1
    llm_top_p: float = 0.95

    # -------------------------------------
    # Agent configuration
    #--------------------------------------

    # The directory path that the agent can access and operate in.
    root_dir: str = os.path.dirname(os.path.abspath(__file__))

    # The list of commands that the agent can execute.
    #
    # WARNING: Be very careful about which commands you allow here.
    #          By running this code you assume all responsibility for
    #          unintended consequences of command execution.
    allowed_commands: list = field(default_factory=lambda: [
        "cd", "cp", "ls", "cat", "find", "touch", "echo", "grep", "pwd", "mkdir", "wget", "sort", "head", "tail", "du", "wc",
    ])

    @property
    def system_prompt(self) -> str:
        """Generate the system prompt for the LLM based on allowed commands."""
        return f"""/think

You are a helpful and very concise Bash assistant with the ability to execute commands in the shell.
You engage with users to help answer questions about bash commands, or execute their intent.
If user intent is unclear, keep engaging with them to figure out what they need and how to best help
them. If they ask question that are not relevant to bash or computer use, decline to answer.

When a command is executed, you will be given the output from that command and any errors. Based on
that, either take further actions or yield control to the user.

The bash interpreter's output and current working directory will be given to you every time a
command is executed. Take that into account for the next conversation.
If there was an error during execution, tell the user what that error was exactly.

You are only allowed to execute the following commands. Break complex tasks into shorter commands from this list:

```
{self.allowed_commands}
```

**Never** attempt to execute a command not in this list. **Never** attempt to execute dangerous commands
like `rm`, `mv`, `rmdir`, `sudo`, etc. If the user asks you to do so, politely refuse.
"""