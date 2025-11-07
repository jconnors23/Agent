from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from config import Config

class Messages:
    """
    An abstraction for a list of system/user/assistant/tool messages.
    """

    def __init__(self, system_message: str = ""):
        self.system_message: Optional[Dict[str, str]] = None
        self.messages: List[Dict[str, Any]] = []
        self.set_system_message(system_message)

    def set_system_message(self, message):
        self.system_message = {"role": "system", "content": message}

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def add_tool_message(self, message, id):
        self.messages.append({"role": "tool", "content": str(message), "tool_call_id": id})

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert to a list of messages.
        """
        result = []
        if self.system_message:
            result.append(self.system_message)
        result.extend(self.messages)
        return result

class LLM:
    """
    An abstraction to prompt an LLM with OpenAI compatible endpoint.
    """

    def __init__(self, config: Config):
        self.client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
        self.config = config
        print(f"Using model '{config.llm_model_name}' from '{config.llm_base_url}'")

    def query(
        self,
        messages: Messages,
        tools: List[Dict[str, Any]],
        max_tokens=None,
    ) -> Tuple[Optional[str], List[ChatCompletionMessageToolCall]]:
        completion = self.client.chat.completions.create(
            model=self.config.llm_model_name,
            messages=messages.to_list(),  # type: ignore
            tools=tools,  # type: ignore
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
            max_tokens=max_tokens,
            stream=False
        )

        return (
            completion.choices[0].message.content,
            completion.choices[0].message.tool_calls or [],
        )