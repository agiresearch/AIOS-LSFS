class Query:
    def __init__(self,
            messages,
            tools = None,
            action_type: str = "llm_message",
            message_return_type = "text"
        ) -> None:
        """Query format

        Args:
            messages (list):
            [
                {"role": "xxx", content_key: content_value}
            ]
            tools (optional): tools that are used for function calling. Defaults to None.
        """
        self.messages = messages
        self.tools = tools
        self.action_type = action_type
        self.message_return_type = message_return_type

class Response:
    def __init__(
            self,
            response_message,
            tool_calls: list = None
        ) -> None:
        """Response format

        Args:
            response_message (str): "generated_text"
            tool_calls (list, optional):
            [
                {"name": "xxx", "parameters": {}}
            ].
            Default to None.
        """
        self.response_message = response_message
        self.tool_calls = tool_calls
