import json
from openai import OpenAI
from openai.resources import Chat
from openai.resources.chat.completions import (
    Completions
)
from openai._compat import cached_property
import datetime
from .tromero_requests import post_data, tromero_model_create


class MockCompletions(Completions):
    def __init__(self, client, log_file):
        super().__init__(client)
        self.log_file = log_file

    def _choice_to_dict(self, choice):
        return {
            "finish_reason": choice.finish_reason,
            "index": choice.index,
            "logprobs": choice.logprobs,
            "message": {
                "content": choice.message.content,
                "role": choice.message.role,
                "function_call": choice.message.function_call,
                "tool_calls": choice.message.tool_calls
            }
        }
    
    def _save_data(self, data):
        for message in data["messages"]:
            if message["role"] == "user":
                message["role"] = "human"
            elif message["role"] == "assistant":
                message["role"] = "gpt"
            message["value"] = message.pop("content")

        post_data(data, self._client.tromero_key)
        with open(self.log_file, 'r+') as f:
            log_data = json.load(f)
            log_data.append(data)
            f.seek(0)  # Move to the beginning of the file before writing
            json.dump(log_data, f, indent=4)

    def check_model(self, model):
        models = self._client.models.list()
        model_names = [m.id for m in models]
        return model in model_names
    
    def create(self, *args, **kwargs):
        input = {"model": kwargs['model'], "messages": kwargs['messages']}
        formatted_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'messages']}
        if self.check_model(kwargs['model']):
            res = Completions.create(self, *args, **kwargs)  
        else:
            res = tromero_model_create(kwargs['model'], kwargs['messages'], self._client.tromero_key)

        if hasattr(res, 'choices'):
            for choice in res.choices:
                formatted_choice = self._choice_to_dict(choice)
                self._save_data({"messages": input['messages'] + [formatted_choice['message']],
                                   "model": input['model'],
                                   "kwargs": formatted_kwargs,
                                   "creation_time": str(datetime.datetime.now().isoformat()),
                                    })
        return res


class MockChat(Chat):
    def __init__(self, client, log_file):
        super().__init__(client)
        self.log_file = log_file

    @cached_property
    def completions(self) -> Completions:
        return MockCompletions(self._client, self.log_file)


class TailorAI(OpenAI):
    chat: MockChat
    def __init__(self, api_key, tromero_key, log_file='openai_interactions.json'):
        super().__init__(api_key=api_key)
        self.log_file = log_file
        self._ensure_log_file()
        self.current_prompt = []
        self.chat = MockChat(self, log_file)
        self.tromero_key = tromero_key

    def _ensure_log_file(self):
        try:
            with open(self.log_file, 'r') as f:
                json.load(f)  # Try to load the JSON to ensure it's valid
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or JSON is invalid, initialize file with an empty list
            with open(self.log_file, 'w') as f:
                json.dump([], f)
