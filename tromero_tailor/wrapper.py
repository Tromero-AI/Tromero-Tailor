import json
from openai import OpenAI
from openai.resources import Chat
from openai.resources.chat.completions import (
    Completions
)
from openai._compat import cached_property
import datetime
from .tromero_requests import post_data, tromero_model_create
from .tromero_utils import mock_openai_format

    

class MockCompletions(Completions):
    def __init__(self, client):
        super().__init__(client)

    def _choice_to_dict(self, choice):
        return {
            "finish_reason": choice.finish_reason,
            "index": choice.index,
            "logprobs": choice.logprobs,
            "message": {
                "content": choice.message.content,
                "role": choice.message.role,
            }
        }
    
    def _save_data(self, data):
        post_data(data, self._client.tromero_key)

    def check_model(self, model):
        try:
            models = self._client.models.list()
        except:
            return False
        model_names = [m.id for m in models]
        return model in model_names
    
    def create(self, *args, **kwargs):
        input = {"model": kwargs['model'], "messages": kwargs['messages']}
        tags = kwargs.get('tags', [])
        formatted_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'messages']}
        if self.check_model(kwargs['model']):
            res = Completions.create(self, *args, **kwargs)  
            if hasattr(res, 'choices'):
                usage = res.usage.model_dump()
                for choice in res.choices:
                    formatted_choice = self._choice_to_dict(choice)
                    self._save_data({"messages": input['messages'] + [formatted_choice['message']],
                                    "model": input['model'],
                                    "kwargs": formatted_kwargs,
                                    "creation_time": str(datetime.datetime.now().isoformat()),
                                    "usage": usage,
                                    "tags": tags
                                    })
        else:
            messages = kwargs['messages']
            res = tromero_model_create(kwargs['model'], messages, self._client.tromero_key)
            # check if res has field 'generated_text'
            if 'generated_text' in res:
                res = mock_openai_format(res['generated_text'])
        return res


class MockChat(Chat):
    def __init__(self, client):
        super().__init__(client)

    @cached_property
    def completions(self) -> Completions:
        return MockCompletions(self._client)


class TailorAI(OpenAI):
    chat: MockChat
    def __init__(self, api_key, tromero_key):
        super().__init__(api_key=api_key)
        self.current_prompt = []
        self.chat = MockChat(self)
        self.tromero_key = tromero_key
