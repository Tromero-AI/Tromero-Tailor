import json
from openai import OpenAI
from openai.resources import Chat
from openai.resources.chat.completions import (
    Completions
)
from openai._compat import cached_property
import datetime
from .tromero_requests import post_data, tromero_model_create, get_model_url, tromero_model_create_stream
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
        if self._client.save_data:
            post_data(data, self._client.tromero_key)

    def _format_kwargs(self, kwargs):
        keys_to_keep = [
            "best_of", "decoder_input_details", "details", "do_sample", 
            "max_new_tokens", "ignore_eos_token", "repetition_penalty", 
            "return_full_outcome", "seed", "stop", "temperature", "top_k", 
            "top_p", "truncate", "typical_p", "watermark", "schema", 
            "adapter_id", "adapter_source", "merged_adapters", "response_format"
        ]
        parameters = {key: kwargs[key] for key in keys_to_keep if key in kwargs}
        return {key: kwargs[key] for key in keys_to_keep if key in kwargs}


    def _format_messages(self, messages):
        system_prompt = ""
        num_prompts = 0
        for message in messages:
            if message['role'] == "system":
                system_prompt += message['content'] + " "
                num_prompts += 1
            else:
                break
        if num_prompts <= 1:
            return messages

        messages = [{"role": "system", "content": system_prompt}] + messages[num_prompts:]
        print("Warning: Multiple system prompts will be combined into one prompt when saving data or calling custom models.")
        return messages
    
    def _tags_to_string(self, tags):
        return ",".join(tags)

    def check_model(self, model):
        try:
            models = self._client.models.list()
        except:
            return False
        model_names = [m.id for m in models]
        return model in model_names
    
    def create(self, *args, **kwargs):
        messages = kwargs['messages']
        formatted_messages =  self._format_messages(messages)
        model = kwargs['model']
        stream = kwargs.get('stream', False)
        tags = kwargs.get('tags', [])
        formatted_kwargs = self._format_kwargs(kwargs)
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['tags']}
        if self.check_model(kwargs['model']):
            res = Completions.create(self, *args, **openai_kwargs)  
            if hasattr(res, 'choices'):
                usage = res.usage.model_dump()
                for choice in res.choices:
                    formatted_choice = self._choice_to_dict(choice)
                    self._save_data({"messages": formatted_messages + [formatted_choice['message']],
                                    "model": model,
                                    "kwargs": formatted_kwargs,
                                    "creation_time": str(datetime.datetime.now().isoformat()),
                                    "usage": usage,
                                    "tags": self._tags_to_string(tags)
                                    })
        else:
            model_name = model
            if model_name not in self._client.model_urls:
                url, base_model = get_model_url(model_name, self._client.tromero_key)
                self._client.model_urls[model_name] = url
                self._client.is_base_model[model_name] = base_model
            model_request_name = model_name if not self._client.is_base_model[model_name] else "NO_ADAPTER"
            if stream:
                return tromero_model_create_stream(model_request_name, self._client.model_urls[model_name], formatted_messages, self._client.tromero_key, parameters=formatted_kwargs)
            else:
                res = tromero_model_create(model_request_name, self._client.model_urls[model_name], formatted_messages, self._client.tromero_key, parameters=formatted_kwargs)
            # check if res has field 'generated_text'
            if 'generated_text' in res:
                generated_text = res['generated_text']
                res = mock_openai_format(generated_text)
                
                self._save_data({"messages": formatted_messages + [{"role": "assistant", "content": generated_text}],
                                    "model": model_name,
                                    "kwargs": formatted_kwargs,
                                    "creation_time": str(datetime.datetime.now().isoformat()),
                                    "tags": self._tags_to_string(tags + ['custom', model_name])
                                    })
        return res


class MockChat(Chat):
    def __init__(self, client):
        super().__init__(client)

    @cached_property
    def completions(self) -> Completions:
        return MockCompletions(self._client)


class TailorAI(OpenAI):
    chat: MockChat
    def __init__(self, api_key, tromero_key, save_data=True):
        super().__init__(api_key=api_key)
        self.current_prompt = []
        self.model_urls = {}
        self.is_base_model = {}
        self.tromero_key = tromero_key
        self.chat = MockChat(self)
        self.save_data = save_data
