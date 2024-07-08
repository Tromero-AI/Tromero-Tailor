import json
import datetime
import warnings
from openai import AsyncOpenAI
from openai.resources import Chat
from openai.resources.chat.completions import (
    AsyncCompletions
)
from openai._compat import cached_property
from .async_tromero_requests import post_data, tromero_model_create, get_model_url, tromero_model_create_stream
from .tromero_utils import mock_openai_format
from .wrapper import MockCompletions, MockChat, TailorAI

class AsyncMockCompletions(AsyncCompletions, MockCompletions):
    def __init__(self, client):
        super().__init__(client)
    
    async def _save_data(self, data):
        if self._client.save_data:
            await post_data(data, self._client.tromero_key)
            
    async def _stream_response(self, response, init_save_data, fall_back_dict):
        try:
            full_message = ''
            async for chunk in response:
                if chunk:
                    if chunk.choices[0].delta.content and chunk.choices[0].delta.content != '</s>':
                        full_message += str(chunk.choices[0].delta.content)
                    yield chunk
        except Exception as e:
            print("Error streaming response:", e, flush=True)
            raise e
        finally:
            if init_save_data != {}:
                init_save_data['messages'].append({"role": "assistant", "content": full_message})
                await self._save_data(init_save_data)

    async def check_model(self, model):
        try:
            models = self._client.models.list()
        except Exception as e:
            print(f"Error checking model: {e}")
            return False
        model_names = []
        async for m in models:
            model_names.append(m.id)
        return model in model_names
    
    async def create(self, *args, **kwargs):
        print("Creating completion", flush=True)
        messages = kwargs['messages']
        formatted_messages = self._format_messages(messages)
        model = kwargs['model']
        stream = kwargs.get('stream', False)
        tags = kwargs.get('tags', [])
        send_kwargs = {}
        use_fallback = kwargs.get('use_fallback', True)
        fallback_model = kwargs.get('fallback_model', '')
        
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['tags', 'use_fallback', 'fallback_model']} 
        if await self.check_model(kwargs['model']):
            res = AsyncCompletions.create(self, *args, **openai_kwargs)  
            send_kwargs = openai_kwargs
        else:
            formatted_kwargs = self._format_kwargs(kwargs)
            send_kwargs = formatted_kwargs
            model_name = model
            if model_name not in self._client.model_urls:
                url, base_model = await get_model_url(model_name, self._client.tromero_key)
                self._client.model_urls[model_name] = url
                self._client.is_base_model[model_name] = base_model
            model_request_name = model_name if not self._client.is_base_model[model_name] else "NO_ADAPTER"
            if stream:
                res, e = await tromero_model_create_stream(model_request_name, self._client.model_urls[model_name], formatted_messages, self._client.tromero_key, parameters=formatted_kwargs)
                if e:
                    if use_fallback and fallback_model:
                        print("Error in making request to model. Using fallback model.")
                        kwargs['model'] = fallback_model
                        kwargs['use_fallback'] = False
                        return await self.create(*args, **kwargs)
            else:
                res = await tromero_model_create(model_request_name, self._client.model_urls[model_name], formatted_messages, self._client.tromero_key, parameters=formatted_kwargs)
                if 'generated_text' in res:
                    generated_text = res['generated_text']
                    res = mock_openai_format(generated_text)

        if hasattr(res, 'choices'):
            for choice in res.choices:
                formatted_choice = self._choice_to_dict(choice)
                save_data = {
                    "messages": formatted_messages + [formatted_choice['message']],
                    "model": model,
                    "kwargs": send_kwargs,
                    "creation_time": str(datetime.datetime.now().isoformat()),
                    "tags": self._tags_to_string(tags)
                }
                if hasattr(res, 'usage'):
                    save_data['usage'] = res.usage.model_dump()
                await self._save_data(save_data)
        elif stream:
            init_save_data = {
                "messages": formatted_messages,
                "model": model,
                "kwargs": send_kwargs,
                "creation_time": str(datetime.datetime.now().isoformat()),
                "tags": self._tags_to_string(tags)
            }
            fall_back_dict = {}
            if use_fallback and fallback_model:
                kwargs['model'] = fallback_model
                kwargs['use_fallback'] = False
                fall_back_dict = {
                    'args': args,
                    'kwargs': kwargs
                }
            return self._stream_response(res, init_save_data, fall_back_dict)
        else:
            if use_fallback and fallback_model:
                print("Error in making request to model. Using fallback model.")
                kwargs['model'] = fallback_model
                kwargs['use_fallback'] = False
                return await self.create(*args, **kwargs)

        return res

class AsyncMockChat(MockChat):
    def __init__(self, client):
        super().__init__(client)

    @cached_property
    def completions(self) -> AsyncCompletions:
        return AsyncMockCompletions(self._client)

class AsyncTailorAI(AsyncOpenAI):
    chat: AsyncMockChat
    def __init__(self, api_key, tromero_key, save_data=True):
        super().__init__(api_key=api_key)
        self.current_prompt = []
        self.model_urls = {}
        self.is_base_model = {}
        self.tromero_key = tromero_key
        self.chat = AsyncMockChat(self)
        self.save_data = save_data
