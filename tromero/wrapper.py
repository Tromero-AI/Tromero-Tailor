import json
from openai import OpenAI
from openai.resources import Chat
from openai.resources.chat.completions import (
    Completions
)
from openai._compat import cached_property
import datetime
from tromero.tromero_requests import post_data, tromero_model_create, get_model_url, tromero_model_create_stream, embedding_request
from tromero.tromero_utils import mock_openai_format, tags_to_string, EmbeddingResponse
import warnings
import threading
from tromero.fine_tuning import TromeroModels, TromeroData, FineTuningJob, Datasets


class MockCompletions(Completions):
    def __init__(self, client):
        super().__init__(client)

    def _choice_to_dict(self, choice):
        return {
            "message": {
                "content": choice.message.content,
                "role": choice.message.role,
            }
        }
    
    def _save_data(self, data, save_data=True):
        if save_data:
            threading.Thread(target=post_data, args=(data, self._client.tromero_key)).start()


    def _format_kwargs(self, kwargs):
        keys_to_keep = [
            "best_of", "decoder_input_details", "details", "do_sample", 
            "max_new_tokens", "ignore_eos_token", "repetition_penalty", 
            "return_full_outcome", "seed", "stop", "temperature", "top_k", 
            "top_p", "truncate", "typical_p", "watermark", 
            "adapter_id", "adapter_source", "merged_adapters", "response_format", 
            "make_synthetic_version", "guided_schema", "guided_regex", "tools",
            "presence_penalty", "frequency_penalty", "min_p", "use_beam_search", 
            "length_penalty", "early_stopping", "stop_token_ids", "include_stop_str_in_output", 
            "max_tokens", "min_tokens", "logprobs", "prompt_logprobs", 
            "detokenize", "skip_special_tokens", "spaces_between_special_tokens", 
            "logits_processors", "truncate_prompt_tokens", "n", "ignore_eos", "response_format",
        ]

        invalid_key_found = False
        parameters = {}
        for key in kwargs:
            if key not in keys_to_keep and key not in ["tags", "model", "messages", "use_fallback", "fallback_model", "stream", "save_data"]:
                print(f"\033[38;5;214mWarning: {key} is not a valid parameter for the model. This parameter will be ignored.\033[0m")
                invalid_key_found = True
            elif key in keys_to_keep:
                parameters[key] = kwargs[key]

        if invalid_key_found:
            print("\033[38;5;214mThe following parameters are valid for the model: ", keys_to_keep, "\033[0m")
        
        return parameters


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
    
    def _stream_response(self, response, init_data, fall_back_dict, save_data):
        try:
            full_message = ''
            for chunk in response:
                if chunk:
                    if chunk.choices[0].delta.content and chunk.choices[0].delta.content != '</s>':
                        full_message += str(chunk.choices[0].delta.content)
                    yield chunk
        except Exception as e:
            print("Error streaming response:", e, flush=True)
            raise e
        finally:
            if init_data != {}:
                init_data['messages'].append({"role": "assistant", "content": full_message})
                self._save_data(init_data, save_data)


    def check_model(self, model):
        if not self._client.openai_models:
            try:
                models = self._client.models.list()
            except:
                return False
            model_names = [m.id for m in models]
            self._client.openai_models = model_names
        return model in self._client.openai_models
    
    def create(self, *args, **kwargs):
        messages = kwargs['messages']
        formatted_messages =  self._format_messages(messages)
        model = kwargs['model']
        stream = kwargs.get('stream', False)
        tags = kwargs.get('tags', [])
        send_kwargs = {}
        use_fallback = kwargs.get('use_fallback', True)
        fallback_model = kwargs.get('fallback_model', '')
        save_data = kwargs.get('save_data', self._client.save_data_default)
        
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['tags', 'use_fallback', 'fallback_model', 'save_data']}
        is_openai_model = False
        if self._client.api_key:
            is_openai_model = self.check_model(kwargs['model'])
        if is_openai_model:
            res = Completions.create(self, *args, **openai_kwargs)  
            send_kwargs = openai_kwargs
        else:
            formatted_kwargs = self._format_kwargs(kwargs)
            send_kwargs = formatted_kwargs
            model_name = model
            if model_name not in self._client.model_urls:
                url, base_model, _ = get_model_url(model_name, self._client.tromero_key, self._client.location_preference)
                self._client.model_urls[model_name] = url
                self._client.is_base_model[model_name] = base_model
            model_request_name = model_name if not self._client.is_base_model[model_name] else "NO_ADAPTER"
            if stream:
                res, e =  tromero_model_create_stream(model_request_name, self._client.model_urls[model_name], formatted_messages, self._client.tromero_key, parameters=formatted_kwargs)
                if e:
                    if use_fallback and fallback_model:
                        print("Error in making request to model. Using fallback model.")
                        kwargs['model'] = fallback_model
                        kwargs['use_fallback'] = False
                        return self.create(*args, **kwargs)

            else:
                res = tromero_model_create(model_request_name, self._client.model_urls[model_name], formatted_messages, self._client.tromero_key, parameters=formatted_kwargs)
                # check if res has field 'generated_text'
                if 'generated_text' in res:
                    generated_text = res['generated_text']
                    usage = res['usage']
                    res = mock_openai_format(generated_text, usage)

        if hasattr(res, 'choices'):
            for choice in res.choices:
                formatted_choice = self._choice_to_dict(choice)
                data = {"messages": formatted_messages + [formatted_choice['message']],
                                "model": model,
                                "kwargs": send_kwargs,
                                "creation_time": str(datetime.datetime.now().isoformat()),
                                "tags": tags_to_string(tags)
                                }
                # if hasattr(res, 'usage'):
                #     save_data['usage'] = res.usage.model_dump()
                self._save_data(data, save_data)
        elif stream:
            init_data = {"messages": formatted_messages,
                                "model": model,
                                "kwargs": send_kwargs,
                                "creation_time": str(datetime.datetime.now().isoformat()),
                                "tags": tags_to_string(tags)
                                }
            fall_back_dict = {}
            if use_fallback and fallback_model:
                kwargs['model'] = fallback_model
                kwargs['use_fallback'] = False
                fall_back_dict = {
                    'args': args,
                    'kwargs': kwargs
                }
            return self._stream_response(res, init_data, fall_back_dict, save_data)
        else:
            if use_fallback and fallback_model:
                print("Error in making request to model. Using fallback model.")
                kwargs['model'] = fallback_model
                kwargs['use_fallback'] = False
                return self.create(*args, **kwargs)

        return res
    
class TromeroEmbeddings:
    def __init__(self, client):
        self._client = client

    def create(self, inputs, model, **kwargs):
        model_name = model
        if model_name not in self._client.model_urls:
            url, base_model, embedding_model = get_model_url(model_name, self._client.tromero_key, self._client.location_preference)
            self._client.model_urls[model_name] = url
            self._client.is_base_model[model_name] = base_model
            self._client.is_embeddinng_model[model_name] = embedding_model
        if not self._client.is_embeddinng_model[model_name]:
            raise Exception(f"\033[95m {model_name} is not an embedding model. Please provide an embedding model name.\033[0m")
        res = embedding_request(inputs, model_name, self._client.model_urls[model_name], self._client.tromero_key)
        formatted_res = EmbeddingResponse(res)
        return formatted_res


class MockChat(Chat):
    def __init__(self, client):
        super().__init__(client)

    @cached_property
    def completions(self) -> Completions:
        return MockCompletions(self._client)


class Tromero(OpenAI):
    chat: MockChat
    embeddings: TromeroEmbeddings
    def __init__(self, tromero_key, api_key="", save_data_default=False, location_preference=None):
        super().__init__(api_key=api_key)
        self.current_prompt = []
        self.model_urls = {}
        self.is_base_model = {}
        self.is_embeddinng_model = {}
        self.tromero_key = tromero_key
        self.chat = MockChat(self)
        self.save_data_default = save_data_default
        self.tromero_models = TromeroModels(tromero_key)
        self.fine_tuning_jobs = FineTuningJob(tromero_key)
        self.data = TromeroData(tromero_key)
        self.datasets = Datasets(tromero_key)
        self.location_preference = location_preference
        self.openai_models = []
        self.embeddings = TromeroEmbeddings(self)
