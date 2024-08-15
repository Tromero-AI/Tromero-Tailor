import json
from openai import OpenAI
from openai.resources import Chat
from openai.resources.chat.completions import (
    Completions
)
from openai._compat import cached_property
import datetime
from tromero_tailor.tromero_requests import post_data, tromero_model_create, get_model_url, tromero_model_create_stream
from tromero_tailor.tromero_utils import mock_openai_format, tags_to_string
import warnings
import threading
from tromero_tailor.fine_tuning import TromeroModels, TromeroData, FineTuningJob, Datasets
from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError
# import check_schema
from jsonschema import Validator, FormatChecker
from jsonschema import Draft7Validator


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
    
    def _save_data(self, data):
        if self._client.save_data:
            threading.Thread(target=post_data, args=(data, self._client.tromero_key)).start()

    def validate_schema(self, schema):
        try:
        # Validate schema against the JSON Schema Draft 7
            Draft7Validator.check_schema(schema)

            # Additional common checks
            if "properties" in schema:
                for prop, details in schema["properties"].items():
                    if "type" in details:
                        valid_types = {"string", "number", "integer", "boolean", "array", "object"}
                        if details["type"] not in valid_types:
                            raise ValueError(f"Invalid type specified: {details['type']} in property '{prop}'")
                    else:
                        raise ValueError(f"No type specified for property '{prop}'")
            else:
                raise ValueError("No properties defined in the schema.")

            print("Detailed validation passed.")
        except (SchemaError, ValueError) as e:
            raise SchemaError(f"Schema validation failed: {e}")


    def _format_kwargs(self, kwargs):
        keys_to_keep = [
            "best_of", "decoder_input_details", "details", "do_sample", 
            "max_new_tokens", "ignore_eos_token", "repetition_penalty", 
            "return_full_outcome", "seed", "stop", "temperature", "top_k", 
            "top_p", "truncate", "typical_p", "watermark", 
            "adapter_id", "adapter_source", "merged_adapters", "response_format", 
            "make_synthetic_version", "guided_schema", "guided_regex", "tools"
        ]
        invalid_key_found = False
        parameters = {}
        for key in kwargs:
            if key not in keys_to_keep and key not in ["tags", "model", "messages", "use_fallback", "fallback_model", "stream"]:
                warnings.warn(f"Warning: {key} is not a valid parameter for the model. This parameter will be ignored.")
                invalid_key_found = True
            elif key in keys_to_keep:
                parameters[key] = kwargs[key]

        if invalid_key_found:
            print("the following parameters are valid for the model: ", keys_to_keep)
        
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
    
    def _stream_response(self, response, init_save_data, fall_back_dict):
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
            if init_save_data != {}:
                init_save_data['messages'].append({"role": "assistant", "content": full_message})
                self._save_data(init_save_data)


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
        send_kwargs = {}
        use_fallback = kwargs.get('use_fallback', True)
        fallback_model = kwargs.get('fallback_model', '')
        
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['tags', 'use_fallback', 'fallback_model']}
        if self.check_model(kwargs['model']):
            res = Completions.create(self, *args, **openai_kwargs)  
            send_kwargs = openai_kwargs
        else:
            formatted_kwargs = self._format_kwargs(kwargs)
            send_kwargs = formatted_kwargs
            model_name = model
            if model_name not in self._client.model_urls:
                url, base_model = get_model_url(model_name, self._client.tromero_key)
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
                save_data = {"messages": formatted_messages + [formatted_choice['message']],
                                "model": model,
                                "kwargs": send_kwargs,
                                "creation_time": str(datetime.datetime.now().isoformat()),
                                "tags": tags_to_string(tags)
                                }
                # if hasattr(res, 'usage'):
                #     save_data['usage'] = res.usage.model_dump()
                self._save_data(save_data)
        elif stream:
            init_save_data = {"messages": formatted_messages,
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
            return self._stream_response(res, init_save_data, fall_back_dict)
        else:
            if use_fallback and fallback_model:
                print("Error in making request to model. Using fallback model.")
                kwargs['model'] = fallback_model
                kwargs['use_fallback'] = False
                return self.create(*args, **kwargs)

        return res


class MockChat(Chat):
    def __init__(self, client):
        super().__init__(client)

    @cached_property
    def completions(self) -> Completions:
        return MockCompletions(self._client)


class TailorAI(OpenAI):
    chat: MockChat
    def __init__(self, tromero_key, api_key="", save_data=False):
        super().__init__(api_key=api_key)
        self.current_prompt = []
        self.model_urls = {}
        self.is_base_model = {}
        self.tromero_key = tromero_key
        self.chat = MockChat(self)
        self.save_data = save_data
        self.tromero_models = TromeroModels(tromero_key)
        self.fine_tuning_jobs = FineTuningJob(tromero_key)
        self.data = TromeroData(tromero_key)
        self.datasets = Datasets(tromero_key)
