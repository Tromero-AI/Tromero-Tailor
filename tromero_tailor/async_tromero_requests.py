import aiohttp
import json
import re
from .tromero_utils import mock_openai_format_stream

data_url = "https://midyear-grid-402910.lm.r.appspot.com/tailor/v1/data"
base_url = "https://midyear-grid-402910.lm.r.appspot.com/tailor/v1"

async def post_data(data, auth_token):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(data_url, json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            return {'error': f'An error occurred: {e}', 'status_code': response.status if 'response' in locals() else 'N/A'}

async def tromero_model_create(model, model_url, messages, tromero_key, parameters={}):
    headers = {'Content-Type': 'application/json'}
    data = {
        "adapter_name": model,
        "messages": messages,
        "parameters": parameters
    }
    headers['X-API-KEY'] = tromero_key
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{model_url}/generate", json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            return {'error': f'An error occurred: {e}', 'status_code': response.status if 'response' in locals() else 'N/A'}

async def get_model_url(model_name, auth_token):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/model/{model_name}/url", headers=headers) as response:
                response.raise_for_status()
                json_response = await response.json()
                return json_response['url'], json_response.get('base_model', False)
        except Exception as e:
            print(f"error: {e}")
            return {'error': f'An error occurred: {e}', 'status_code': response.status if 'response' in locals() else 'N/A'}

class StreamResponse:
    def __init__(self, response):
        self.response = response

    async def __aiter__(self):
        try:
            last_chunk = None
            async for chunk in self.response.content.iter_chunked(10000000):
                chunk = chunk.decode('utf-8')
                json_str = chunk[5:]
                last_chunk = json_str
                pattern = r'\"token\":({.*?})'
                match = re.search(pattern, json_str)
                if match:
                    json_str = match.group(1)
                else:
                    break
                chunk_dict = json.loads(json_str)
                formatted_chunk = mock_openai_format_stream(chunk_dict['text'])
                yield formatted_chunk
        except Exception as e:
            print(f"Error: {e}")
            return

async def tromero_model_create_stream(model, model_url, messages, tromero_key, parameters={}):
    headers = {'Content-Type': 'application/json'}
    data = {
        "adapter_name": model,
        "messages": messages,
        "parameters": parameters
    }
    headers['X-API-KEY'] = tromero_key
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.post(model_url + "/generate_stream", json=data, headers=headers)
            return StreamResponse(response), None
        except Exception as e:
            return None, {'error': f'An error occurred: {e}', 'status_code': response.status if 'response' in locals() else 'N/A'}
