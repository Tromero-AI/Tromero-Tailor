import requests
import json
from .tromero_utils import mock_openai_format_stream
from .constants import DATA_URL, BASE_URL


def post_data(data, auth_token):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(DATA_URL, json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except Exception as e:
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}
    

def tromero_model_create(model, model_url, messages, tromero_key, parameters={}):
    headers = {'Content-Type': 'application/json'}
    data = {
        "adapter_name": model,
        "messages": messages,
        "parameters": parameters
    }
    headers['X-API-KEY'] = tromero_key
    try:
        response = requests.post(f"{model_url}/generate", json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except Exception as e:
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}
    

def get_model_url(model_name, auth_token):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.get(f"{BASE_URL}/model/{model_name}/url", headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()['url'], response.json().get('base_model', False)  # Return the JSON response if request was successful
    except Exception as e:
        print(f"error: {e}")
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}
    
class StreamResponse:
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        try:
            last_chunk = None
            for chunk in self.response.iter_content(chunk_size=10000000):
                # chunk_dict = json.loads(chunk)
                chunk = chunk.decode('utf-8')
                json_str = chunk[5:]
                last_chunk = json_str
                chunk_dict = json.loads(json_str)
                formatted_chunk = mock_openai_format_stream(chunk_dict['token']['text'])
                yield formatted_chunk
        except Exception as e:
            print(f"Error: {e}")
            return
    
def tromero_model_create_stream(model, model_url, messages, tromero_key, parameters={}):
    headers = {'Content-Type': 'application/json'}
    data = {
        "adapter_name": model,
        "messages": messages,
        "parameters": parameters
    }
    headers['X-API-KEY'] = tromero_key
    try:
        response = requests.post(model_url + "/generate_stream", json=data, headers=headers, stream=True)
        return StreamResponse(response), None
    except Exception as e:
        return None, {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}



    

