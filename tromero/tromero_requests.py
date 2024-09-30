import requests
import json
from .tromero_utils import mock_openai_format_stream
from .constants import DATA_URL, BASE_URL
import traceback

class TromeroError(Exception):
    def __init__(self, message):
        super().__init__(message)

def raise_for_status(response):
    # if status code does not start with 2, raise an error
    if not str(response.status_code).startswith('2'):
        json_response = response.json()
        message = json_response.get('message', json_response.get('error', 'An error occurred'))
        raise TromeroError(f"\033[95m{message}\033[0m")


def post_data(data, auth_token):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(DATA_URL, json=data, headers=headers)
        raise_for_status(response)  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except TromeroError as e:
        raise e
    except Exception as e:
        raise TromeroError(f'An error occurred: {e}')
    
def tromero_model_create(model, model_url, messages, tromero_key, parameters={}):
    try:
        headers = {'Content-Type': 'application/json'}
        data = {
            "adapter_name": model,
            "messages": messages,
            "parameters": parameters
        }
        headers['X-API-KEY'] = tromero_key
        response = requests.post(f"{model_url}/generate", json=data, headers=headers)
        raise_for_status(response)  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except TromeroError as e:
        raise e
    except Exception as e:
        raise TromeroError(f'An error occurred: {e}')
    

def get_model_url(model_name, auth_token, location_preference):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    if location_preference:
        url = f"{BASE_URL}/model/{model_name}/url?location_preference={location_preference.lower()}" 
    else:
        url = f"{BASE_URL}/model/{model_name}/url"
    try:
        response = requests.get(url, headers=headers)
        raise_for_status(response)  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()['url'], response.json().get('base_model', False), response.json().get('embedding_model', False)  
    except TromeroError as e:
        raise e
    except Exception as e:
        raise TromeroError(f'An error occurred: {e}')
    
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
    except TromeroError as e:
        raise e
    except Exception as e:
        raise TromeroError(f'An error occurred: {e}')

def embedding_request(inputs, model, model_url, tromero_key):
    headers = {'Content-Type': 'application/json'}
    data = {
        "inputs": inputs,
        "model": model
    }
    headers['X-API-KEY'] = tromero_key
    try:
        response = requests.post(model_url + "/embed", json=data, headers=headers)
        raise_for_status(response)  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except TromeroError as e:
        raise e
    except Exception as e:
        raise TromeroError(f'An error occurred: {e}')




    

