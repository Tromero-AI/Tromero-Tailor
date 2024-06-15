import requests
import json
from .tromero_utils import StreamResponse

data_url = "https://midyear-grid-402910.lm.r.appspot.com/tailor/v1/data"
models_url = "http://35.246.163.71:5000/generate"
self_hosted_models_url = "https://midyear-grid-402910.lm.r.appspot.com/tailor/v1/generate"
base_url = "https://midyear-grid-402910.lm.r.appspot.com/tailor/v1"

def post_data(data, auth_token):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(data_url, json=data, headers=headers)
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
        response = requests.get(f"{base_url}/model/{model_name}/url", headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()['url'], response.json().get('base_model', False)  # Return the JSON response if request was successful
    except Exception as e:
        print(f"error: {e}")
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}
    
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
        return StreamResponse(response)

        # Stream the response back to the client
        def generate():
            try:
                last_chunk = None
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        last_chunk = chunk
                        yield chunk
            except Exception as e:
                print("Error streaming response:", e, flush=True)
                raise e

        return generate()
    except Exception as e:
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}



    

