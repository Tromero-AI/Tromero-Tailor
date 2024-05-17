import requests

data_url = "https://midyear-grid-402910.lm.r.appspot.com/tailor/v1/data"
models_url = "http://35.246.163.71:5000/generate"
self_hosted_models_url = "https://midyear-grid-402910.lm.r.appspot.com/tailor/v1/generate"

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
    

def tromero_model_create(model, messages, tromero_key, parameters={}, self_hosted=False):
    url = self_hosted_models_url if self_hosted else models_url
    headers = {'Content-Type': 'application/json'}
    data = {
        "adapter_name": model,
        "messages": messages,
        "parameters": parameters
    }
    headers['X-API-KEY'] = tromero_key
    if self_hosted:
        url = self_hosted_models_url 
    else:
        url = models_url
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except Exception as e:
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}




    

