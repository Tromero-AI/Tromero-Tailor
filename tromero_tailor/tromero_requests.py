import requests

data_url = "https://api.example.com/data"
models_url = "http://87.120.209.240:5000/generate"

def post_data(data, auth_token):
    print(f"Posting data to {data_url}")
    print(f"Data: {data}")
    print(f"Auth Token: {auth_token}")
    headers = {
        'Authorization': f'Bearer {auth_token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(data_url, json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except Exception as e:
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}
    
def tromero_model_create(model, messages, tromero_key):
    headers = {
        'Authorization': f'Bearer {tromero_key}',
        'Content-Type': 'application/json'
    }

    data = {
        "adapter_name": model,
        "inputs": messages,
    }

    try:
        response = requests.post(models_url, json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except Exception as e:
        return {'error': f'An error occurred: {e}', 'status_code': response.status_code if 'response' in locals() else 'N/A'}




    

