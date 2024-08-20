import requests
import json
from .constants import DATA_URL, BASE_URL

def genric_request(method, path, data, tromero_key):
    headers = {'Content-Type': 'application/json',
               'X-API-KEY': tromero_key}
    if method == "GET":
        response = requests.get(f"{BASE_URL}{path}", headers=headers)
    else :
        response = requests.request(method, f"{BASE_URL}{path}", json=data, headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful


def get_signed_url(auth_token):
    headers = {
        'X-API-KEY': auth_token,
        'Content-Type': 'application/json'
    }
    response = requests.get(f"{BASE_URL}/generate_signed_url", headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()['signedUrl'], response.json()['filename']  # Return the JSON response if request was successful
    

def upload_file_to_url(signed_url, file_path):
    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        headers = {'Content-Type': 'application/octet-stream'}  # Adjust this based on the actual file type you are uploading.
        response = requests.put(signed_url, data=f, headers=headers)

    # Check if the upload was successful
    if response.status_code != 200:
        raise Exception(f"An error occurred in upload: {response.text}")
    

def save_logs(custom_logs_filename, save_logs_with_tags, tromero_key, make_synthetic_version=False):
    headers = {'Content-Type': 'application/json'}
    data = {
        "custom_logs_filename": custom_logs_filename,
        "save_logs_with_tags": save_logs_with_tags,
        "make_synthetic_version": bool(make_synthetic_version)
    }
    headers['X-API-KEY'] = tromero_key
    response = requests.post(f"{BASE_URL}/custom_log_upload", json=data, headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful
    
def create_fine_tuning_job(data, tromero_key):
    headers = {'Content-Type': 'application/json'}
    headers['X-API-KEY'] = tromero_key
    
    response = requests.post(f"{BASE_URL}/training-pod", json=data, headers=headers)
    return response.json()  # Return the JSON response if request was successful

def get_models(tromero_key):
    headers = {
        'X-API-KEY': tromero_key,
        'Content-Type': 'application/json'
    }
    response = requests.get(f"{BASE_URL}/models?show_full=true", headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful

def get_model_training_info(model_name, tromero_key):
    headers = {
        'X-API-KEY': tromero_key,
        'Content-Type': 'application/json'
    }
    response = requests.get(f"{BASE_URL}/named-training-info-log/{model_name}", headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful

def deploy_model_request(model_name, tromero_key):
    headers = {
        'X-API-KEY': tromero_key,
        'Content-Type': 'application/json'
    }

    data = {
        "model_name": model_name
    }
    response = requests.post(f"{BASE_URL}/deploy_model", json=data, headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful

def get_model_request(model_name, tromero_key):
    headers = {
        'X-API-KEY': tromero_key,
        'Content-Type': 'application/json'
    }
    response = requests.get(f"{BASE_URL}/models/by_name/{model_name}", headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful
    
def undeploy_model_request(model_name, tromero_key):
    headers = {
        'X-API-KEY': tromero_key,
        'Content-Type': 'application/json'
    }

    data = {
        "model_name": model_name
    }
    response = requests.post(f"{BASE_URL}/undeploy_model", json=data, headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful

def get_tags(tromero_key):
    headers = {
        'X-API-KEY': tromero_key,
    }
    response = requests.get(f"{BASE_URL}/tags", headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful

def create_dataset(name, description, tags, tromero_key):
    data = {
        "name": name,
        "description": description,
        "tags": tags
    }
    headers = {'Content-Type': 'application/json'}
    headers['X-API-KEY'] = tromero_key
    response = requests.post(f"{BASE_URL}/datasets", json=data, headers=headers)
    response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful

def model_evaluation_request(model_name, tromero_key):
    path = f"/evaluate/named/{model_name}"
    return genric_request("GET", path, {}, tromero_key)
    