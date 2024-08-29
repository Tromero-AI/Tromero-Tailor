import requests
import json
from .constants import DATA_URL, BASE_URL
from .tromero_requests import TromeroError, raise_for_status


def genric_request(method, path, data, tromero_key):
    try:
        headers = {'Content-Type': 'application/json',
                'X-API-KEY': tromero_key}
        if method == "GET":
            response = requests.get(f"{BASE_URL}{path}", headers=headers)
        else :
            response = requests.request(method, f"{BASE_URL}{path}", json=data, headers=headers)
        raise_for_status(response)  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()  # Return the JSON response if request was successful
    except TromeroError as e:
        raise e
    except Exception as e:
        raise TromeroError(f'An error occurred: {e}')
    
def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise TromeroError(f'An error occurred: {e}')
    return wrapper

@exception_handler
def get_signed_url(auth_token):
    json_response = genric_request(method="GET", path="/generate_signed_url", data={}, tromero_key=auth_token)
    return json_response['signedUrl'], json_response['filename']
    

def upload_file_to_url(signed_url, file_path):
    # Open the file in binary mode
    with open(file_path, 'rb') as f:
        headers = {'Content-Type': 'application/octet-stream'}  # Adjust this based on the actual file type you are uploading.
        response = requests.put(signed_url, data=f, headers=headers)

    # Check if the upload was successful
    if response.status_code != 200:
        raise Exception(f"An error occurred in upload: {response.text}")
    
@exception_handler
def save_logs(custom_logs_filename, save_logs_with_tags, tromero_key, make_synthetic_version=False):
    return genric_request(method="POST", path="/custom_log_upload", data={"custom_logs_filename": custom_logs_filename, 
                        "save_logs_with_tags": save_logs_with_tags, 
                        "make_synthetic_version": bool(make_synthetic_version)}, tromero_key=tromero_key)

    

@exception_handler
def create_fine_tuning_job(data, tromero_key):
    return genric_request("POST", "/training-pod", data, tromero_key)

def get_models(tromero_key):
    headers = {
        'X-API-KEY': tromero_key,
        'Content-Type': 'application/json'
    }
    response = requests.get(f"{BASE_URL}/models?show_full=true", headers=headers)
    raise_for_status(response)  # Raises HTTPError for bad responses (4XX, 5XX)
    return response.json()  # Return the JSON response if request was successful

@exception_handler
def get_model_training_info(model_name, tromero_key):
    return genric_request("GET", f"/named-training-info-log/{model_name}", {}, tromero_key)

@exception_handler
def deploy_model_request(model_name, tromero_key):
    return genric_request("POST", "/deploy_model", {"model_name": model_name}, tromero_key)

@exception_handler
def get_model_request(model_name, tromero_key):
    return genric_request("GET", f"/models/by_name/{model_name}", {}, tromero_key)
    
@exception_handler
def undeploy_model_request(model_name, tromero_key):
    return genric_request("POST", "/undeploy_model", {"model_name": model_name}, tromero_key)

@exception_handler
def get_tags(tromero_key):
    return genric_request("GET", "/tags", {}, tromero_key)

@exception_handler
def create_dataset(name, description, tags, tromero_key):
    return genric_request("POST", "/datasets", {"name": name, "description": description, "tags": tags}, tromero_key)

def model_evaluation_request(model_name, tromero_key):
    path = f"/evaluate/named/{model_name}"
    return genric_request("GET", path, {}, tromero_key)
    