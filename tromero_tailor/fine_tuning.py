from .fine_tuning_requests import (get_signed_url, upload_file_to_url, save_logs, create_fine_tuning_job,
                                   get_model_training_info, get_models, deploy_model_request, get_model_request, undeploy_model_request, 
                                   get_tags, create_dataset, model_evaluation_request)
from .tromero_utils import tags_to_string, validate_file_content
from .fine_tuning_models import Model, TrainingMetrics, Dataset
import uuid
import json

def set_raw(val, default):
    return val if val is not None else default
    
class Datasets:
    def __init__(self, tromero_key, raw_default=False):
        self.tromero_key = tromero_key
        self.raw_default = raw_default

    def create_from_file(self, file_path, name, description, tags):
        id_tag = f"dataset_tag_{str(uuid.uuid4())}"
        if type(tags) == str:
            tags = [tags]
        tags = list(tags)
        tags.append(id_tag)
        if not validate_file_content(file_path):
            return
        signed_url, filename = get_signed_url(self.tromero_key)
        upload_file_to_url(signed_url, file_path)
        save_logs(filename, tags, self.tromero_key)
        print(f"File uploaded successfully! Tags: {tags}")
        create_dataset(name, description, [id_tag], self.tromero_key)
        return True
    
    def create_from_tags(self, name, description, tags):
        create_dataset(name, description, tags, self.tromero_key)
        return True
    
    def list(self, raw=None):
        raw = set_raw(raw, self.raw_default)
        response = get_tags(self.tromero_key)
        datasets = response.get("datasets", [])
        if raw:
            return datasets
        return [Dataset(**dataset) for dataset in datasets]
    
class FineTuningJob:
    def __init__(self, tromero_key, raw_default=False):
        self.tromero_key = tromero_key
        self.raw_default = raw_default

    # Valid parameters for fine tuning
    # epoch, learning_rate, batch_size, tags, custom_logs_filename, save_logs_with_tags, custom_dataset, skip_logs_with_errors
    def create(self, model_name, base_model, parameters=None):
        data = {"model_name": model_name, "base_model": base_model}   
        if parameters:
            if type(parameters) == str:
                parameters = json.loads(parameters)
                print(parameters)
            data.update(parameters)
        response = create_fine_tuning_job(data, self.tromero_key)
        return response
    
    def get_metrics(self, model_name, raw=None):
        raw = set_raw(raw, self.raw_default)
        response = get_model_training_info(model_name, self.tromero_key)
        metrics = response.get("metrics", {})
        if not metrics:
            print("Metrics are not available for this model yet.")
        if raw:
            return metrics
        ret = TrainingMetrics(**metrics)
        return ret

    
class TromeroModels:
    def __init__(self, tromero_key, raw_default=False):
        self.tromero_key = tromero_key
        self.raw_default = raw_default

    def list(self, raw=None):
        """Returns a list of the users fine tuned models"""
        raw = set_raw(raw, self.raw_default)
        response = get_models(self.tromero_key)
        if raw:
            return response["message"]
        model_data = [Model(**model) for model in response["message"]]
        return model_data
    
    def deploy(self, model_name):
        """Deploys a fine tuned model. Model must be undeoloyed to work. Takes model_name"""
        response = deploy_model_request(model_name, self.tromero_key)
        return response
    
    def get_info(self, model_name, raw=None):
        """Returns information about the model. Takes model_name"""
        raw = set_raw(raw, self.raw_default)
        response = get_model_request(model_name, self.tromero_key)
        if raw:
            return response
        model_data = Model(**response["message"])
        return model_data
    
    def undeploy(self, model_name):
        """Undeploys a fine tuned model. Model must be deployed to work. Takes model_name"""
        response = undeploy_model_request(model_name, self.tromero_key)
        return response

    
class TromeroData:
    def __init__(self, tromero_key):
        self.tromero_key = tromero_key

    def upload(self, file_path, tags, make_synthetic_version=False):
        if type(tags) == str:
            tags = [tags]
        tags = list(tags)
        if not validate_file_content(file_path):
            return
        signed_url, filename = get_signed_url(self.tromero_key)
        upload_file_to_url(signed_url, file_path)
        save_logs(filename, tags, self.tromero_key, make_synthetic_version)
        print(f"File uploaded successfully! Tags: {tags}")
        return True
    
    def get_tags(self):
        response = get_tags(self.tromero_key)
        return response["message"]
    

# class Evaluations:
#     def __init__(self, tromero_key, raw_default=False):
#         self.tromero_key = tromero_key
#         self.raw_default = raw_default

#     def model_evaluation(self, model_name, raw=None):
#         raw = set_raw(raw, self.raw_default)
#         response = model_evaluation_request(model_name, self.tromero_key)
#         return response
#         if raw:
#             return response
#         evaluation = response.get("evaluation", {})
#         return evaluation

        