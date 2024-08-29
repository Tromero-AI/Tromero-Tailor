import json

class ModelConfig:
    def __init__(self, base_model, batch_size=None, epoch=None, learning_rate=None, custom_dataset=None, custom_logs_filename=None, 
                 number_of_bad_logs=None, save_logs_with_tags=None, tags=None, external_service_id=None, pod_id=None):
        self.base_model = base_model
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.custom_dataset = custom_dataset
        self.custom_logs_filename = custom_logs_filename
        self.number_of_bad_logs = number_of_bad_logs
        self.save_logs_with_tags = save_logs_with_tags
        self.tags = tags
        self.external_service_id = external_service_id
        self.pod_id = pod_id

class UsageData:
    def __init__(self, date, tokens):
        self.date = date
        self.tokens = tokens


class BaseModelData:
    def __init__(self, available_for_finetuning, available_for_inference, default_batch_size, default_lr, display_name, hf_repo, id, model_name,
                 model_size, supported_context_len, training_time_per_log, training_time_y_intercept):
        self.available_for_finetuning = available_for_finetuning
        self.available_for_inference = available_for_inference
        self.default_batch_size = default_batch_size
        self.default_lr = default_lr
        self.display_name = display_name
        self.hf_repo = hf_repo
        self.id = id
        self.model_name = model_name
        self.model_size = model_size
        self.supported_context_len = supported_context_len
        self.training_time_per_log = training_time_per_log
        self.training_time_y_intercept = training_time_y_intercept

class EvaluationState:
    def __init__(self, status):
        self.status = status


class ModelEvaluationState:
    def __init__(self, mix_eval, needlehaystack):
        self.mix_eval = EvaluationState(**mix_eval)
        self.needlehaystack = EvaluationState(**needlehaystack)


class Model:
    def __init__(self, model_id, model_name, state, model_config, cost_per_1000_tokens=0, created_at=None, created_at_unix=None, 
                 last_deployed_on=None, last_deployed_on_unix=None, last_used=None, last_used_unix=None, training_ended_at=None, 
                 training_ended_at_unix=None, updated_at=None, user_id=None, model_evaluation=None, self_hosted=False, server_id=None,
                 usage_data=[], base_model_data=None, base_model_id=None, model_evaluation_state=None):
        self.model_id = model_id
        self.model_name = model_name
        self.state = state
        self.cost_per_1000_tokens = cost_per_1000_tokens
        self.created_at = created_at
        self.created_at_unix = created_at_unix
        self.last_deployed_on = last_deployed_on
        self.last_deployed_on_unix = last_deployed_on_unix
        self.last_used = last_used
        self.last_used_unix = last_used_unix
        self.training_ended_at = training_ended_at
        self.training_ended_at_unix = training_ended_at_unix
        self.updated_at = updated_at
        self.user_id = user_id
        self.model_evaluation = model_evaluation  
        self.self_hosted = self_hosted
        self.server_id = server_id
        self.usage = [UsageData(**data) for data in usage_data]
        self.model_config = ModelConfig(**model_config)
        self.base_model_data = BaseModelData(**base_model_data)
        self.base_model_id = base_model_id
        self.model_evaluation_state = ModelEvaluationState(**model_evaluation_state)

class TrainingMetrics:
    def __init__(self, eval_loss, loss, eval_perplexity, perplexity):
        self.eval_loss = eval_loss
        self.loss = loss
        self.eval_perplexity = eval_perplexity
        self.perplexity = perplexity


class Filter:
    def __init__(self, from_date, models, tags, to_date):
        self.from_date = from_date
        self.models = models
        self.tags = tags
        self.to_date = to_date

class Dataset:
    def __init__(self, id, name, description, filters, user_id, created_at=None, updated_at=None):
        self.dataset_id = id
        self.name = name
        self.description = description
        self.filters = Filter(**filters)
        self.user_id = user_id
        self.created_at = created_at
        self.updated_at = updated_at

# {
#   "evaluation": {
#     "mix_eval": {
#       "global_status": "partially_complete",
#       "ongoing_tests": [
#         "GSM8K",
#         "AGIEVAL",
#         "TRIVIAQA",
#         "HELLASWAG",
#         "BOOLQ",
#         "OPENBOOKQA",
#         "COMMONSENSEQA"
#       ],
#       "results": {
#         "ARC": 0.859,
#         "BBH": 0.755,
#         "DROP": 0.746,
#         "GPQA": 0.5,
#         "MATH": 0.57,
#         "MBPP": 0.0,
#         "MMLU": 0.662,
#         "PIQA": 0.823,
#         "SIQA": 0.684,
#         "overall score (final score)": 0.6221111111111111
#       },
#       "status": {
#         "AGIEVAL": "started",
#         "ARC": "complete",
#         "BBH": "complete",
#         "BOOLQ": "started",
#         "COMMONSENSEQA": "started",
#         "DROP": "complete",
#         "GPQA": "complete",
#         "GSM8K": "started",
#         "HELLASWAG": "started",
#         "MATH": "complete",
#         "MBPP": "complete",
#         "MMLU": "complete",
#         "OPENBOOKQA": "started",
#         "PIQA": "complete",
#         "SIQA": "complete",
#         "TRIVIAQA": "started"
#       }
#     },
#     "needlehaystack": {
#       "global_status": "complete",
#       "results": {
#         "scores": [
#           [
#             10,
#             10,
#             10,
#             10,
#             10
#           ],
#           [
#             10,
#             10,
#             10,
#             10,
#             10
#           ],
#           [
#             10,
#             10,
#             10,
#             10,
#             10
#           ],
#           [
#             10,
#             10,
#             10,
#             10,
#             10
#           ],
#           [
#             3,
#             1,
#             3,
#             7,
#             10
#           ],
#           [
#             10,
#             3,
#             1,
#             3,
#             10
#           ]
#         ],
#         "x_axis": [
#           1000,
#           2400,
#           3800,
#           5200,
#           6600,
#           8000
#         ],
#         "y_axis": [
#           0,
#           25,
#           50,
#           75,
#           100
#         ]
#       },
#       "status": "complete"
#     }
#   },
#   "status": "success"
# }


# class MixEval:
#     def __init__(self, global_status, ongoing_tests, results, status):
#         self.global_status = global_status
#         self.ongoing_tests = ongoing_tests
#         self.results = results
#         self.status = status

# class NeedleHaystack:   
#     def __init__(self, global_status, results, status):
#         self.global_status = global_status
#         self.results = results
#         self.status = status