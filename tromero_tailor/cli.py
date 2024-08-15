import fire

from tromero_tailor.fine_tuning import TromeroModels, TromeroData, FineTuningJob, Datasets
import os 


class TromeroCli():
    def __init__(self):
        tromero_key=os.getenv("TROMERO_API_KEY")
        self.models = TromeroModels(tromero_key, raw_default=True)
        self.fine_tuning_jobs = FineTuningJob(tromero_key, raw_default=True)
        self.data = TromeroData(tromero_key)
        self.datasets = Datasets(tromero_key, raw_default=True)

def main():
    fire.Fire(TromeroCli)
