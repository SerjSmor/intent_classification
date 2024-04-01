import json


def load_main_dataset():
    return _load_dataset("data/dataset.json")

def load_simple_dataset():
    return _load_dataset("data/simple_dataset.json")

def _load_dataset(path):
    with open(path) as f:
        dataset = json.load(f)

    return dataset