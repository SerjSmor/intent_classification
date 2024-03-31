import json


def load_dataset():
    with open("data/dataset.json") as f:
        dataset = json.load(f)
        print(dataset)
        print(f"number of examples: {len(dataset[0]['Examples'])}")

    return dataset