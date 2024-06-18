import os

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import classification_report
from tqdm import tqdm

from app.atis.utils import create_prompt_options, ATIS_INTENT_MAPPING as intent_mapping
from consts import GENERATOR_LABELS
from dataset import load_main_dataset
from preprocess import preprocess

from dotenv import load_dotenv
load_dotenv()

class OpenAIClassifier:

    def __init__(self, prompt_options: str, model_name: str):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.prompt_options = prompt_options
        self.model_name = model_name


    def predict(self, txt: str) -> str:
        prompt = "You are an expert in customer service domain. You need to classify a customer message into one of " \
                 f"the following classes: {self.prompt_options} % Customer message: {txt} % Please return json object with the following structure: {'{class_name: ''}'} class name should not contain a number. "

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        response_dict = eval(completion.choices[0].message.content)
        return response_dict["class_name"]


def clean_prediction(prediction):
    if type(prediction) is int:
        return str(int)
    elif type(prediction) is str:
        translation_table = str.maketrans('', '', '0123456789')
        no_numbers_str = prediction.translate(translation_table)
        no_numbers_str.replace(".", " ")
        return no_numbers_str
    else:
        return ""


def predict_atis(model_name):
    dataset = load_dataset("tuetschek/atis")
    intents = set([row["intent"] for row in dataset["test"]])

    prompt_options = create_prompt_options(intent_mapping, intents)

    model = OpenAIClassifier(prompt_options, model_name)
    results = []
    # dataset["test"] = dataset["test"].select(range(10))
    exceptions = 0
    for row in tqdm(dataset["test"]):
        intent = row["intent"]
        if intent not in intent_mapping:
            continue

        try:
            prediction = model.predict(row["text"])
            results.append(
                {"prediction": prediction, "y": intent_mapping[intent], "text": row["text"], "model_name": model_name})
        except Exception as e:
            exceptions += 1
            print(e)

    print(f"total exceptions: {exceptions}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"data/{model_name}_atis_test_set.csv", index=False)

    y = [row["y"].lower().strip() for row in results]
    predictions = [clean_prediction(row["prediction"]).lower().strip() for row in results]

    print(classification_report(y, predictions))

    classification_report_df = pd.DataFrame(classification_report(y, predictions, output_dict=True)).T

    classification_report_df["id"] = classification_report_df.index.tolist()

    # use several prompts - run predictions
    # save predictions
    # run results (same results?)

def predict_synthetic_dataset(dataset_name: str, model_name: str):
    dataset = load_main_dataset()
    # remove atis
    filtered_dataset = [var for var in dataset if var["Company Name"] == dataset_name]

    _, test_df, _ = preprocess(filtered_dataset)
    prompt_options = test_df["prompt_options"].iloc[0]
    model = OpenAIClassifier(prompt_options, model_name)
    results = []
    exceptions = 0
    for index, row in tqdm(test_df.iterrows()):
        try:
            prediction = model.predict(row["sample_text"])
            results.append(
                {"prediction": prediction, "y": row[GENERATOR_LABELS], "text": row["sample_text"], "model_name": model_name})
        except Exception as e:
            exceptions += 1
            print(e)

    print(f"total exceptions: {exceptions}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"data/{model_name}_{dataset_name}_test_set.csv", index=False)

    y = [row["y"].lower().strip() for row in results]
    predictions = [clean_prediction(row["prediction"]).lower().strip() for row in results]

    print(classification_report(y, predictions))

    classification_report_df = pd.DataFrame(classification_report(y, predictions, output_dict=True)).T
    classification_report_df.to_csv(f"data/{model_name}_{dataset_name}_classification_report.csv")

    classification_report_df["id"] = classification_report_df.index.tolist()

if __name__ == '__main__':
    # load atis dataset
    models = ["gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09"]
    datasets = ["Online_Banking", "Pizza_Mia", "Clinc_oos_41_classes"]
    # predict_atis(model_name="gpt-3.5-turbo-0125")
    # predict_atis(model_name="gpt-4o-2024-05-13")
    # predict_atis(model_name="gpt-4-turbo-2024-04-09")
    for model in models:
        for dataset in datasets:
            predict_synthetic_dataset(dataset, model_name=model)