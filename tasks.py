from datetime import datetime
import json
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from invoke import task, Context
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

# app
from dataset import load_main_dataset
from preprocess import preprocess_simple_dataset, preprocess
from app.model import IntentClassifier
from app.utils import get_model_suffix
from app.atis.utils import ATIS_INTENT_MAPPING as intent_mapping


FLAN_T_BASE = "google/flan-t5-base"

@dataclass
class Summary:
    batch_size: int

@task
def preprocess_dataset(c):
    c.run("python preprocess.py")

@task
def train(c, is_preprocess=False, model_name=FLAN_T_BASE, epochs=6, batch_size=8, per_dataset=False, name="", test_atis=False):
    if is_preprocess:
        c.run("python preprocess.py")
    print(f"(train) experiment name: {name}")
    test_atis_str = "--test-atis" if test_atis else ""
    commands = f"--model-name {model_name} -e {epochs} -bs {batch_size} --name {name} {test_atis_str}"
    print(commands)
    c.run(f"python t5_generator_trainer.py {commands}")
    # model_suffix = model_name.split("/")[1]
    # c.run(f"python predict.py --model-name models/{model_suffix} -p data/test.csv")
    # per_dataset_str = "" if not per_dataset else " -pd "
    # c.run(f"python results.py -p results/predictions_test.csv {per_dataset_str}")


@task
def _train_atis_zero_shot(c, model_name=FLAN_T_BASE, epochs=15, batch_size=8, dataset_to_remove="Atis Arilines", name=""):
    dataset = load_main_dataset()
    # remove atis
    filtered_dataset = [var for var in dataset if var["Company Name"] != dataset_to_remove]

    preprocess(filtered_dataset)
    print(f"(_train_atis)experiment name: {name}")
    train(c, is_preprocess=False, model_name=model_name, epochs=epochs, batch_size=batch_size, name=name, test_atis=True)


@task
def test_atis(c, model_name=FLAN_T_BASE):
    # atis test set
    model_suffix = get_model_suffix(model_name)
    full_model_path = f"models/{model_suffix}"
    model = IntentClassifier(model_name=full_model_path)

    dataset = load_dataset("tuetschek/atis")
    intents = set([row["intent"] for row in dataset["test"]])

    prompt_options = "OPTIONS\n"
    index = 1

    for intent in intents:
        if intent not in intent_mapping:
            continue

        mapping = intent_mapping[intent]
        prompt_options += f" {index}. {mapping} \n"
        index += 1
    prompt_options = f'''{prompt_options}'''

    results = []
    company_name = "Atis Airlines"
    company_specific = "Airlines flights, meals, seats customer requests"

    for row in tqdm(dataset["test"]):
        intent = row["intent"]
        if intent not in intent_mapping:
            continue

        prediction = model.predict(row["text"], prompt_options, company_name, company_specific)
        # keywords = model.raw_predict(f"All of the verbs: {row['text']}")
        results.append({"prediction": prediction, "y": intent_mapping[intent], "text": row["text"]})

    from sklearn.metrics import classification_report
    y = [r["y"] for r in results]
    predictions = [r["prediction"].replace("Class name: ", "") for r in results]
    classification_report_df = pd.DataFrame(classification_report(y, predictions, output_dict=True)).T
    print(classification_report_df)

    classification_report_df.to_csv("results/classification_report.csv", index=False)


@task
def atis_pipeline(c, name="", model_name=FLAN_T_BASE, epochs=15, batch_size=8):
    if len(name) == 0:
        model_suffix = get_model_suffix(model_name)
        date_time = datetime.now().strftime("%m%d%Y-%H:%M:%S")
        print("date and time:", date_time)
        name = f"{model_suffix}_e{epochs}_b{batch_size}_t{date_time}"
    print(f"experiment name: {name}")
    _train_atis_zero_shot(c, model_name=model_name, epochs=epochs, batch_size=batch_size, name=name)

    pipeline_args = {"model name": model_name, "epochs": epochs, "batch size": batch_size, "name": name}
    with open("data/train_args.txt", "w") as f:
        json.dump(pipeline_args, f)

    test_atis(c, model_name)
    archive(c, name)

@task
def archive(c, name):
    experiment_path = f"archived_experiments/{name}"
    c.run(f"mkdir {experiment_path}")
    c.run(f"cp -R data {experiment_path}/")
    c.run(f"cp -R models {experiment_path}/")
    c.run(f"cp -R results {experiment_path}/")


@task
def train_small(c):
    # train(c, model_name="google/flan-t5-small")
    # train(c, model_name="google/flan-t5-small", epochs=20)
    train(c, model_name="flan-t5-small", epochs=20)

@task
def simple_dataset_base(c):
    preprocess_simple_dataset()
    c.run("python predict.py --model-name models/flan-t5-base -p data/simple_dataset.csv")
    c.run("python results.py -p data/predictions_simple_dataset.csv")

@task
def simple_dataset_small(c):
    preprocess_simple_dataset()
    c.run("python predict.py --model-name models/flan-t5-small -p data/simple_dataset.csv")
    c.run("python results.py -p data/predictions_simple_dataset.csv")


@task
def predict_test_hf_model(c):
    c.run("python preprocess.py")
    c.run("python predict.py --model-name serj/intent-classifier -p data/test.csv")
    c.run("python results.py -p data/predictions_test.csv -pd")

@task
def upload_to_hf(c):
    model = T5ForConditionalGeneration.from_pretrained("models/flan-t5-base")
    # push to the hub
    model.push_to_hub("intent-classifier", token="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", commit_message="Add clinc_oos dataset sub samples to training dataset")

@task
def upload_to_hf_small(c):
    model = T5ForConditionalGeneration.from_pretrained("models/flan-t5-small")
    # push to the hub
    model.push_to_hub("intent-classifier-flan-t5-small", token="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",
                      commit_message="Add Atis dataset sub samples to training dataset")





if __name__ == '__main__':
    _train_atis_zero_shot(Context())