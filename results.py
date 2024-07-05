import argparse

import evaluate
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

from app.model import IntentClassifier
from consts import DEFAULT_PREDICTION_CSV, TEST_CSV, BEST_ENTITY_EXTRACTION_MODEL_COMBINED, GENERATOR_TEXT_NO_COMPANY, \
    GENERATOR_LABELS, BLEU_PREDICTIONS_CSV

bleu = evaluate.load("sacrebleu")

def extract_class_name(prediction_text: str) -> str:
    if prediction_text == pd.isnull(prediction_text):
        return ""

    prediction_text = prediction_text.lower()
    class_name = prediction_text
    if ":" in prediction_text:
        class_name = prediction_text.split(":")[1]
        if ":" in prediction_text:
            class_name = prediction_text.split(":")[1]

    return class_name.strip()


def is_generator_label_in_prompt_options(row):
    class_name_prediction = row["generator_label_prediction"]
    prompt_options = row["prompt_options"]
    # go over for each option and check if class_name_prediction
    prompt_options = prompt_options.replace("OPTIONS:", "")
    lines = prompt_options.strip().split('\n')

    # Process each line to extract the option text
    options = [line.split('.', 1)[1].strip().lower() for line in lines if '.' in line]
    for option in options:
        if option == class_name_prediction:
            return True

    return False

def results(csv_path: str, per_dataset: bool):
    predictions_df = pd.read_csv(csv_path)
    if per_dataset:
        datasets = predictions_df["dataset_name"].unique().tolist()
        for dataset in datasets:
            df = predictions_df[predictions_df["dataset_name"] == dataset]
            calculate_classification_report(df)
    else:
        return calculate_classification_report(predictions_df)

def calculate_classification_report(df):
    print(df["prediction"].value_counts())
    df["class_name"] = df["class_name"].apply(str.lower)
    df["generator_label_prediction"] = df["prediction"].apply(extract_class_name)
    # count how many predictions are actually in the taxonomy 1X1
    # df["is_generator_label_in_prompt_options"] = df.apply(
    #     lambda row: is_generator_label_in_prompt_options(row), axis=1)
    # print(df["is_generator_label_in_prompt_options"].value_counts())
    # missing_class_name = df[df['is_generator_label_in_prompt_options'] == False].shape[0]
    # miss_percentage = missing_class_name / df.shape[0]
    # print(f"{miss_percentage:.3f}")
    # good_df = df[df["is_generator_label_in_prompt_options"] == True]
    miss_percentage = 0
    print(classification_report(df["class_name"], df["generator_label_prediction"]))
    classification_report_dict = \
        classification_report(df["class_name"], df["generator_label_prediction"], output_dict=True)

    return classification_report_dict, miss_percentage


def calculate_bleu_file(output_csv=TEST_CSV, model_path=BEST_ENTITY_EXTRACTION_MODEL_COMBINED,
                        text_column=GENERATOR_TEXT_NO_COMPANY, label_column=GENERATOR_LABELS, tasks=["entity_extraction"]):
    df_validation = pd.read_csv(output_csv)
    df_validation = df_validation[df_validation["task"].isin(tasks)]
    df_validation = df_validation[[text_column, label_column]]

    m = IntentClassifier(model_path)
    pred_results = []
    for index, row in df_validation.iterrows():
        decoded_pred = m.raw_predict(row[text_column])
        pred_results.append({"text": row[text_column], "pred": decoded_pred, "label": row[label_column]})

    result = bleu.compute(predictions=[v["pred"] for v in pred_results], references=[v["label"] for v in pred_results])
    df = pd.DataFrame(pred_results)
    df.to_csv(BLEU_PREDICTIONS_CSV, index=False)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Results")
    parser.add_argument("-p", "--csv-path", default=DEFAULT_PREDICTION_CSV)
    parser.add_argument("-pd", "--per-dataset", action="store_true")
    args = parser.parse_args()

    results(args.csv_path, args.per_dataset)
