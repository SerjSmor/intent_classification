import argparse

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report

from app.model import IntentClassifier
from app.utils import get_model_suffix, build_prompt, build_entity_extraction_prompt
from consts import ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV, ATIS_PREDICTIONS_CSV, FLAN_T5_SMALL, LOCAL_FLAN_T5_SMALL, \
    BEST_LONG_FORMAT_SINGLE_TASK_EXTRACTION_MODEL, BEST_SHORT_FORMAT_SINGLE_TASK_EXTRACTION_MODEL, ANOTHER_MODEL

ATIS_INTENT_MAPPING = {
    'abbreviation': "Abbreviation and Fare Code Meaning Inquiry",
    'aircraft': "Aircraft Type Inquiry",
    # 'aircraft+flight+flight_no': "Flight Details Inquiry",
    'airfare': "Airfare and Fares Questions",
    # 'airfare+flight_time': "Flight Cost and Duration Inquiry",
    'airline': "Airline Information Request",
    # 'airline+flight_no': "Airline and Flight Number Query",
    'airport': "Airport Information and Queries",
    'capacity': "Aircraft Seating Capacity Inquiry",
    'cheapest': "Cheapest Fare Inquiry",
    'city': "Airport Location Inquiry",
    'distance': "Airport Distance Inquiry",
    'flight': "Flight Booking Request",
    # 'flight+airfare': "Flight and Fare Inquiry",
    'flight_no': "Flight Number Inquiry",
    'flight_time': "Time Inquiry",
    'ground_fare': "Ground Transportation Cost Inquiry",
    'ground_service': "Ground Transportation Inquiry",
    'ground_service+ground_fare': "Airport Ground Transportation and Cost Query",
    'meal': "Inquiry about In-flight Meals",
    'quantity': "Flight Quantity Inquiry",
    'restriction': "Flight Restriction Inquiry"
}

def test_atis_dataset(full_model_path, intent_mapping=ATIS_INTENT_MAPPING, save_file=True, use_default_labels=False,
                      no_company_specific=True, no_number_prompt=False, extract_entities=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentClassifier(model_name=full_model_path, device=device)
    print(f"extract entities: {extract_entities}")
    dataset = load_dataset("tuetschek/atis")
    all_intents = [row["intent"] for row in dataset["test"]]
    intents = np.unique(all_intents)

    prompt_options = create_prompt_options(intent_mapping, intents, no_number_prompt, use_default_labels)
    results = []
    company_name = "Atis Airlines"
    company_specific = "Airlines flights, meals, seats customer requests"

    extraction_model = None
    if extract_entities:
        extraction_model = IntentClassifier(ANOTHER_MODEL)
    unique_texts = []
    for row in tqdm(dataset["test"]):
        intent = row["intent"]
        text = row["text"]
        original = ""
        if intent not in intent_mapping:
            continue
        if text in unique_texts:
            continue

        if extract_entities:
            original = text
            # text = model.raw_predict(build_entity_extraction_prompt(text))
            text = extraction_model.raw_predict(build_entity_extraction_prompt(text))
        input_text = build_prompt(text, prompt_options, company_name, company_specific, no_company_specific)
        prediction = model.raw_predict(input_text)
        prediction = prediction.replace("Class name: ", "")
        # keywords = model.raw_predict(f"All of the verbs: {row['text']}")
        y = intent_mapping[intent]
        if use_default_labels:
            y = row["intent"]
        result = {"prediction": prediction, "y": y, "text": text, "prompt": input_text,
                        "prompt_options": prompt_options}
        if len(original) > 0:
            result["original"] = original
        results.append(result)

        unique_texts.append(row["text"])

    df_results = pd.DataFrame(results)
    print(f"## total rows: {len(results)}, printing atis predictions distribution for model: {full_model_path}")
    print(df_results["prediction"].value_counts())

    if save_file:
        df_results.to_csv(ATIS_PREDICTIONS_CSV, index=False)
    df_errors = df_results[df_results["y"] != df_results["prediction"]]
    if save_file:
        df_errors.to_csv("results/atis_errors.csv", index=False)
    y = [r["y"] for r in results]
    predictions = [r["prediction"].replace("Class name: ", "") for r in results]
    print(predictions[:10])
    print(y[:10])
    print(classification_report(y, predictions))
    classification_report_df = pd.DataFrame(classification_report(y, predictions, output_dict=True)).T
    classification_report_df["id"] = classification_report_df.index.tolist()

    if save_file:
        classification_report_df.to_csv(ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV, index=False)

    return classification_report_df


def create_prompt_options(intent_mapping, intents, no_number_prompt=False, use_default_labels=False) -> str:
    print(f"create prompt options: intent_mapping: {intent_mapping}, no_number_prompt: {no_number_prompt},"
          f" use_default_labels: {use_default_labels}")
    prompt_options = "OPTIONS\n"
    index = 1
    for intent in intents:
        if intent not in intent_mapping:
            continue

        class_name = intent_mapping[intent]
        if use_default_labels:
            class_name = intent
        if no_number_prompt:
            prompt_options += f"# {class_name} \n"
        else:
            prompt_options += f" {index}. {class_name} \n"

        index += 1
    prompt_options = f'''{prompt_options}'''
    print(f"prompt options: {prompt_options}")
    return prompt_options

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict atis")
    parser.add_argument("-m", "--model-name", default=LOCAL_FLAN_T5_SMALL)
    parser.add_argument("--use-default-labels", action="store_true")
    parser.add_argument("--no-number-prompt", action="store_true")
    args = parser.parse_args()

    print(vars(args))

    test_atis_dataset(args.model_name, save_file=True, use_default_labels=args.use_default_labels, no_number_prompt=args.no_number_prompt,
                      no_company_specific=True)

