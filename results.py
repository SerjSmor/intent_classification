import argparse

import pandas as pd
from sklearn.metrics import classification_report

from consts import DEFAULT_PREDICTION_CSV


def extract_class_name(prediction_text: str) -> str:
    prediction_text = prediction_text.lower()
    print(prediction_text)
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

def results(csv_path):
    predictions_df = pd.read_csv(csv_path)
    print(predictions_df["prediction"].value_counts())
    predictions_df["class_name"] = predictions_df["class_name"].apply(str.lower)
    predictions_df["generator_label_prediction"] = predictions_df["prediction"].apply(extract_class_name)
    # count how many predictions are actually in the taxonomy 1X1
    predictions_df["is_generator_label_in_prompt_options"] = predictions_df.apply(
        lambda row: is_generator_label_in_prompt_options(row), axis=1)
    print(predictions_df["is_generator_label_in_prompt_options"].value_counts())
    missing_class_name = predictions_df[predictions_df['is_generator_label_in_prompt_options'] == False].shape[0]
    print(f"{missing_class_name / predictions_df.shape[0]}")
    good_predictions_df = predictions_df[predictions_df["is_generator_label_in_prompt_options"] == True]

    print(classification_report(good_predictions_df["class_name"], good_predictions_df["generator_label_prediction"]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Results")
    parser.add_argument("-p", "--csv-path", default=DEFAULT_PREDICTION_CSV)
    args = parser.parse_args()

    results(args.csv_path)
