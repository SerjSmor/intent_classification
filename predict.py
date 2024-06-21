import argparse

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

from app.model import IntentClassifier
from app.utils import build_prompt
from consts import FLAN_T5_SMALL


# Load the pre-trained T5 model and tokenizer
# model_name = "google/flan-t5-base"
# model_name = "google/flan-t5-xl"
# model_name = "google/flan-t5-large"
# model_name = "models/flan_t5_base_generator"


def predict(csv_path, model_name=FLAN_T5_SMALL, no_company_specific=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intent_classifier = IntentClassifier(model_name=model_name, device=device)

    # TODO: change to batched predictions
    df = pd.read_csv(csv_path)
    for index, row in tqdm(df.iterrows()):
        # Concatenate the prompt with the example
        input_text = build_prompt(row["sample_text"], row["prompt_options"], row["Company Name"],
                                  no_company_specific=no_company_specific)
        decoded_output = intent_classifier.raw_predict(input_text)
        # Append the prediction to the list
        df.loc[df.index == index, "prediction"] = decoded_output

    predictions_file_name_suffix = csv_path.split("/")[1].replace(".csv", "") + ".csv"
    predictions_file_path = f"results/predictions_{predictions_file_name_suffix}"
    print(f"%% Predictions file path: {predictions_file_path}")

    df.to_csv(predictions_file_path)
    return predictions_file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("-p", "--csv-path", default="data/test.csv")
    parser.add_argument("-m", "--model-name", default=FLAN_T5_SMALL)
    args = parser.parse_args()

    predict(args.csv_path, args.model_name, no_company_specific=True)
