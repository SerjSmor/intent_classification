import argparse

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

from app.utils import build_prompt

# Load the pre-trained T5 model and tokenizer
# model_name = "google/flan-t5-base"
# model_name = "google/flan-t5-xl"
# model_name = "google/flan-t5-large"
# model_name = "models/flan_t5_base_generator"


def predict(csv_path, model_name="models/flan-t5-small"):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Prompt containing the taxonomy
    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Classify each example
    results = []


    # TODO: change to batched predictions
    df = pd.read_csv(csv_path)
    for index, row in tqdm(df.iterrows()):
        # Concatenate the prompt with the example
        input_text = build_prompt(row["sample_text"], row["prompt_options"], row["Company Name"])
        # print(input_text)
        # Tokenize the concatenated inp_ut text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        # Generate the output
        output = model.generate(input_ids)

        # Decode the output tokens
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Append the prediction to the list
        df.loc[df.index == index, "prediction"] = decoded_output

    predictions_file_name_suffix = csv_path.split("/")[1].replace(".csv", "") + ".csv"
    predictions_file_path = f"results/predictions_{predictions_file_name_suffix}"
    print(f"%% Predictions file path: {predictions_file_path}")
    df.to_csv(predictions_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("-p", "--csv-path")
    parser.add_argument("-m", "--model-name")
    args = parser.parse_args()

    predict(args.csv_path, args.model_name)
