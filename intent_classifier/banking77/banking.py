import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report

from app.banking77.banking_utils import BANKING77_INTENT_MAPPING, FEW_SHOT_EXAMPLES, ORIGINAL_BANKING77_INTENT_MAPPING
from app.model import IntentClassifier
from app.utils import get_model_suffix, build_prompt, build_entity_extraction_prompt
from consts import ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV, ATIS_PREDICTIONS_CSV, FLAN_T5_SMALL, LOCAL_FLAN_T5_SMALL, \
    BEST_LONG_FORMAT_SINGLE_TASK_EXTRACTION_MODEL, BEST_SHORT_FORMAT_SINGLE_TASK_EXTRACTION_MODEL, ANOTHER_MODEL, \
    BANKING77_PREDICTIONS_CSV, BANKING77_CLASSIFICATION_REPORT_CSV, BEST_MODEL_W90



def test_banking_dataset(full_model_path, intent_mapping=BANKING77_INTENT_MAPPING, save_file=True, use_default_labels=False,
                      no_company_specific=True, no_number_prompt=False, extract_entities=False, few_shot_map=FEW_SHOT_EXAMPLES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentClassifier(model_name=full_model_path, device=device)
    print(f"extract entities: {extract_entities}, model: {full_model_path}")
    dataset = load_dataset("PolyAI/banking77")


    all_intents = [row["label"] for row in dataset["test"]]
    intents = np.unique(all_intents)

    prompt_options = create_prompt_options(intent_mapping, intents, no_number_prompt, use_default_labels, few_shot_map)
    print(prompt_options)
    results = []
    company_name = "Atis Airlines"
    company_specific = "Airlines flights, meals, seats customer requests"
    #
    extraction_model = None
    if extract_entities:
        extraction_model = IntentClassifier(ANOTHER_MODEL)
    unique_texts = []


    for i, row in tqdm(enumerate(dataset["test"])):
        # if row["label"] != 22:
        #     continue

        intent = row["label"]
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
        prediction = prediction.replace("Class name: ", "").lower()
        # keywords = model.raw_predict(f"All of the verbs: {row['text']}")
        y = intent_mapping[intent].lower()

        if intent in few_shot_map:
            for few_sample_class_name in few_shot_map[intent]:
                few_sample_class_name = few_sample_class_name.lower()
                if prediction == few_sample_class_name:
                    y = few_sample_class_name
                    break
        result = {"prediction": prediction, "y": y, "text": text, "prompt": input_text,
                        "prompt_options": prompt_options}
        if len(original) > 0:
            result["original"] = original
        results.append(result)

        unique_texts.append(row["text"])

    df_results = pd.DataFrame(results)
    print(f"## total rows: {len(results)}, printing bank predictions distribution for model: {full_model_path}")
    print(df_results["prediction"].value_counts())

    if save_file:
        df_results.to_csv(BANKING77_PREDICTIONS_CSV, index=False)
    df_errors = df_results[df_results["y"] != df_results["prediction"]]
    if save_file:
        df_errors.to_csv("results/banking77_errors.csv", index=False)
    y = [r["y"] for r in results]
    predictions = [r["prediction"].replace("Class name: ", "") for r in results]
    print(predictions[:10])
    print(y[:10])
    print(classification_report(y, predictions))
    classification_report_df = pd.DataFrame(classification_report(y, predictions, output_dict=True)).T
    classification_report_df["id"] = classification_report_df.index.tolist()

    if save_file:
        classification_report_df.to_csv(BANKING77_CLASSIFICATION_REPORT_CSV, index=False)

    return classification_report_df


def create_prompt_options(intent_mapping, intents, no_number_prompt=False, use_default_labels=False, few_shot_map=Dict) -> str:
    print(f"create prompt options: intent_mapping: {intent_mapping}, no_number_prompt: {no_number_prompt},"
          f" use_default_labels: {use_default_labels}")
    prompt_options = "OPTIONS\n"
    index = 1
    for intent in intents:
        if intent not in intent_mapping:
            continue
        if intent in few_shot_map:
            for few_shot_example in few_shot_map[intent]:
                prompt_options += f"# {few_shot_example}\n"
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
    prompt_options += " # other # general # unknown "
    print(f"prompt options: {prompt_options}")
    return prompt_options

if __name__ == '__main__':

    test_banking_dataset(
        BEST_MODEL_W90,
        ORIGINAL_BANKING77_INTENT_MAPPING,
        no_number_prompt=True, few_shot_map={})



    # with verbalizer and

    # 57 accuracy 58 weighted f1 with few shot. 58 accuracy 56 f1 without few shot.

    # with few shot 5 and 20 n samples and BAII base
    #  weighted avg       0.74      0.70      0.70      3080

    # with top 5 similarity weighted avg       0.69      0.62      0.61
    # top 2
    # top 10

    # test_banking_dataset("Serj/intent-classifier", no_number_prompt=True)
    # test_banking_dataset("archived_experiments/b0.00_f1w_0.85_m:flan-t5-base_e:1_b:8_t:07122024-11:31:23_ncs:True_ups:False_udl:False_nnp:True_eet:False/models/flan-t5-base", no_number_prompt=True)

    # test_banking_dataset(
    #     "archived_experiments/b0.00_f1w_0.86_m:flan-t5-base_e:3_b:8_t:07122024-14:28:46_ncs:True_ups:False_udl:False_nnp:True_eet:False/models/flan-t5-base",
    #     no_number_prompt=True, few_shot_map={}) # 57 accuracy 58 weighted f1 with few shot. 58 accuracy 56 f1 without few shot.

    # test_banking_dataset(
    #     "archived_experiments/b0.00_f1w_0.87_m:flan-t5-base_e:2_b:8_t:07122024-14:22:38_ncs:True_ups:False_udl:False_nnp:True_eet:False/models/flan-t5-base",
    #     no_number_prompt=True, few_shot_map={}) # 53 54 without few shot


    # test_banking_dataset(
    #     "archived_experiments/b81.74_f1w_0.86_m:flan-t5-base_e:3_b:8_t:07052024-15:41:21_ncs:True_ups:False_udl:False_nnp:True_eet:True/models/flan-t5-base",
    #     no_number_prompt=True, few_shot_map={}) # 53 54 without few shot

    # test_banking_dataset(
    #     "archived_experiments/b0.00_f1w_0.84_m:flan-t5-base_e:6_b:8_t:07122024-15:53:41_ncs:True_ups:False_udl:False_nnp:True_eet:False/models/flan-t5-base",
    #     no_number_prompt=True, few_shot_map={}) # 54 52

    # test_banking_dataset(
    #     "archived_experiments/b0.00_f1w_0.86_m:flan-t5-base_e:4_b:8_t:07122024-16:16:51_ncs:True_ups:False_udl:False_nnp:True_eet:False/models/flan-t5-base",
    #     no_number_prompt=True, few_shot_map={}) # 55 57

    # test_banking_dataset(
    #     "archived_experiments/b76.91_f1w_0.79_m:flan-t5-base_e:4_b:8_t:07122024-18:04:52_ncs:True_ups:False_udl:False_nnp:True_eet:True/models/flan-t5-base",
    #     no_number_prompt=True, few_shot_map={}) # 53 55
    pass

# df = pd.read_csv(BANKING77_PREDICTIONS_CSV)
    # df["prediction"] = df["prediction"].apply(str.lower)
    # df["y"] = df["y"].apply(str.lower)
    # print(classification_report(df["y"], df["prediction"]))
    # parser = argparse.ArgumentParser(description="Predict atis")
    # parser.add_argument("-m", "--model-name", default=LOCAL_FLAN_T5_SMALL)
    # parser.add_argument("--use-default-labels", action="store_true")
    # parser.add_argument("--no-number-prompt", action="store_true")
    # args = parser.parse_args()
    #
    # print(vars(args))
    #
    # test_atis_dataset(args.model_name, save_file=True, use_default_labels=args.use_default_labels, no_number_prompt=args.no_number_prompt,
    #                   no_company_specific=True)

