import json
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from app.utils import build_prompt, build_entity_extraction_prompt, ENTITY_EXTRACTION_DATASET_JSON
from consts import GENERATOR_LABELS, PROMPT_OPTIONS, GENERATOR_TEXT, GENERATOR_TEXT_NO_COMPANY
from dataset import _load_dataset, load_main_dataset, load_simple_dataset

def extract_prompt(examples, no_number_prompt=False):
    class_names_str = ""
    class_names_set = set()
    for example in examples:
        if example['class'] in class_names_set:
            continue
        if no_number_prompt:
            class_names_str += f"% {example['class']} \n"
        else:
            class_names_str += f"{example['class number']}.{example['class']} \n"

        class_names_set.add(example['class'])
    prompt = f'''
        OPTIONS:
        {class_names_str}
    '''
    return prompt

def parse_entity_extraction_dataset(json_dataset_path) -> pd.DataFrame:
    f = open(json_dataset_path)
    dataset = json.load(f)
    results = []
    for company_dict in dataset:
        for row in company_dict["Examples"]:
            if "entity_action" not in row:
                continue
            result = {"sample_text": row["sample_text"], GENERATOR_TEXT: build_entity_extraction_prompt(row['sample_text']),
                      GENERATOR_LABELS: row["entity_action"], "task": "entity_extraction"}
            result[GENERATOR_TEXT_NO_COMPANY] = result[GENERATOR_TEXT]
            results.append(result)

    return pd.DataFrame(results)


def preprocess(dataset: List[Dict] = None, no_classification_task=False, save_to_disk=True, no_number_prompt=False, positive_samples=False, entity_extraction_task=False):
    print(f"(preprocess) no number prompt: {no_number_prompt}, positive samples: {positive_samples}, "
          f"entity_extraction_task: {entity_extraction_task}")
    if not dataset:
        dataset = load_main_dataset()


    all_df = prepare_dataset(dataset, no_number_prompt=no_number_prompt)
    print(f"total samples: {all_df.shape[0]}")

    df_train, df_test = train_test_split(all_df, test_size=0.2, random_state=42)
    df_train_positive_df = create_positive_negative_samples(all_df)
    if entity_extraction_task:
        entity_extraction_df = parse_entity_extraction_dataset(ENTITY_EXTRACTION_DATASET_JSON)
        # entity_extraction_df = pd.concat([entity_extraction_df.copy(), entity_extraction_df.copy(), entity_extraction_df.copy(),
        #                                   entity_extraction_df.copy(), entity_extraction_df.copy(), entity_extraction_df.copy(),
        #                                   entity_extraction_df.copy(), entity_extraction_df.copy()
        #                                   ])

        if no_classification_task:
            df_train, df_test = train_test_split(entity_extraction_df, test_size=0.2, random_state=42)
        else:
            sample_texts = entity_extraction_df["sample_text"].tolist()
            df_train_entity, df_test_entity = train_test_split(entity_extraction_df, test_size=0.2, random_state=42)

            df_train = df_train[~df_train["sample_text"].isin(sample_texts)]
            df_test = df_test[~df_test["sample_text"].isin(sample_texts)]
            df_train = pd.concat([df_train, df_train_entity], ignore_index=True)
            df_test = pd.concat([df_test, df_test_entity], ignore_index=True)

            # df_train = pd.concat([df_train, entity_extraction_df], ignore_index=True)
        print(f"total entity extraction samples: {entity_extraction_df.shape[0]}")
        print(f"total train: {df_train.shape[0]}")
    if save_to_disk:
        df_train.to_csv("data/train.csv", index=False)
        df_train_positive_df.to_csv("data/train_positive.csv", index=False)
        df_test.to_csv("data/test.csv", index=False)

    return df_train, df_test, df_train_positive_df


def prepare_dataset(dataset, no_number_prompt=False):
    all_df = pd.DataFrame()
    for company_dict in dataset:
        # create train validation split
        examples = company_dict["Examples"]
        prompt_options = extract_prompt(examples, no_number_prompt=no_number_prompt)
        df = pd.DataFrame(examples)
        df.rename(columns={"class": "class_name", "class number": "class_number"}, inplace=True)
        company_name = company_dict["Company Name"]
        df["Company Name"] = company_name
        # BUILD prompts
        df[GENERATOR_LABELS] = df["class_name"]
        df[PROMPT_OPTIONS] = prompt_options
        df[GENERATOR_TEXT] = df.apply(
            lambda row: build_prompt(row["sample_text"], row[PROMPT_OPTIONS], row["Company Name"]), axis=1)
        df[GENERATOR_TEXT_NO_COMPANY] = df.apply(
            lambda row: build_prompt(row["sample_text"], row[PROMPT_OPTIONS], company_specific=False), axis=1)

        df["dataset_name"] = company_name
        df["task"] = "classification"
        all_df = pd.concat([all_df, df])
        print(f"company: {company_name}, samples: {df.shape[0]}")
    return all_df

def create_positive_negative_samples(dataset_df: pd.DataFrame):
    # positive
    # randomize labels
    prompt = f'''
        Same customer intent? %% First customer: @@ Second customer: $$ END. \nOPTIONS: 1. yes, 2. no. Class name:
    '''
    positive_labels = dataset_df["class_name"].sample(frac=0.3)
    first_n = 8
    new_rows = []
    for label in positive_labels:
        current_n = first_n
        rows_df = dataset_df[dataset_df["class_name"] == label]
        while current_n > 0:
            if rows_df.shape[0] >= current_n:
                break
            current_n -= 2

        if current_n == 0:
            continue

        rows_sample_df = rows_df.sample(n=current_n)
        i = 0
        current_sample_text = ""
        for index, row in rows_sample_df.iterrows():
            if i % 2 == 0:
                # we create a new tuple
                current_sample_text = row["sample_text"]
                for sample in new_rows:
                    if current_sample_text in sample[GENERATOR_TEXT]:
                        current_sample_text = ""
                        break
            else:
                if len(current_sample_text) > 0:
                    generator_text = prompt.replace("@@", current_sample_text).replace("$$", row["sample_text"])
                    generator_label = "yes"

                    new_rows.append({GENERATOR_TEXT_NO_COMPANY: generator_text, GENERATOR_TEXT: generator_text,
                                     GENERATOR_LABELS: generator_label})

            i += 1
    return pd.DataFrame(new_rows)


def preprocess_simple_dataset():
    dataset = load_simple_dataset()
    df = prepare_dataset(dataset)
    df.to_csv("data/simple_dataset.csv", index=False)

if __name__ == '__main__':
    preprocess()
    # preprocess_simple_dataset()