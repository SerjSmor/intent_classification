from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from app.utils import build_prompt
from consts import GENERATOR_LABELS, PROMPT_OPTIONS, GENERATOR_TEXT, GENERATOR_TEXT_NO_COMPANY
from dataset import _load_dataset, load_main_dataset, load_simple_dataset


def extract_prompt(examples):
    # "class_name", "class number
    class_names = []
    for example in examples:
        full_class_name = f"{example['class number']}.{example['class']}"
        if full_class_name not in class_names:
            class_names.append(full_class_name)
    options = "\n".join(class_names)
    prompt = f'''
        OPTIONS:
        {options}
    '''
    return prompt

def preprocess(dataset: List[Dict] = None, save_to_disk=True):
    if not dataset:
        dataset = load_main_dataset()

    all_df = prepare_dataset(dataset)

    df_train, df_test = train_test_split(all_df, test_size=0.2, random_state=42)
    df_train_positive_df = create_positive_negative_samples(all_df)
    print(f"total samples: {all_df.shape[0]}")
    if save_to_disk:
        df_train.to_csv("data/train.csv", index=False)
        df_train_positive_df.to_csv("data/train_positive.csv", index=False)
        df_test.to_csv("data/test.csv", index=False)

    return df_train, df_test, df_train_positive_df


def prepare_dataset(dataset):
    all_df = pd.DataFrame()
    for company_dict in dataset:
        # create train validation split
        examples = company_dict["Examples"]
        prompt_options = extract_prompt(examples)
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