import pandas as pd
from sklearn.model_selection import train_test_split

from app.utils import build_prompt
from consts import GENERATOR_LABELS, PROMPT_OPTIONS, GENERATOR_TEXT
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

def preprocess():
    dataset = load_main_dataset()
    all_df = prepare_dataset(dataset)

    # df_train, df_test = train_test_split(all_df, test_size=0.2, random_state=42, stratify=all_df["class_number"])
    df_train, df_test = train_test_split(all_df, test_size=0.2, random_state=42)
    df_train.to_csv("data/train.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)


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
        df["dataset_name"] = company_name
        all_df = pd.concat([all_df, df])
    return all_df


def preprocess_simple_dataset():
    dataset = load_simple_dataset()
    df = prepare_dataset(dataset)
    df.to_csv("data/simple_dataset.csv", index=False)

if __name__ == '__main__':
    preprocess()
    # preprocess_simple_dataset()