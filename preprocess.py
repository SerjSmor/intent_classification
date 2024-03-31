import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import load_dataset


def preprocess():
    dataset = load_dataset()
    pizza_company = dataset[0]
    # create train validation split
    examples = pizza_company["Examples"]
    df = pd.DataFrame(examples)
    df.rename(columns={"class": "class_name", "class number": "class_number"}, inplace=True)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["class_number"])
    df_train.to_csv("data/train.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)

if __name__ == '__main__':
    preprocess()