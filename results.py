import pandas as pd

if __name__ == '__main__':
    predictions_df = pd.read_csv("data/predictions.csv")
    print(predictions_df["prediction"].value_counts())

