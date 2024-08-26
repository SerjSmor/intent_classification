from sklearn.metrics import classification_report

from difflib import get_close_matches
from typing import List

import pandas as pd

from app.banking77.banking_utils import BANKING77_INTENT_MAPPING


def analyze(predictions: List[str], label_names: List[str]) -> List[str]:
    '''
    :param predictions: original predictions
    :param label_names: label names to match with
    :return: for each prediction, return the closest lexical label
    '''

    predictions = [pred.lower() for pred in predictions]
    label_names = [label.lower() for label in label_names]

    new_predictions = []
    for pred in predictions:
        # Get the closest match from label_names
        if pred in label_names:
            new_predictions.append(pred)
            continue

        closest_match = get_close_matches(pred, label_names, n=1, cutoff=0.0)
        if closest_match:
            new_predictions.append(closest_match[0])
        else:
            # If no close match is found, return the original prediction or a placeholder
            new_predictions.append(pred)

    return new_predictions


if __name__ == '__main__':
    df = pd.read_csv("data/atis_zero_shot.csv")
    predictions = df["prediction"]
    labels = df["label"]
    new_predictions = analyze(predictions, BANKING77_INTENT_MAPPING.values())
    print(classification_report(labels, predictions))

    print(classification_report(labels, new_predictions))
