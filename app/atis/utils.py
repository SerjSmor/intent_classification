import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report

from app.model import IntentClassifier
from app.utils import get_model_suffix
from consts import ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV

ATIS_INTENT_MAPPING = {
    'abbreviation': "Abbreviation and Fare Code Meaning Inquiry",
    'aircraft': "Aircraft Type Inquiry",
    # 'aircraft+flight+flight_no': "Flight Details Inquiry",
    'airfare': "Airfare Information Requests",
    # 'airfare+flight_time': "Flight Cost and Duration Inquiry",
    'airline': "Airline Information Request",
    # 'airline+flight_no': "Airline and Flight Number Query",
    'airport': "Airport Information and Queries",
    'capacity': "Aircraft Seating Capacity Inquiry",
    'cheapest': "Cheapest Fare Inquiry",
    'city': "Airport Location Inquiry",
    'distance': "Airport Distance Inquiry",
    'flight': "Flight Booking Request",
    # 'flight+airfare': "Flight and Fare Inquiry",
    'flight_no': "Flight Number Inquiry",
    'flight_time': "Flight Schedule Inquiry",
    'ground_fare': "Ground Transportation Cost Inquiry",
    'ground_service': "Ground Transportation Inquiry",
    'ground_service+ground_fare': "Airport Ground Transportation and Cost Query",
    'meal': "Inquiry about In-flight Meals",
    'quantity': "Flight Quantity Inquiry",
    'restriction': "Flight Restriction Inquiry"
}

def test_atis_dataset(full_model_path, intent_mapping=ATIS_INTENT_MAPPING, no_company_specific=False):
    model = IntentClassifier(model_name=full_model_path)

    dataset = load_dataset("tuetschek/atis")
    intents = set([row["intent"] for row in dataset["test"]])

    prompt_options = create_prompt_options(intent_mapping, intents)

    results = []
    company_name = "Atis Airlines"
    company_specific = "Airlines flights, meals, seats customer requests"

    for row in tqdm(dataset["test"]):
        intent = row["intent"]
        if intent not in intent_mapping:
            continue

        prediction = model.predict(row["text"], prompt_options, company_name, company_specific, no_company_specific)
        # keywords = model.raw_predict(f"All of the verbs: {row['text']}")
        results.append({"prediction": prediction.replace("Class name: ", ""), "y": intent_mapping[intent], "text": row["text"]})

    df_results = pd.DataFrame(results)
    df_results.to_csv("results/atis_predictions.csv", index=False)
    df_errors = df_results[df_results["y"] != df_results["prediction"]]
    df_errors.to_csv("results/atis_errors.csv", index=False)
    y = [r["y"] for r in results]
    predictions = [r["prediction"].replace("Class name: ", "") for r in results]
    classification_report_df = pd.DataFrame(classification_report(y, predictions, output_dict=True)).T
    classification_report_df["id"] = classification_report_df.index.tolist()
    print(classification_report_df.index.tolist())

    classification_report_df.to_csv(ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV, index=False)
    return classification_report_df


def create_prompt_options(intent_mapping, intents) -> str:
    prompt_options = "OPTIONS\n"
    index = 1
    for intent in intents:
        if intent not in intent_mapping:
            continue

        mapping = intent_mapping[intent]
        prompt_options += f" {index}. {mapping} \n"
        index += 1
    prompt_options = f'''{prompt_options}'''
    return prompt_options