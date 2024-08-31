from dataclasses import dataclass
from difflib import get_close_matches
from typing import List, Union

import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from setfit import get_templated_dataset, TrainingArguments, SetFitModel, Trainer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from app.atis.utils import ATIS_INTENT_MAPPING, create_prompt_options
from app.banking77.banking_utils import BANKING77_INTENT_MAPPING
from app.embeddings import Embedder
from app.model import IntentClassifier
from app.utils import build_prompt
from app.verbalizer import analyze
from consts import BEST_BANKING_MODEL

BAII_SMALL = "BAAI/bge-small-en-v1.5"
BAII_BASE = "BAAI/bge-base-en-v1.5"
BAII_LARGE = "BAAI/bge-large-en-v1.5"

SNOWFLAKE_EMBEDDINGS = "Snowflake/snowflake-arctic-embed-m-v1.5"

DEFAULT_TOP_N_SIMILARITY = 5
RANDOM_STATE = 42
MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"
ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PREDICTION_COLUMN = "prediction"


@dataclass
class Category:
    text: str = None
    n: int = 0
    n_representatives: List['Sample'] = None


@dataclass
class Sample:
    text: str = None
    category: Category = None
    embedding: np.ndarray = None


@dataclass
class LabelledDataset:
    name: str
    df: pd.DataFrame
    text_column: str
    class_column: str
    prediction_column: str = "prediction"
    prompt_column: str = "prompt"


class FewShotDataset(LabelledDataset):

    def __init__(self, df: pd.DataFrame, text_column: str, class_column, n: int, should_embed=True,
                 embedding_model=MPNET_BASE_V2, prediction_column=DEFAULT_PREDICTION_COLUMN, name="atis_zero_shot"):
        super().__init__(name, df, text_column, class_column, prediction_column, "prompt")
        self.n = n
        self.categories: List[Category] = self.transform_df_to_categories(df, text_column, class_column, n)

        if should_embed:
            self.embedder = Embedder(embedding_model)
            self.embed_samples()
        self.name = name


    @classmethod
    def transform_df_to_categories(cls, df: pd.DataFrame, text_column: str, class_column: str, n: int) -> \
            List[Category]:
        # for each class
        all_categories = df[class_column].tolist()
        unique_categories = np.unique(all_categories)
        process_categories: List[Category] = []
        for category in unique_categories:
            samples_df = df[df[class_column] == category]
            c = Category(category, n_representatives=[])
            process_categories.append(c)
            if n == 0:
                s = Sample(text=c.text, category=c)
                c.n_representatives.append(s)
                continue

            current_n = min(n, samples_df.shape[0])
            sampled_samples_df = samples_df.sample(n=current_n, random_state=RANDOM_STATE)
            for index, row in sampled_samples_df.iterrows():
                s = Sample(text=row[text_column], category=c)
                c.n_representatives.append(s)

        return process_categories

    def embed_samples(self):
        self.texts = []
        self.samples = []
        for category in tqdm(self.categories):
            for sample in category.n_representatives:
                self.texts.append(sample.text)
                self.samples.append(sample)

        self.embeddings = self.embedder.embed(self.texts)

    def find_n_similar_samples(self, text, n) -> Union[List[Category], List[Sample]]:
        indices, angles = self.embedder.top_n(text, n)
        top_n_samples = [self.samples[i] for i in indices]
        if self.n == 0:
            # top_n_samples = self.samples[indices]
            return [sample.category for sample in top_n_samples]
        else:
            return top_n_samples


class FewShotPipeline:
    def __init__(self, model: IntentClassifier, few_shot_dataset: FewShotDataset, n: int):
        self.model = model
        self.few_shot_dataset = few_shot_dataset
        self.n = n

    def run(self):
        raise NotImplementedError()


def map_text_to_sample(samples, closest_option):
    for sample in samples:
        if sample.text.lower() == closest_option:
            return sample

    raise Exception(f"Didn't find closest sample: {closest_option} from: {samples}")


class ClassificationPipeline(FewShotPipeline):
    def __init__(self, model: IntentClassifier, few_shot_dataset: FewShotDataset, n: int = 0, use_only_similarity=False):
        super().__init__(model, few_shot_dataset, n)
        self.use_only_similarity = use_only_similarity

    @classmethod
    def create_prompt_options(cls, categories: List[Category]):
        # extract options
        # prompt = return f"Topic %% Customer: $$.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n{prompt}\nClass name: "
        prompt_options = "Options:\n"
        for category in categories:
            prompt_options += f"# {category.text} \n"


        return prompt_options

    def classify(self, text: str, use_similarity=False, top_n_similarity=10, use_only_similarity=False) -> (str, str, List):
        samples = self.few_shot_dataset.categories
        if use_similarity:
            samples = self.few_shot_dataset.find_n_similar_samples(text, top_n_similarity)

            if use_only_similarity:
                return samples[0].text, "", samples

        prompt_options = self.create_prompt_options(samples)
        prompt = build_prompt(text, prompt_options, no_company_specific=True)
        prediction = self.model.predict(prompt)

        # prediction, prompt = self.model.predict(text, prompt_options, return_prompt=True)
        return prediction, prompt, samples

    def pipeline(self, dataset: LabelledDataset, top_n_similarity=DEFAULT_TOP_N_SIMILARITY, use_similarity=False, use_verbalizer=False) -> List[str]:
        predictions = []
        verbalizer_predictions = []
        prompts = []
        closest_matches = []
        for index, row in tqdm(dataset.df.iterrows()):
            prediction, prompt, samples = self.classify(row[dataset.text_column], use_similarity=use_similarity,
                                       top_n_similarity=top_n_similarity, use_only_similarity=self.use_only_similarity)
            predictions.append(prediction)
            prompts.append(prompt)
            options = [category.text.lower() for category in samples]

            closest_match = get_close_matches(prediction, options, n=1, cutoff=0.0)
            closest_option = closest_match[0]
            verbalizer_predictions.append(closest_option)
            # how do I map between
            sample = map_text_to_sample(samples, closest_option)
            category_name = sample.text if type(sample) == Category else sample.category.text.lower()
            closest_matches.append(category_name)

        dataset.df[dataset.prediction_column] = predictions
        dataset.df[dataset.prompt_column] = prompts
        dataset.df[dataset.prediction_column] = dataset.df[dataset.prediction_column].apply(str.lower)
        dataset.df[dataset.class_column] = dataset.df[dataset.class_column].apply(str.lower)

        y = dataset.df[dataset.class_column].tolist()
        predictions = dataset.df[dataset.prediction_column].tolist()

        # if self.few_shot_dataset.n > 0:
            # check equality between sample category and sample
        dataset.df["original_prediction"] = dataset.df[dataset.prediction_column]
        dataset.df["few_shot"] = closest_matches

        few_shot_predictions = closest_matches

        self.calculate_classification_report(y, predictions)
        # new_predictions = analyze(predictions, BANKING77_INTENT_MAPPING.values())
        dataset.df["v_pred"] = verbalizer_predictions
        print("**** classification report with verbalizer ****")
        print(classification_report(y, verbalizer_predictions))

        print("**** classification report with few shot closes matches ****")
        print(classification_report(y, closest_matches))

        verbalizer_output_dict = classification_report(y, verbalizer_predictions, output_dict=True)
        verbalizer_weighted_f1 = verbalizer_output_dict["weighted avg"]["f1-score"]
        out_predictions_file_path = f"data/{dataset.name}_top_{top_n_similarity}_sim_{use_similarity}_vf1_{verbalizer_weighted_f1:.2f}.csv"
        dataset.df.to_csv(out_predictions_file_path, index=False)

        return predictions

    def calculate_classification_report(self, y, predictions):
        print(classification_report(y, predictions))
        return predictions


def atis_convert_old_label_to_class_atis(old_label: str):
    if old_label in ATIS_INTENT_MAPPING:
        return ATIS_INTENT_MAPPING[old_label]
    return None


def atis_pipeline(use_similarity=True, n_shot=0, n_similarity=5, embedding_model=SNOWFLAKE_EMBEDDINGS, n_samples=0, use_only_similarity=False):
    dataset = load_dataset("tuetschek/atis")
    dataset.set_format(type="pandas")

    df_train: pd.DataFrame = dataset["train"][:]
    df_test: pd.DataFrame = dataset["test"][:]

    df_train["label"] = df_train["intent"].apply(lambda label: atis_convert_old_label_to_class_atis(label))
    df_test["label"] = df_test["intent"].apply(lambda label: atis_convert_old_label_to_class_atis(label))
    df_train.dropna(subset=["label"], inplace=True)
    df_test.dropna(subset=["label"], inplace=True)

    # categories = FewShotDataset.transform_df_to_categories(df, "text", "label", 0)
    few_shot_dataset = FewShotDataset(df_train, "text", "label", n=n_shot,
                                      embedding_model=embedding_model)

    model = IntentClassifier(BEST_BANKING_MODEL)
    zero_shot = ClassificationPipeline(model, few_shot_dataset, use_only_similarity=use_only_similarity)
    name = f"atispredictions_top{n_shot}_similarity{use_similarity}"

    if n_samples > 0:
        df_test = df_test.head(n_samples)

    test_dataset = LabelledDataset(name, df_test, "text", "label")
    predictions = zero_shot.pipeline(test_dataset, use_similarity=use_similarity, top_n_similarity=n_similarity)
    # print(categories)

def banking77_pipeline(use_similarity=True, few_shot_n = 0, top_n=5, use_verbalizer=False, num_samples=0, embeddings_model=SNOWFLAKE_EMBEDDINGS, use_only_similarity=False):
    dataset = load_dataset("PolyAI/banking77")
    dataset.set_format(type="pandas")

    df_train: pd.DataFrame = dataset["train"][:]
    df_test: pd.DataFrame = dataset["test"][:]

    df_test["orig_label"] = df_test["label"]
    df_test["label"] = df_test["label"].apply(lambda label_num: BANKING77_INTENT_MAPPING[label_num] if label_num in BANKING77_INTENT_MAPPING else None)
    df_train["label"] = df_train["label"].apply(
        lambda label_num: BANKING77_INTENT_MAPPING[label_num] if label_num in BANKING77_INTENT_MAPPING else None)
    df_test.dropna(subset=["label"], inplace=True)
    df_train.dropna(subset=["label"], inplace=True)

    few_shot_dataset = FewShotDataset(df_train, "text", "label", n=few_shot_n,
                                      embedding_model=embeddings_model)
    model = IntentClassifier(BEST_BANKING_MODEL)
    zero_shot = ClassificationPipeline(model, few_shot_dataset, use_only_similarity=use_only_similarity)
    name = f"banking77predictions_top{top_n}_similarity{use_similarity}"
    if num_samples > 0:
        df_test = df_test.head(num_samples)

    test_dataset = LabelledDataset(name, df_test, "text", "label")
    zero_shot.pipeline(test_dataset, use_similarity=use_similarity, top_n_similarity=top_n, use_verbalizer=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')

    # You can include any other metrics you want here
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def setfit_zero_shot_pipeline_banking77(model_name="BAAI/bge-small-en-v1.5"):

    dataset = load_dataset("PolyAI/banking77")
    dataset.set_format(type="pandas")

    df: pd.DataFrame = dataset["test"][:]
    df["orig_label"] = df["label"]
    # df["label"] = df["label"].apply(
    #     lambda label_num: BANKING77_INTENT_MAPPING[label_num] if label_num in BANKING77_INTENT_MAPPING else None)
    # df.dropna(subset=["label"], inplace=True)

    # labels = list(set(df["label"]))
    labels = BANKING77_INTENT_MAPPING.values()
    train_dataset = get_templated_dataset(candidate_labels=labels, sample_size=1)
    model = SetFitModel.from_pretrained(model_name)

    args = TrainingArguments(
        batch_size=32,
        num_epochs=1,
        num_iterations=1,
    )

    test_dataset = datasets.Dataset.from_pandas(df[["label", "text"]])

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # metric="f1",
        # metric_kwargs={"weighted": ""}
        # compute_metrics=compute_metrics

    )
    trainer.train()

    metrics = trainer.evaluate()
    print(metrics)

    # Evaluate the model
    preds = trainer.model.predict(df["text"].tolist())
    y_true = df["label"].tolist()

    # Calculate weighted metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average='weighted'
    )

    # Print the metrics
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")



if __name__ == '__main__':
    # atis_pipeline(n_shot=5, n_similarity=20, use_similarity=True, embedding_model=BAII_SMALL, use_only_similarity=False)
    # banking77_pipeline(use_similarity=False, top_n=20, few_shot_n=5, embeddings_model=BAII_LARGE)

    # banking77_pipeline(True, top_n=20, few_shot_n=5, embeddings_model=BAII_LARGE)
    # banking77_pipeline(True, top_n=5, few_shot_n=0, num_samples=100, embeddings_model=BAII_SMALL, use_only_similarity=False)
    setfit_zero_shot_pipeline_banking77(model_name="BAAI/bge-small-en-v1.5")

    # setfit_zero_shot_pipeline_banking77(model_name="Snowflake/snowflake-arctic-embed-m-v1.5")

