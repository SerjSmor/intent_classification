import argparse

import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    T5ForConditionalGeneration

import torch
import pandas as pd

from app.atis.utils import test_atis_dataset
from app.utils import get_model_suffix
from consts import GENERATOR_TEXT, GENERATOR_LABELS
from predict import predict
from results import calculate_classification_report, results

FLAN_T5_BASE = "google/flan-t5-base"
FLAN_T5_LARGE = "google/flan-t5-large"
FLAN_T5_SMALL = "google/flan-t5-small"

TWENTY_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_EXPERIMENT_NAME = ""

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def tokenize_pairs(tokenizer, source_texts, target_texts):
    model_inputs = tokenizer(source_texts, padding=True, truncation=True, max_length=512)
    # Tokenize the targets and ensure the token type ids are not returned
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, padding=True, truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def train(model_name: str = FLAN_T5_BASE, epochs: int = TWENTY_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE, test_atis: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train = pd.read_csv("data/train.csv")
    # df_train = df_train.head(10)
    df_validation = pd.read_csv("data/test.csv")
    # df_validation = df_validation.head(10)
    # Example dataset

    train_texts = df_train[GENERATOR_TEXT].tolist()  # Should be a list of texts
    train_labels = df_train[GENERATOR_LABELS].tolist()  # Should be a list of integer labels

    val_texts = df_validation[GENERATOR_TEXT].tolist()  # Validation texts
    val_labels = df_validation[GENERATOR_LABELS].tolist()  # Validation labels

        # Example usage
    train_encodings = tokenize_pairs(tokenizer, train_texts, train_labels)
    val_encodings = tokenize_pairs(tokenizer, val_texts, val_labels)

    train_dataset = Seq2SeqDataset(train_encodings)
    val_dataset = Seq2SeqDataset(val_encodings)

    # Load the model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)


    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.cuda()  # Move model to CUDA
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Define training arguments

    args = {
        "bf16": True,
        "output_dir": './results',
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": './logs',
        "logging_steps": 10,
        "evaluation_strategy": "epoch"
    }

    if model_name == FLAN_T5_LARGE:
        args["optim"] = "adafactor"
        args["gradient_accumulation_steps"] = 4

    training_args = TrainingArguments(**args)

    # training_args = Seq2SeqTrainingArguments(
    #     fp16=True,
    #     output_dir="./results",
    #     num_train_epochs=epochs,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=10,
    #     evaluation_strategy="epoch"
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    # trainer.evaluate()
    model_suffix = get_model_suffix(model_name)
    save_dir = f"models/{model_suffix}/"
    print(f"saving model in {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # run test set predictions
    predictions_file_path = predict("data/test.csv", save_dir)
    # report test set results
    test_set_classification_report, missing_percentage = results(predictions_file_path, per_dataset=False)
    test_set_classification_report_df = pd.DataFrame(data=test_set_classification_report).transpose()
    test_set_classification_report_df["id"] = test_set_classification_report_df.index.tolist()
    test_set_classification_report = test_set_classification_report_df

    atis_classification_report = test_atis_dataset(full_model_path=save_dir)
    wandb.log({"test_set_classification_report": wandb.Table(dataframe=test_set_classification_report),
               "tet_set_missing_percentage": missing_percentage,
               "atis_test_set": wandb.Table(dataframe=atis_classification_report)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-m", "--model-name", default=FLAN_T5_BASE)
    parser.add_argument("-e", "--epochs", type=int, default=TWENTY_EPOCHS)
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--test-atis", action="store_true")

    args = parser.parse_args()
    print(vars(args))
    if len(args.name) > 0:
        wandb.init(
            project="intent_classifier",
            name=args.name,
            config={
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            },
        )

    train(args.model_name, epochs=args.epochs, batch_size=args.batch_size, test_atis=args.test_atis)
