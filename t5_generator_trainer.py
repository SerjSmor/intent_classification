import argparse
import gc
from typing import Dict

from datasets import Dataset, Value

import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    T5ForConditionalGeneration, DataCollatorForSeq2Seq

import torch
import pandas as pd

from app.atis.utils import test_atis_dataset
from app.utils import get_model_suffix
from consts import GENERATOR_TEXT, GENERATOR_LABELS, FLAN_T5_BASE, TWENTY_EPOCHS, DEFAULT_WARMUP_STEPS, \
    DEFAULT_BATCH_SIZE, DEFAULT_WEIGHT_DECAY, FLAN_T5_LARGE, DEFAULT_EXPERIMENT_NAME, GENERATOR_TEXT_NO_COMPANY, \
    TEST_SET_CLASSIFICATION_REPORT_CSV
from predict import predict
from results import calculate_classification_report, results


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



def train(model_name: str = FLAN_T5_BASE, epochs: int = TWENTY_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
          test_atis: bool = False, peft: bool = False, warmup_steps=DEFAULT_WARMUP_STEPS,
          weight_decay=DEFAULT_WEIGHT_DECAY, dataset_names=[], no_company_specific: bool = False,
          use_positive: bool =False):

    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train = pd.read_csv("data/train.csv")
    # df_train = df_train.head(10)
    df_validation = pd.read_csv("data/test.csv")

    # df_validation = df_validation.head(10)
    # Example dataset


    if use_positive:
        print(f"before adding positive samples - train rows {df_train.shape[0]}")
        positive_df = pd.read_csv("data/train_positive.csv")
        df_train = pd.concat([df_train, positive_df], ignore_index=True)
        print(f"after adding positive samples - train rwos {df_train.shape[0]}")

    wandb.log({"train_rows": df_train.shape[0], "validation_rows": df_validation.shape[0],
               "dataset_names": dataset_names, "dataset_num": len(dataset_names)})

    text_column = GENERATOR_TEXT if no_company_specific else GENERATOR_TEXT_NO_COMPANY
    df_train = df_train[[text_column, GENERATOR_LABELS]]
    df_validation = df_validation[[text_column, GENERATOR_LABELS]]

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_validation)

    train_dataset = train_dataset.rename_column(text_column, "text")
    train_dataset = train_dataset.rename_column(GENERATOR_LABELS, "label")
    val_dataset = val_dataset.rename_column(text_column, "text")
    val_dataset = val_dataset.rename_column(GENERATOR_LABELS, "label")

    def get_lengths(examples):
        return {'input_length': [len(tokenizer.encode(text)) for text in examples['text']]}

    lengths_dataset = train_dataset.map(get_lengths, batched=True, batch_size=1000)
    lengths = lengths_dataset['input_length']
    print(lengths)
    # Optional: Use map to preprocess data

    max_input_length = 320
    if model_name == FLAN_T5_LARGE:
        max_input_length = 320

    def preprocess_function(examples):
        inputs = tokenizer(examples["text"], truncation=True, max_length=max_input_length)
        # inputs["label"] = examples["label"]

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(examples['label'], truncation=True, max_length=40)
        inputs["labels"] = targets["input_ids"]
        return inputs

    train_tokenized_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=batch_size).shuffle(seed=42)
    val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True, batch_size=batch_size).shuffle(seed=42)

    train_tokenized_dataset = train_tokenized_dataset.remove_columns(
        train_dataset.column_names
    )

    val_tokenized_dataset = val_tokenized_dataset.remove_columns(
        val_dataset.column_names
    )

    features = [train_tokenized_dataset[i] for i in range(2)]

    model_args = {}
    if peft:
        model_args = {
            "load_in_8bit": True,
            "device_map": "auto",
        }

    model = T5ForConditionalGeneration.from_pretrained(model_name, **model_args)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    data_collator(features)
    print(features)

    if not peft:
        if torch.cuda.is_available():
            model.cuda()  # Move model to CUDA
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # Define training arguments

    args = {
        # "bf16": True,
        "output_dir": './results',
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "save_strategy": "no",
        "logging_dir": './logs',
        "logging_steps": 10,
        "evaluation_strategy": "epoch",
    }

    if model_name == FLAN_T5_LARGE:
        args["optim"] = "adafactor"
        args["gradient_accumulation_steps"] = 4

    training_args = TrainingArguments(**args)

    from peft import LoraConfig


    if peft:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.add_adapter(peft_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
    predictions_file_path = predict("data/test.csv", save_dir, no_company_specific)
    # report test set results
    test_set_classification_report, missing_percentage = results(predictions_file_path, per_dataset=False)
    test_set_classification_report_df = pd.DataFrame(data=test_set_classification_report).transpose()
    test_set_classification_report_df["id"] = test_set_classification_report_df.index.tolist()
    test_set_classification_report = test_set_classification_report_df
    test_set_last_row_dict = test_set_classification_report_df.iloc[-1].to_dict()
    test_set_accuracy = test_set_classification_report_df.iloc[-3]["precision"]
    test_set_classification_report_df.to_csv(TEST_SET_CLASSIFICATION_REPORT_CSV, index=False)
    # Function to rename keys
    def rename_keys(d: Dict, prefix: str):
        renamed_dict = {prefix + key: value for key, value in d.items()}
        return renamed_dict

    test_set_last_row_dict = rename_keys(test_set_last_row_dict, "val_weighted_")


    if test_atis:
        atis_classification_report_df = test_atis_dataset(full_model_path=save_dir,
                                                          no_company_specific=no_company_specific)
        atis_last_row_dict = atis_classification_report_df.iloc[-1].to_dict()
        atis_accuracy = atis_classification_report_df.iloc[-3]["precision"]
        atis_last_row_dict = rename_keys(atis_last_row_dict, "atis_weighted_")

        wandb.log({**atis_last_row_dict, **test_set_last_row_dict,
                   "weight_decay": weight_decay, "peft": peft, "warmup_steps": warmup_steps,
                   "val_set_accuracy": test_set_accuracy, "atis_accuracy": atis_accuracy,
                   "max_input_length": max_input_length,
            "test_set_classification_report": wandb.Table(dataframe=test_set_classification_report),
                   "tet_set_missing_percentage": missing_percentage,
                   "atis_test_set": wandb.Table(dataframe=atis_classification_report_df)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-m", "--model-name", default=FLAN_T5_BASE)
    parser.add_argument("-e", "--epochs", type=int, default=TWENTY_EPOCHS)
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--test-atis", action="store_true")
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--dataset-names", type=str, nargs="+", help="Param for wandb logging")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--no-company-specific", action="store_true")
    parser.add_argument("--use-positive-samples", action="store_true")

    args = parser.parse_args()
    print(vars(args))
    if len(args.name) > 0:
        wandb.init(
            project="intent_classifier_paper",
            name=args.name,
            config={
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "dataset_names": args.dataset_names,
                "peft": args.peft,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "use_positive_samples": args.use_positive_samples,
            },
        )

    train(args.model_name, epochs=args.epochs, batch_size=args.batch_size, test_atis=args.test_atis, peft=args.peft,
          warmup_steps=args.warmup_steps, weight_decay=args.weight_decay, dataset_names=args.dataset_names,
          no_company_specific=args.no_company_specific, use_positive=args.use_positive_samples)
