import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import pandas as pd

from app.utils import build_prompt
from consts import FULL_PROMPT, GENERATOR_TEXT, GENERATOR_LABELS

FLAN_T5_BASE = "google/flan-t5-base"
FLAN_T5_LARGE = "google/flan-t5-large"
FLAN_T5_SMALL = "google/flan-t5-small"

TWENTY_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16

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



def train(model_name: str = FLAN_T5_BASE, epochs: int = TWENTY_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train = pd.read_csv("data/train.csv")
    df_validation = pd.read_csv("data/test.csv")
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

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.cuda()  # Move model to CUDA
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

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
    trainer.evaluate()
    model_suffix = model_name.split("/")[1]
    save_dir = f"models/{model_suffix}/"
    print(f"saving model in {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-m", "--model-name", default=FLAN_T5_BASE)
    parser.add_argument("-e", "--epochs", type=int, default=TWENTY_EPOCHS)
    parser.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()
    print(vars(args))
    train(args.model_name, epochs=args.epochs, batch_size=args.batch_size)
