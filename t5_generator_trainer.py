from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import pandas as pd

from app.utils import build_prompt
from consts import FULL_PROMPT

FLAN_T5_BASE = 'google/flan-t5-base'

# Convert to PyTorch datasets

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



def train(model_name=FLAN_T5_BASE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train = pd.read_csv("data/train.csv")
    df_validation = pd.read_csv("data/test.csv")
    # Example dataset


    df_train["generator_text"] = df_train["sample_text"].apply(lambda text: build_prompt(text, FULL_PROMPT))
    train_texts = df_train["generator_text"].tolist()  # Should be a list of texts
    df_train["generator_labels"] = df_train.apply(lambda row: f"Class name: {row['class_name']}, Class number: {row['class_number']}", axis=1)
    train_labels = df_train["generator_labels"].tolist()  # Should be a list of integer labels

    df_validation["generator_text"] = df_validation["sample_text"].apply(lambda text: build_prompt(text, FULL_PROMPT))
    val_texts = df_validation["generator_text"].tolist()  # Validation texts
    df_validation["generator_labels"] = df_validation.apply(lambda row: f"Class name: {row['class_name']}, Class number: {row['class_number']}", axis=1)
    val_labels = df_validation["generator_labels"].tolist()  # Validation labels

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
        num_train_epochs=40,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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
    save_dir = "models/pizza_company/flan_t5_base_generator/"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':
    train()