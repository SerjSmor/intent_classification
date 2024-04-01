from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import pandas as pd

FLAN_T5_BASE = 'google/flan-t5-base'

# Convert to PyTorch datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train(model_name=FLAN_T5_BASE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train = pd.read_csv("data/train.csv")
    df_validation = pd.read_csv("data/test.csv")
    # Example dataset

    train_texts = df_train["sample_text"].tolist()  # Should be a list of texts
    train_labels = df_train["class_number"].tolist()  # Should be a list of integer labels

    val_texts = df_validation["sample_text"].tolist()  # Validation texts
    val_labels = df_validation["class_number"].tolist()  # Validation labels

    # Tokenize the texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)


    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    # Load the model
    unique_train_labels = df_train["class_number"].unique()
    unique_validation_labels = df_validation["class_number"].unique()

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_train_labels))
    if torch.cuda.is_available():
        model.cuda()  # Move model to CUDA
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )


    # Define the compute_metrics function for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0], axis=1)
        # predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()
    save_dir = "archived_experiments/pizza_company/flan_t5_base/"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':
    train()