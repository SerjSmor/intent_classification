from invoke import task
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

from app.model import IntentClassifier
from preprocess import preprocess_simple_dataset

@task
def preprocess(c):
    c.run("python preprocess.py")

@task
def train(c, model_name="google/flan-t5-base", epochs=6, batch_size=8):
    c.run("python preprocess.py")
    c.run(f"python t5_generator_trainer.py --model-name {model_name} -e {epochs} -bs {batch_size}")
    model_suffix = model_name.split("/")[1]
    c.run(f"python predict.py --model-name models/{model_suffix} -p data/test.csv")
    c.run("python results.py -p data/predictions_test.csv -pd")

@task
def train_base(c):
    train(c, model_name="google/flan-t5-base")

@task
def train_small(c):
    # train(c, model_name="google/flan-t5-small")
    # train(c, model_name="google/flan-t5-small", epochs=20)
    train(c, model_name="flan-t5-small", epochs=20)

@task
def simple_dataset_base(c):
    preprocess_simple_dataset()
    c.run("python predict.py --model-name models/flan-t5-base -p data/simple_dataset.csv")
    c.run("python results.py -p data/predictions_simple_dataset.csv")

@task
def simple_dataset_small(c):
    preprocess_simple_dataset()
    c.run("python predict.py --model-name models/flan-t5-small -p data/simple_dataset.csv")
    c.run("python results.py -p data/predictions_simple_dataset.csv")


@task
def predict_test_hf_model(c):
    c.run("python preprocess.py")
    c.run("python predict.py --model-name serj/intent-classifier -p data/test.csv")
    c.run("python results.py -p data/predictions_test.csv -pd")

@task
def upload_to_hf(c):
    model = T5ForConditionalGeneration.from_pretrained("models/flan-t5-base")
    # push to the hub
    model.push_to_hub("intent-classifier", token="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", commit_message="Add Atis dataset sub samples to training dataset")

@task
def upload_to_hf_small(c):
    model = T5ForConditionalGeneration.from_pretrained("models/flan-t5-small")
    # push to the hub
    model.push_to_hub("intent-classifier-flan-t5-small", token="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",
                      commit_message="Add Atis dataset sub samples to training dataset")