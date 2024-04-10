from invoke import task
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

from app.model import IntentClassifier
from preprocess import preprocess_simple_dataset


@task
def train_base(c):
    model_name = "google/flan-t5-base"
    c.run("python preprocess.py")
    c.run(f"python t5_generator_trainer.py --model-name {model_name} -e 20")
    c.run(f"python predict.py --model-name models/flan-t5-base -p data/test.csv")
    c.run("python results.py -p data/predictions_test.csv")

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
def hf_pipeline_example(c):
    classifier = pipeline("zero-shot-classification", model="serj/intent-classifier")
    sequence_to_classify = "Hey, after recent changes, I want to cancel subscription, please help."
    candidate_labels = ["refund", "cancel_subscription", "damaged_item", "return_item"]
    print(classifier(sequence_to_classify, candidate_labels))

@task
def helper_classes_example(c):
    m = IntentClassifier("serj/intent-classifier")
    print(m.predict("Hey, after recent changes, I want to cancel subscription, please help.", "OPTIONS: refund\n cancel subscription\n damaged item\n return_item\n", "Company", "Products and subscriptions"))


@task
def no_helper_classes_example(c, model_name="serj/intent-classifier"):
    device = "cuda"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    device = device
    input_text = '''
    Company name: Company, is doing: products and subscription 
    Customer: Hey, after recent changes, I want to cancel subscription, please help.
    END MESSAGE
    Choose one topic that matches customer's issue.
    OPTIONS: 
    refund 
    cancel subscription 
    damaged item 
    return_item
    Class name: "
    
    '''
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate the output
    output = model.generate(input_ids)

    # Decode the output tokens
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)