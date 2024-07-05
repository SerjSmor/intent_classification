import evaluate
import numpy as np

bleu = evaluate.load("sacrebleu")

ENTITY_EXTRACTION_DATASET_JSON = "data/entity_extraction_dataset.json"
ENTITY_EXTRACTION_TASK_PREFIX = "Extraction"

def build_entity_extraction_prompt(input_text):
    # prompt = f"{ENTITY_EXTRACTION_TASK_PREFIX} %% Customer: {input_text} END What is the main entity and action separated by $?"
    prompt = f"{ENTITY_EXTRACTION_TASK_PREFIX} %% Customer: {input_text}"
    return prompt

def build_prompt(text, prompt="", company_name="", company_specific="", no_company_specific=False):
    if company_name == "Pizza Mia":
        company_specific = "This company is a pizzeria place."
    if company_name == "Online Banking":
        company_specific = "This company is an online banking."

    if no_company_specific:
        return f"Topic %% Customer: {text}.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n{prompt}\nClass name: "

    else:
        return f"Topic %% Company name: {company_name} is doing: {company_specific}\nCustomer: {text}.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n{prompt}\nClass name: "

def get_model_suffix(model_name):
    return model_name.split("/")[1]


def compute_metric_with_tokenizer(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print("(compute metrics)")
        if isinstance(preds, tuple):
            preds = preds[0]

        result = calculate_bleu(labels, preds)
        return result

    def calculate_bleu(labels, preds):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # can't decode -100 (why?)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         some simpple post processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = {"score": -1}
        try:
            result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        except Exception as e:
            print(f"error: {e}")
        print(f"decoded labels: {decoded_labels[:5]}, decoded_preds: {decoded_preds[:5]}")

        return result

    return compute_metrics

