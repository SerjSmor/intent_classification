# Description
This project aims to build a small language model for few-shot classification tasks. 
This includes all of the infrastracture needed to train it, run predictions and compare methodologies. 
Overall I'm interested in researching how far can we push the performance of small language models. 

Current focus - create the best zero shot classifier on a T5 architecture (Flan-T5-Base).

Model link: https://huggingface.co/Serj/intent-classifier

Throughout the project I will refer to this model as intent-classifier, and it wasn't fine tuned on Atis.

# Zero shot Leaderboard on Atis 

| Model name             | Weighted F1 AVG | Accuracy | Num parameters |
|------------------------|-----------------|----------|---------------|
| intent-classifier      | 0.7             | 0.63     | 248M          |
| setfit (BAAI/bge-small-en-v1.5) | 0.687            | 0.58     | 33.4M         |
| bart-mnli              | 0.4             | 0.32     | 406M          |
| flan-t5-base           | 0.006           | 0.01     | 248M          |


# Setup
Create virtualenv 
```
$ pip install -r requirements.txt
```

# Project Structure 
The structure follows a common pattern of machine learning projects.
Separating each major task to a different script.

More information and examples about the pattern:
https://serj-smor.medium.com/mlops-without-magic-100365b22d1a

# Hugging Face Model Link
https://huggingface.co/Serj/intent-classifier

# Run a prediction
Once you install requirements and specifically 'invoke' library, you can start running the tasks that are defined in 'tasks.py'.
Invoke creates an easy CMD (command line) interface using @task decorators.

More on python invoke: https://www.pyinvoke.org/

```commandline
$ inv no-helper-classes-example
```
If you don't want to use Invoke, you can use the following code:

```
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
```


# Train 
```
$ inv train
```

