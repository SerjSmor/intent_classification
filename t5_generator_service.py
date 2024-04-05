import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

from app.utils import build_prompt

# Load the pre-trained T5 model and tokenizer
# model_name = "google/flan-t5-base"
# model_name = "google/flan-t5-xl"
# model_name = "google/flan-t5-large"
# model_name = "models/flan_t5_base_generator"
# model_name = "models/flan-t5-small"
model_name = "Serj/intent-classifier"
model_suffix = model_name.split("/")[1]
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Prompt containing the taxonomy
# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Classify each example
results = []

examples = [
    "Hi, I'm trying to use my gift card to pay for my order, but it's not working. Can you assist?",
    "Hello, I just noticed a discrepancy in the total amount charged on my credit card statement. Can you help me understand?",
    "Hey there! I received a promo code for a discount, but it's not being applied to my order. Can you check?",
    "I want to cancel subscription",
    "I want a refund",
    "Hey",
    "what is your name",
    "I hate this thing, it never works, how about just fixing it once?",
    "Where is the box?",
    "When should the package arrive?",
    "I don't get it, I really don't how many times I can ask for the same thing. Can I speak to someone who actually has some sense? Every day is the same with you. This shipment should have been here by now!"
    # Add more examples here
]
prompt = '''
1. Billing inquiries 
2. Gift Cards payment issues
3. Promo Codes and discounts
4. Refund requests
5. Cancel subscription
6. General inquiries
7. Broken item
8. Missing item
9. Order status
10. Other
'''

for example in tqdm(examples):
    # Concatenate the prompt with the example
    input_text = build_prompt(example, prompt, "Regular Online Business", "Regular stuff")
    # print(input_text)
    # Tokenize the concatenated inp_ut text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate the output
    output = model.generate(input_ids)

    # Decode the output tokens
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Append the prediction to the list
    results.append({"text": example, "prediction": decoded_output})

# Create a DataFrame to store the results
df = pd.DataFrame(data=results)
print(df.to_string())

df.to_csv(f"data/{model_suffix}_predictions.csv")