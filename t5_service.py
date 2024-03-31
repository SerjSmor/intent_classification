import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
# Load the pre-trained T5 model and tokenizer
model_name = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Examples to classify
examples = [
    "Hi, I'm trying to use my gift card to pay for my order, but it's not working. Can you assist?",
    "Hello, I just noticed a discrepancy in the total amount charged on my credit card statement. Can you help me understand?",
    "Hey there! I received a promo code for a discount, but it's not being applied to my order. Can you check?"
    # Add more examples here
]

# Prompt containing the taxonomy
prompt = """
OPTIONS
1. Order Placement
2. Order Tracking
3. Menu and Ingredients
4. Delivery Issues
5. Payment and Billing
6. Technical Support
7. Quality Concerns
8. General Inquiry
9. Compliments and Feedback
10. Special Requests
"""
# Examples and Classes:
# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Classify each example
predictions = []
for example in examples:
    # Concatenate the prompt with the example
    input_text = prompt + f"\n{example}"

    # Tokenize the concatenated input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate the output
    output = model.generate(input_ids)
    # Decode the output tokens
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Append the prediction to the list
    predictions.append(output)
# Create a DataFrame to store the results
df = pd.DataFrame({"Example": examples, "Prediction": predictions})
print(df)
