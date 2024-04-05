from transformers import T5ForConditionalGeneration, T5Tokenizer

from app.utils import build_prompt


class IntentClassifier:
    def __init__(self, model_name="models/flan-t5-base", device="cuda"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = device

    def predict(self, text, prompt_options, company_name, company_portion) -> str:
        input_text = build_prompt(text, prompt_options, company_name, company_portion)
        # print(input_text)
        # Tokenize the concatenated inp_ut text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        # Generate the output
        output = self.model.generate(input_ids)

        # Decode the output tokens
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded_output

if __name__ == '__main__':
    m = IntentClassifier()
    print(m.predict("I have some stuff", "OPTIONS: 1. stuff, 2. stuff", "stuff", "stuff"))