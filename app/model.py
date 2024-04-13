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
        decoded_output = self._predict(input_text)

        return decoded_output

    def _predict(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(
            self.device)
        # Generate the output
        output = self.model.generate(input_ids)
        # Decode the output tokens
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output

    def raw_predict(self, text) -> str:
        return self._predict(text)

if __name__ == '__main__':
    m = IntentClassifier("serj/intent-classifier")
    print(m.predict("Hey, after recent changes, I want to cancel subscription, please help.",
                    "OPTIONS:\n refund\n cancel subscription\n damaged item\n return item\n", "Company",
                    "Products and subscriptions"))