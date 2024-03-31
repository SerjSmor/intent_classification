

def build_prompt(text, prompt):
    return f"Customer: {text}.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n{prompt}\n"