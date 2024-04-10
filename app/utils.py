

def build_prompt(text, prompt="", company_name="", company_specific=""):
    if company_name == "Pizza Mia":
        company_specific = "This company is a pizzeria place."
    if company_name == "Online Banking":
        company_specific = "This company is an online banking."

    return f"Company name: {company_name} is doing: {company_specific}\nCustomer: {text}.\nEND MESSAGE\nChoose one topic that matches customer's issue.\n{prompt}\nClass name: "