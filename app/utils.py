

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

