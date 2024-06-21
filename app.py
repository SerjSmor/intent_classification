import streamlit as st

from app.model import IntentClassifier
from consts import FLAN_T5_SMALL, FLAN_T5_BASE


@st.cache_resource
def load_model(model_name):

    st.write(f"model path: {model_name}")
    m = IntentClassifier(model_name=model_name)
    return m

# model_path = f"archived_experiments/f1w_0.73_m:flan-t5-base_e:3_b:8_t:06182024-06:01:44_ncs:True_ups:False/models/flan-t5-base"
# model_path = f"archived_experiments/f1w_0.77_m:flan-t5-base_e:3_b:8_t:06182024-09:37:12_ncs:True_ups:False/models/flan-t5-base"
# model_path = f"archived_experiments/f1w_0.79_m:flan-t5-small_e:7_b:16_t:06182024-05:52:53_ncs:True_ups:False/models/flan-t5-small"
model_path = "models/flan-t5-small"


model = load_model(model_path)
st.title("Input")
# text = "Hey, I want to stop my subscription. Can I do that?"
text = "How do I get my money back?"
input = st.text_area("text", value=text)
prompt_options = st.text_area("prompt_options", value="% renew subscription % account deletion % cancel subscription % resume subscription % refund requests %")
is_clicked = st.button("Submit")
if is_clicked:
    output = model.predict(input, prompt_options)
    st.write(output)






