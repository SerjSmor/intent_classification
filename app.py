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
# best_model_path = "archived_experiments/f1w_0.87_m:flan-t5-base_e:2_b:8_t:06232024-23:03:05_ncs:True_ups:False_udl:False_nnp:True_best/models/flan-t5-base"
# best_model_path_ett = "archived_experiments/f1w_0.88_m:flan-t5-base_e:3_b:8_t:06282024-09:36:50_ncs:True_ups:False_udl:False_nnp:True_eet:True/models/flan-t5-base"
# best_model_path_ett = "archived_experiments/with_short_prompt_f1w_0.00_no_classification_m:flan-t5-base_e:9_b:8_t:06282024-14:42:55/models/flan-t5-base"
best_model_path_ett = "archived_experiments/with_long_prompt_f1w_0.00_no_classification_m:flan-t5-base_e:9_b:8_t:06282024-14:39:41/models/flan-t5-base"
# model_path = f"archived_experiments/f1w_0.87_m:flan-t5-base_e:2_b:8_t:06232024-23:03:05_ncs:True_ups:False_udl:False_nnp:True_best/models/flan-t5-base"
# model_path = "models/flan-t5-small"

# model_path = best_model_path


model_path = best_model_path_ett
model = load_model(model_path)
st.title("Input")
# text = "Hey, I want to stop my subscription. Can I do that?"
text = "How do I get my money back?"
input = st.text_area("text", value=text)
prompt_options = st.text_area("prompt_options", value="# renew subscription # account deletion # cancel subscription # resume subscription # refund requests # other # general # item damaged # malfunction # hello # intro # question")
is_clicked = st.button("Submit")
if is_clicked:
    output = model.predict(input, prompt_options, print_input=True)
    st.write(output)

input = st.text_area("entity extraction", value="I just understood that we're having a trouble logging in, the button is missing")
is_clicked_entity = st.button("Submit entity")
if is_clicked_entity:
    output = model.raw_predict(f"Extraction %% {input} END. What is the main entity action separated by $?", print_input=True)
    st.write(output)





