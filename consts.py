FULL_PROMPT = '''
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
'''

GENERATOR_LABELS = "generator_labels"
PROMPT_OPTIONS = "prompt_options"
GENERATOR_TEXT = "generator_text"
DEFAULT_PREDICTION_CSV = "data/predictions.csv"
GENERATOR_TEXT_NO_COMPANY = "generator_text_no_company"

NON_ATIS_DATASETS = "Pizza_Mia Online_Banking Clinc_oos_41_classes"
ALL_DATASETS = f"{NON_ATIS_DATASETS} Atis_Airlines"

# models
FLAN_T5_BASE = "google/flan-t5-base"
FLAN_T5_LARGE = "google/flan-t5-large"
FLAN_T5_SMALL = "google/flan-t5-small"

# hyperparams
TWENTY_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_EXPERIMENT_NAME = ""
DEFAULT_WARMUP_STEPS = 500
DEFAULT_WEIGHT_DECAY = 0.01

# datasets

# classification reports

ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV = "results/atis_classification_report.csv"
TEST_SET_CLASSIFICATION_REPORT_CSV = "results/test_set_classification_report.csv"