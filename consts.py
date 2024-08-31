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
LOCAL_FLAN_T5_SMALL = "models/flan-t5-small"

TEST_CSV = "data/test.csv"

# hyperparams
TWENTY_EPOCHS = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_EXPERIMENT_NAME = "test"
DEFAULT_WARMUP_STEPS = 500
DEFAULT_WEIGHT_DECAY = 0.01

# datasets

# classification reports

TEST_SET_PREDICTIONS = "results/predictions_test.csv"
ATIS_PREDICTIONS_CSV = "results/atis_predictions.csv"
ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV = "results/atis_classification_report.csv"
TEST_SET_CLASSIFICATION_REPORT_CSV = "results/test_set_classification_report.csv"
BLEU_RESULTS_JSON = "results/bleu_results.json"
BLEU_CLASSIFICATION_RESULTS = "results/bleu_classification.json"
BLEU_PREDICTIONS_CSV = "results/bleu_predictions.csv"
HUNDRED_COMPANIES_DATASET = "data/100_companies_1_shot.json"
BANKING77_PREDICTIONS_CSV = "data/banking77_predictions.csv"
BANKING77_CLASSIFICATION_REPORT_CSV = "data/banking77_classification_report.txt"
# best model
BEST_MODEL_W87 = "archived_experiments/f1w_0.87_m:flan-t5-base_e:2_b:8_t:06232024-23:03:05_ncs:True_ups:False_udl:False_nnp:True_best/models/flan-t5-base"
BEST_MODEL_W90 = "archived_experiments/f1w_0.90_m:flan-t5-base_e:2_b:8_t:06282024-11:55:57_ncs:True_ups:False_udl:False_nnp:True_eet:False/models/flan-t5-base"

BEST_LONG_FORMAT_SINGLE_TASK_EXTRACTION_MODEL = "archived_experiments/with_long_prompt_f1w_0.00_no_classification_m:flan-t5-base_e:9_b:8_t:06282024-14:39:41/models/flan-t5-base"
BEST_SHORT_FORMAT_SINGLE_TASK_EXTRACTION_MODEL = "archived_experiments/with_short_prompt_f1w_0.00_no_classification_m:flan-t5-base_e:9_b:8_t:06282024-14:42:55/models/flan-t5-base"
ANOTHER_MODEL = "archived_experiments/f1w_0.00_no_classification_m:flan-t5-base_e:5_b:8_t:06302024-08:46:19/models/flan-t5-base"

BEST_ENTITY_EXTRACTION_MODEL_COMBINED = "archived_experiments/f1w_0.88_m:flan-t5-base_e:3_b:8_t:06292024-14:04:22_ncs:True_ups:False_udl:False_nnp:True_eet:True/models/flan-t5-base"

# best banking model
BEST_BANKING_MODEL = "archived_experiments/b0.00_f1w_0.86_m:flan-t5-base_e:3_b:8_t:07122024-14:28:46_ncs:True_ups:False_udl:False_nnp:True_eet:False/models/flan-t5-base"