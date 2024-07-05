import os.path
import random
from datetime import datetime
import json
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from invoke import task, Context
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from tqdm import tqdm

from consts import NON_ATIS_DATASETS, ALL_DATASETS, FLAN_T5_BASE, FLAN_T5_LARGE, FLAN_T5_SMALL, \
    ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV, BEST_MODEL_W87, BEST_ENTITY_EXTRACTION_MODEL_COMBINED, BEST_MODEL_W90, \
    BLEU_RESULTS_JSON
# app
from dataset import load_main_dataset
from preprocess import preprocess_simple_dataset, preprocess
from app.model import IntentClassifier
from app.utils import get_model_suffix, compute_metric_with_tokenizer
from app.atis.utils import ATIS_INTENT_MAPPING as intent_mapping, test_atis_dataset
from results import calculate_bleu_file
from t5_generator_trainer import DEFAULT_WARMUP_STEPS, DEFAULT_WEIGHT_DECAY


@dataclass
class Summary:
    batch_size: int

@task
def preprocess_dataset(c):
    c.run("python preprocess.py")

@task
def train(c, is_preprocess=False, model_name=FLAN_T5_BASE, epochs=6, batch_size=8,
          per_dataset=False, name="", test_atis=False, dataset_names=ALL_DATASETS, peft=False,
          warmup_steps=DEFAULT_WARMUP_STEPS, weight_decay=DEFAULT_WEIGHT_DECAY, no_company_specific=False,
          use_default_labels=False, use_positive=False, no_number_prompt=False, entity_extraction_task=False):
    if is_preprocess:
        c.run("python preprocess.py")

    print(f"(train) experiment name: {name}")
    test_atis_str = "--test-atis" if test_atis else ""
    peft_str = " --peft " if peft else ""
    no_company_specific_str = "--no-company-specific" if no_company_specific else ""
    use_positive_str = "--use-positive-samples" if use_positive else ""
    no_number_prompt_str = "--no-number-prompt" if no_number_prompt else ""
    use_default_labels_str = "--use-default-labels" if use_default_labels else ""
    entity_extraction_task_str = "--entity-extraction-task" if entity_extraction_task else ""

    commands = f"--model-name {model_name} -e {epochs} -bs {batch_size} --name {name} {test_atis_str} " \
               f"--dataset-names {dataset_names} {peft_str} --warmup-steps {warmup_steps} --weight-decay {weight_decay} " \
               f" {no_company_specific_str} {use_positive_str} {no_number_prompt_str} {use_default_labels_str} " \
               f" {entity_extraction_task_str}"
    print(commands)

    c.run(f"python t5_generator_trainer.py {commands}")
    # model_suffix = model_name.split("/")[1]
    # c.run(f"python predict.py --model-name models/{model_suffix} -p data/test.csv")
    # per_dataset_str = "" if not per_dataset else " -pd "
    # c.run(f"python results.py -p results/predictions_test.csv {per_dataset_str}")


@task
def _train_atis_zero_shot(c, model_name=FLAN_T5_BASE, epochs=15, batch_size=8, name="", dataset_names="", peft=False,
                          warmup_steps=DEFAULT_WARMUP_STEPS, weight_decay=DEFAULT_WEIGHT_DECAY,
                          no_company_specific=True, use_positive=False, use_default_labels=False,
                          no_number_prompt=False, entity_extraction_task=False, no_classification_task=False):
    dataset = load_main_dataset()
    # remove atis
    filtered_dataset = [var for var in dataset if var["Company Name"] in dataset_names]

    preprocess(filtered_dataset, no_number_prompt=no_number_prompt, entity_extraction_task=entity_extraction_task,
               no_classification_task=no_classification_task)
    test_atis = False if no_classification_task else True

    print(f"(_train_atis)experiment name: {name}")
    train(c, is_preprocess=False, model_name=model_name, epochs=epochs,
          batch_size=batch_size, name=name, test_atis=test_atis, dataset_names=dataset_names, peft=peft,
          warmup_steps=warmup_steps, weight_decay=weight_decay, no_company_specific=no_company_specific,
          use_positive=use_positive, no_number_prompt=no_number_prompt, use_default_labels=use_default_labels,
          entity_extraction_task=entity_extraction_task)


@task
def preprocess_task(c, entity_extraction_task=False, no_classification_task=False):
    dataset = load_main_dataset()
    # remove atis
    dataset_names = NON_ATIS_DATASETS.split()
    filtered_datasets = [var for var in dataset if var["Company Name"] in dataset_names]
    preprocess(filtered_datasets, no_number_prompt=True, entity_extraction_task=entity_extraction_task,
               no_classification_task=no_classification_task)

@task
def test_atis(c, model_path="", use_default_labels=False, no_number_prompt=False):
    # atis test set
    if not model_path:
        model_path = "archived_experiments/f1w_0.87_m:flan-t5-base_e:2_b:8_t:06232024-23:03:05_ncs:True_ups:False_udl:False_nnp:True_best/models/flan-t5-base"
    test_atis_dataset(model_path, no_company_specific=True, save_file=True, use_default_labels=use_default_labels, no_number_prompt=no_number_prompt)



@task
def atis_pipeline(c, name="", model_name=FLAN_T5_BASE, epochs=15, batch_size=8, dataset_names=NON_ATIS_DATASETS,
                  peft=False, warmup_steps=DEFAULT_WARMUP_STEPS, weight_decay=DEFAULT_WEIGHT_DECAY,
                  no_company_specific=True, use_positive=False, should_archive=False, use_default_labels=False,
                  no_number_prompt=False, entity_extraction_task=False, no_classification_task=False):

    c.run("rm -rf models/*")
    c.run("rm -rf results/*")

    if len(name) == 0:
        model_suffix = get_model_suffix(model_name)
        date_time = datetime.now().strftime("%m%d%Y-%H:%M:%S")
        print("date and time:", date_time)
        name = f"m:{model_suffix}_e:{epochs}_b:{batch_size}_t:{date_time}_ncs:{no_company_specific}_ups:" \
               f"{use_positive}_udl:{use_default_labels}_nnp:{no_number_prompt}_eet:{entity_extraction_task}"
        if no_classification_task:
            name = f"no_classification_m:{model_suffix}_e:{epochs}_b:{batch_size}_t:{date_time}"
    print(f"experiment name: {name}")
    dataset_names = dataset_names.split()
    _train_atis_zero_shot(c, model_name=model_name, epochs=epochs, batch_size=batch_size, name=name,
                          dataset_names=dataset_names, peft=peft, warmup_steps=warmup_steps, weight_decay=weight_decay,
                          no_company_specific=no_company_specific, use_positive=use_positive, use_default_labels=use_default_labels,
                          no_number_prompt=no_number_prompt, entity_extraction_task=entity_extraction_task, no_classification_task=no_classification_task)

    pipeline_args = {"model name": model_name, "epochs": epochs, "batch size": batch_size, "name": name}
    with open("data/train_args.txt", "w") as f:
        json.dump(pipeline_args, f)

    if os.path.exists(ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV):
        atis_classification_report_df = pd.read_csv(ATIS_TEST_SET_CLASSIFICATION_REPORT_CSV)
        # last row is weighted
        f1_weighted = atis_classification_report_df["f1-score"].iloc[-1]
    else:
        f1_weighted = 0

    print(f"atis weighted f1: {f1_weighted}")
    if f1_weighted > 0.77:
        # print(f"starting to archive weighted f1: {f1_weighted}, model: {model_name}")
        name = f"f1w_{f1_weighted:.2f}_{name}"

    if os.path.exists(BLEU_RESULTS_JSON):
        bleu_results = json.load(open(BLEU_RESULTS_JSON, "r"))
        name = f"b{bleu_results['score']:.2f}_{name}"

    if should_archive:
        archive(c, name)
@task
# finding the optimal epoch number
def find_epoch_experiments(c, model_name, no_company_specific=True, use_positive=False, datasets=NON_ATIS_DATASETS, no_number_prompt=False,
                           use_default_labels=False, entity_extraction_task=False, no_classification_task=False, should_archive=False):
    small_model_epochs = list(range(1, 9))
    base_model_epochs = list(range(1, 6))
    # base_model_epochs = list(range(9, 10))
    large_model_epochs = list(range(1, 4))

    epochs = small_model_epochs
    batch_size = 16
    if model_name == FLAN_T5_BASE:
        epochs = base_model_epochs
        batch_size = 8
    elif model_name == FLAN_T5_LARGE:
        epochs = large_model_epochs
        batch_size = 2

    for epoch in epochs:
        atis_pipeline(c, model_name=model_name, epochs=epoch, batch_size=batch_size,
                      no_company_specific=no_company_specific, use_positive=use_positive, dataset_names=datasets,
                      no_number_prompt=no_number_prompt, use_default_labels=use_default_labels,
                      entity_extraction_task=entity_extraction_task, no_classification_task=no_classification_task,
                      should_archive=should_archive)


@task
# finding the optimal epoch number
def best_epoch_experiments(c, model_name, no_company_specific=True, use_positive=False, datasets=NON_ATIS_DATASETS,
                           should_archive=False):
    small_model_epochs = list(range(7, 9))
    base_model_epochs = list(range(2, 4))
    large_model_epochs = list(range(1, 2))

    epochs = small_model_epochs
    batch_size = 16
    if model_name == FLAN_T5_BASE:
        epochs = base_model_epochs
        batch_size = 8
    elif model_name == FLAN_T5_LARGE:
        epochs = large_model_epochs
        batch_size = 2

    for epoch in epochs:
        atis_pipeline(c, model_name=model_name, epochs=epoch, batch_size=batch_size,
                      no_company_specific=no_company_specific, use_positive=use_positive, dataset_names=datasets,
                      should_archive=should_archive)

@task
def create_best_models(c):
    best_epoch_experiments(c, FLAN_T5_SMALL, datasets=NON_ATIS_DATASETS, should_archive=True)
    best_epoch_experiments(c, FLAN_T5_BASE, datasets=NON_ATIS_DATASETS, should_archive=True)
    best_epoch_experiments(c, FLAN_T5_LARGE, datasets="Clinc_oos_41_classes", should_archive=True)

@task
def train_best_base_model(c, use_default_labels=False, entity_extraction_task=False):
    find_epoch_experiments(c, FLAN_T5_BASE, no_company_specific=True, use_default_labels=use_default_labels,
                           no_number_prompt=True, entity_extraction_task=entity_extraction_task)

@task
def datasets_plot(c):

    dataset_group = ["Clinc_oos_41_classes", "Clinc_oos_41_classes Online_Banking", "Clinc_oos_41_classes Online_Banking Pizza_Mia"]

    # for num_datasets in range(1, len(dataset_name_array)):
    for group in dataset_group:
        # dataset_names = random.sample(dataset_name_array, num_datasets)
        # dataset_names_str = " ".join(dataset_names)
        find_epoch_experiments(c, FLAN_T5_SMALL, no_company_specific=True, no_number_prompt=True, datasets=group)
        find_epoch_experiments(c, FLAN_T5_BASE, no_company_specific=True, no_number_prompt=True, datasets=group)
        find_epoch_experiments(c, FLAN_T5_LARGE, no_company_specific=True, no_number_prompt=True, datasets=group)

@task
def base_plot(c, use_default_labels=False, no_number_prompt=False):
    find_epoch_experiments(c, FLAN_T5_SMALL, no_company_specific=True, use_default_labels=use_default_labels,
                           no_number_prompt=no_number_prompt)
    find_epoch_experiments(c, FLAN_T5_BASE, no_company_specific=True, use_default_labels=use_default_labels,
                           no_number_prompt=no_number_prompt)
    find_epoch_experiments(c, FLAN_T5_LARGE, no_company_specific=True, use_default_labels=use_default_labels,
                           no_number_prompt=no_number_prompt)

@task
def train_entity_extraction(c):
    # find_epoch_experiments(c, FLAN_T5_BASE, entity_extraction_task=True, no_classification_task=True, should_archive=True)
    # find_epoch_experiments(c, FLAN_T5_BASE, entity_extraction_task=True, no_classification_task=True,
    #                        should_archive=True)
    find_epoch_experiments(c, FLAN_T5_BASE, entity_extraction_task=True, no_classification_task=False,
                           should_archive=True, use_default_labels=True)


@task
def find_optimal_weight_decay(c):
    pass

@task
def find_optimal_warmup_steps(c, warmup_step):
    pass

# lets run

@task
def archive(c, name):
    experiment_path = f"archived_experiments/{name}"
    c.run(f"mkdir {experiment_path}")
    c.run(f"cp -R data {experiment_path}/")
    c.run(f"cp -R models {experiment_path}/")
    c.run(f"cp -R results {experiment_path}/")


@task
def train_small(c):
    # train(c, model_name="google/flan-t5-small")
    # train(c, model_name="google/flan-t5-small", epochs=20)
    train(c, model_name="flan-t5-small", epochs=20)

@task
def test_atis_best_model(c, model_path=BEST_MODEL_W87, use_default_labels=False):
    test_atis_dataset(model_path, save_file=True, use_default_labels=use_default_labels,
                      no_number_prompt=True,
                      no_company_specific=True)

@task
def test_entity_extraction_model(c, model_path=BEST_ENTITY_EXTRACTION_MODEL_COMBINED):
    test_atis_dataset(model_path, save_file=True, use_default_labels=False, no_number_prompt=True, no_company_specific=True,
                      extract_entities=True)

@task
def test_single_task_entity_extraction_model(c, model_path=BEST_MODEL_W90):
    test_atis_dataset(model_path, save_file=True, use_default_labels=True, no_number_prompt=True, no_company_specific=True,
                      extract_entities=True)

@task
def test_bleu(c, model_path=BEST_ENTITY_EXTRACTION_MODEL_COMBINED, ):
    res = calculate_bleu_file(model_path=model_path)
    print(res)


@task
def simple_dataset_base(c):
    preprocess_simple_dataset()
    c.run("python predict.py --model-name models/flan-t5-base -p data/simple_dataset.csv")
    c.run("python results.py -p data/predictions_simple_dataset.csv")

@task
def simple_dataset_small(c):
    preprocess_simple_dataset()
    c.run("python predict.py --model-name models/flan-t5-small -p data/simple_dataset.csv")
    c.run("python results.py -p data/predictions_simple_dataset.csv")


@task
def predict_test_hf_model(c):
    c.run("python preprocess.py")
    c.run("python predict.py --model-name serj/intent-classifier -p data/test.csv")
    c.run("python results.py -p data/predictions_test.csv -pd")


@task
def gui(c):
    c.run("streamlit run app.py")



if __name__ == '__main__':
    dataset = load_main_dataset()
    # remove atis
    dataset_names = NON_ATIS_DATASETS.split()
    filtered_dataset = [var for var in dataset if var["Company Name"] in dataset_names]

    preprocess(filtered_dataset, no_number_prompt=True, entity_extraction_task=True,
               no_classification_task=False)