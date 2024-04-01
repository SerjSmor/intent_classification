from invoke import task

from preprocess import preprocess_simple_dataset


@task
def train_base(c):
    model_name = "google/flan-t5-base"
    c.run("python preprocess.py")
    c.run(f"python t5_generator_trainer.py --model-name {model_name} -e 20")
    c.run(f"python predict.py --model-name models/flan-t5-base -p data/test.csv")
    c.run("python results.py -p data/predictions_test.csv")

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
