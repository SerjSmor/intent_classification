from invoke import task


@task
def train(c):
    c.run("python preprocess.py")
    c.run("python t5_trainer.py")

