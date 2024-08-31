from setuptools import setup, find_packages


setup(
    name="intent_classifier",
    version="0.0.1",
    packages=["intent_classifier"],
    description="This library enables to mix and match classification components for prompt based classifier.",
    install_requires=["transformers", "datasets", "evaluate", "sacrebleu", "torch", "sentencepiece"]
)