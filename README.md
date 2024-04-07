# Description
This project aims to build a small language model for few-shot classification tasks. 
This includes all of the infrastracture needed to train it, run predictions and compare methodologies. 
Overall I'm interested in researching how far can we push the performance of small language models. 

Current focus - create the best zero shot classifier on a T5 architecture (Flan-T5-Base).

Model link: https://huggingface.co/Serj/intent-classifier

Throughout the project I will refer to this model as intent-classifier, and it wasn't fine tuned on Atis.

# Zero shot Leaderboard on Atis 

| Model name | Weighted F1 AVG | Num parameters |
| -------- | ------- | -------|
| intent-classifier | 0.69 | 250M |
| bart-mnli | 0.4 | 406M |
| flan-t5-base | 0.006 | 250M |


# Setup
Create virtualenv 
```
$ pip install -r requirements.txt
```

# Structure 
The structure follows a common pattern of machine learning projects.
Separating each major task to a different script.

More information and examples about the pattern:
https://serj-smor.medium.com/mlops-without-magic-100365b22d1a

# Hugging Face Model Link
https://huggingface.co/Serj/intent-classifier

# Train 
```
$ inv train
```

# Run a prediction
predict.py
