# Self-supervised learning for Event Sequences on synthetic task of next item prediction (GPT-approach based module)

This repository contains a code implementation of the final project **Self-supervised learning for Event Sequences on synthetic task of next item prediction (GPT-approach based module)** for Machine Learning 2023 course.

__Project team:__

1) Egor Fadeev
2) Alexander Ganibaev
3) Matvey Lukyanov
4) Aleksandr Yugay

## Idea:

![alt text](/pics/transfer.png)

We address the problem of self-supervised learning for event sequences. By means of self-supervised learning, we map the sophisticated information from transaction data to a low-dimensional space of fixed-length vectors. We implement embedding pre-training by optimization of representation loss. We implement embedding pre-training by optimization of contrastive loss. We use the obtained embeddings in downstream GPT-like tasks of the next MCC (item) prediction and the next MCC embedding prediction respectively. We experimentally find that usage of pre-trained embeddings of both types outperforms the baselines and the models without pre-training.

## Repo description

* In [`sber_experiments.ipynb`](sber_experiments.ipynb) run of experiments in Sber datasets <a target="_blank" href="https://colab.research.google.com/github/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/sber_experiments.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* In [`rosbank_experiments.ipynb`](rosbank_experiments.ipynb) run of experiments in Rosbank datasets <a target="_blank" href="https://colab.research.google.com/github/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/rosbank_experiments.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* In [`sber_notebook.ipynb`](https://github.com/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/eda/sber_notebook.ipynb) visualizations of Sber dataset

* In [`rosbank_notebook.ipynb`](https://github.com/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/eda/rosbank_notebook.ipynb) visualizations of Rosbank dataset

* In [`models.py`](/models.py) model pipeline, including its layers, architecture and etc

* In [`dataset.py`](/dataset.py) data preprocessing for model usage


## Prerequisites
```commandline
git clone https://github.com/Matteus1904/GPT-like_approach_for_event_sequences
pip install -r requirements.txt
mkdir -p data
curl -OL https://storage.yandexcloud.net/di-datasets/age-prediction-nti-sbebank-2019.zip
unzip -j -o age-prediction-nti-sbebank-2019.zip 'data/*.csv' -d data/sberbank
mv age-prediction-nti-sbebank-2019.zip data/
mkdir -p data/rosbank
curl -OL https://storage.yandexcloud.net/di-datasets/rosbank-ml-contest-boosters.pro.zip
unzip -j -o rosbank-ml-contest-boosters.pro.zip '*.csv' -d data/rosbank
mv rosbank-ml-contest-boosters.pro.zip data/rosbank/
```

## Model training
```
python main.py
```
