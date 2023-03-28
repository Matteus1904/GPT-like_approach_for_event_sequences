# Self-supervised learning for Event Sequences on synthetic task of next item prediction (GPT-approach based module)

This repository contains a code implementation of the final project **Self-supervised learning for Event Sequences on synthetic task of next item prediction (GPT-approach based module)** for Machine Learning 2023 course.

__Project team:__

1) Egor Fadeev
2) Alexander Ganibaev
3) Matvey Lukyanov
4) Aleksandr Yugay

## Idea:

![alt text](/pics/transfer.png)

Self-supervised learning is a powerful technique for leveraging large amounts of unlabeled data to improve the performance of machine learning models, particularly in domains where labeled data is scarce or expensive to obtain. In this project, we focus on self-supervised learning applied to event sequences, specifically transaction data, and explore the use of two different pre-training approaches for obtaining embeddings: classical representations and contrastive representations. We demonstrate that both embedding models are viable for downstream tasks, specifically in predicting the next merchant category code (MCC) of a transaction. Our experiments show that the pre-trained contrastive embeddings perform better on less stable data, while the pre-trained representation embeddings suit better for homogeneous transaction data. These findings can help guide the selection of pre-training approaches for transactional data, and our work opens up opportunities for further exploration of self-supervised learning in other domains.

## Repo description

* [`sber_experiments.ipynb`](sber_experiments.ipynb) — running experiments on Sber datasets <a target="_blank" href="https://colab.research.google.com/github/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/sber_experiments.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* [`rosbank_experiments.ipynb`](rosbank_experiments.ipynb) — running experiments on Rosbank datasets <a target="_blank" href="https://colab.research.google.com/github/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/rosbank_experiments.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* [`sber_notebook.ipynb`](https://github.com/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/eda/sber_notebook.ipynb) — visualizations of Sber dataset

* [`rosbank_notebook.ipynb`](https://github.com/Matteus1904/GPT-like_approach_for_event_sequences/blob/master/eda/rosbank_notebook.ipynb) — visualizations of Rosbank dataset

* [`models.py`](/models.py) — model pipeline, including its layers, architecture and etc

* [`dataset.py`](/dataset.py) — data preprocessing for model usage


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

## Run experiments
To run the default experiments, use the following command:
```
python main.py
```
It will save the results into `results/stats_contr_{dataset}_config_{config_name}.csv`.

You can run the model with your own hyperparameters — to do that, you can either change the corresponding values in an existing config in `config.ini`, or create your own config in `config.ini` using the same data format and add it after the existing ones.
