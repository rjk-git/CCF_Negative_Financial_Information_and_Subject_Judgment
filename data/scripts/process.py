import csv
from itertools import islice
import jieba.posseg
import multiprocessing
from gluonnlp.data import Counter
import pandas as pd


def split_to_train_valid():
    valid_ratio = 0.1
    data = pd.read_csv("../Train_Data.csv", doublequote=True)
    print("总计：{}条样本".format(len(data)))
    train = data.sample(frac=0.9, random_state=0, axis=0)
    valid = data[~data.index.isin(train.index)]
    train.to_csv("../train.csv", index=False)
    valid.to_csv("../valid.csv", index=False)
    print("训练集：{}条样本".format(len(train)))
    print("验证集：{}条样本".format(len(valid)))


if __name__ == "__main__":
    split_to_train_valid()
