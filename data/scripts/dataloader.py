import json
import multiprocessing
import os
import re
import sys
sys.path.append("../data/scripts")

import gluon
import gluonnlp as nlp
from gluonnlp import Vocab
from gluonnlp.data import BERTSentenceTransform, BERTTokenizer, Counter
from mxnet import nd
from mxnet.gluon import data

import tokenization


class DatasetAssiantTransformer():
    def __init__(self, ch_vocab=None, max_seq_len=None, istrain=True):
        self.ch_vocab = ch_vocab
        self.max_seq_len = max_seq_len
        self.istrain = istrain
        self.tokenizer = BERTTokenizer(ch_vocab)  # 后面没用bert的tokenizer，感觉效果反而好些。

    def ClassProcess(self, *data):
        if self.istrain:
            example_id, source, label = data
        else:
            example_id, source, entity = data
        content = re.sub("[ \n\t\\n\u3000]", " ", source)
        content = re.sub("[?？]+", "？", content)
        content = [char for char in content]
        content = [self.ch_vocab(self.ch_vocab.cls_token)] + content
        content = self.ch_vocab(content)
        if self.max_seq_len and len(content) > self.max_seq_len:
            content = content[:self.max_seq_len]
        valid_len = len(content)
        token_type = [0] * valid_len
        if self.istrain:
            return content, token_type, valid_len, label, example_id
        else:
            return content, token_type, valid_len, source, entity, example_id


class ClassDataLoader(object):
    def __init__(self, dataset, batch_size, assiant, shuffle=False, num_workers=3, lazy=True):
        trans_func = assiant.ClassProcess
        self.istrain = assiant.istrain
        self.assiant = assiant
        self.dataset = dataset.transform(trans_func, lazy=lazy)
        self.batch_size = batch_size
        self.pad_val = assiant.ch_vocab[assiant.ch_vocab.padding_token]
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataloader = self._build_dataloader()

    def _build_dataloader(self):
        if self.istrain:
            batchify_fn = nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(pad_val=self.pad_val),
                nlp.data.batchify.Pad(pad_val=0),
                nlp.data.batchify.Stack(dtype="float32"),
                nlp.data.batchify.Stack(dtype="float32"),
                nlp.data.batchify.List()
            )
        else:
            batchify_fn = nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(pad_val=self.pad_val),
                nlp.data.batchify.Pad(pad_val=0),
                nlp.data.batchify.Stack(dtype="float32"),
                nlp.data.batchify.List(),
                nlp.data.batchify.List(),
                nlp.data.batchify.List()
            )
        dataloader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                     shuffle=self.shuffle, batchify_fn=batchify_fn,
                                     num_workers=self.num_workers)
        return dataloader

    @property
    def dataiter(self):
        return self.dataloader

    @property
    def data_lengths(self):
        return len(self.dataset)
