from mxnet.gluon import nn, rnn
import mxnet as mx
from mxnet import nd
import numpy as np


class BertClass(nn.Block):
    def __init__(self, bert, max_seq_len, ctx=mx.cpu(), **kwargs):
        super(BertClass, self).__init__(**kwargs)
        self.ctx = ctx
        self.max_seq_len = max_seq_len
        self.bert = bert
        self.output_dense = nn.Dense(2)

    def forward(self, content, token_types, valid_len):
        bert_output = self.bert(content, token_types, valid_len)
        bert_output = bert_output[:, 1, :]
        output = self.output_dense(bert_output)
        return output
