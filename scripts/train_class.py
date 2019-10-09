import argparse
import csv
import math
import os
import re
import sys
sys.path.append("..")

import gluonnlp
import jieba.posseg
import mxnet as mx
import numpy as np
from gluonnlp.data import train_valid_split
from gluonnlp.model import BeamSearchScorer
from mxboard import *
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss
from numpy import random
from tqdm import tqdm

from data.scripts.dataloader import ClassDataLoader, DatasetAssiantTransformer
from data.scripts.dataset import ClassDataset, ClassTestDataset
from data.scripts.vocab import load_label_vocab
from models.BertClass import BertClass
from models.MaskedCELoss import MaskedCELoss
from utils import config_logger, get_entities


np.random.seed(100)
random.seed(100)
mx.random.seed(10000)


def train_and_valid(ch_bert, model, ch_vocab, train_dataiter, dev_dataiter, trainer, finetune_trainer, epochs, loss_func, ctx, lr, batch_size, params_save_step, params_save_path_root, eval_step, log_step, check_step, logger, num_train_examples, warmup_ratio):
    batches = len(train_dataiter)

    num_train_steps = int(num_train_examples / batch_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    global_step = 0

    dev_bleu_score = 0

    for epoch in range(epochs):
        for content, token_types, valid_len, label, example_id in train_dataiter:
            # learning rate schedule
            if global_step < num_warmup_steps:
                new_lr = lr * global_step / num_warmup_steps
            else:
                non_warmup_steps = global_step - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)

            content = content.as_in_context(ctx)
            token_types = token_types.as_in_context(ctx)
            valid_len = valid_len.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                output = model(content, token_types, valid_len)
                loss_mean = loss_func(output, label)
                loss_mean = nd.sum(loss_mean) / batch_size
            loss_mean.backward()
            loss_scalar = loss_mean.asscalar()

            trainer.step(1)
            finetune_trainer.step(1)

            if global_step and global_step % log_step == 0:
                acc = nd.sum(nd.equal(nd.argmax(nd.softmax(
                    output, axis=-1), axis=-1), label)) / batch_size
                acc = acc.asscalar()
                logger.info("epoch:{}, batch:{}/{}, acc:{}, loss:{}, (lr:{}s)".format(epoch, global_step %
                                                                                      batches, batches, acc, loss_scalar, trainer.learning_rate))
            global_step += 1
        F1 = dev(ch_bert, model, ch_vocab, dev_dataiter, logger, ctx)
        if not os.path.exists(params_save_path_root):
            os.makedirs(params_save_path_root)
        model_params_file = params_save_path_root + \
            "model_step_{}_{}.params".format(global_step, F1)
        model.save_parameters(model_params_file)
        logger.info("{} Save Completed.".format(model_params_file))


def dev(ch_bert, model, ch_vocab, dev_dataiter, logger, ctx):
    TP_s = 0
    FP_s = 0
    FN_s = 0
    example_ids = []
    for content, token_types, valid_len, label, example_id in tqdm(dev_dataiter):
        example_ids.extend(example_id)
        content = content.as_in_context(ctx)
        token_types = token_types.as_in_context(ctx)
        valid_len = valid_len.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = model(content, token_types, valid_len)
        predict = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
        label = label.as_in_context(ctx)
        tp_s = int(nd.sum(nd.equal(predict, label)).asscalar())
        fp_s = int(nd.sum(nd.not_equal(predict, label) * nd.equal(label, 0)).asscalar())
        fn_s = int(nd.sum(nd.not_equal(predict, label) * nd.equal(label, 1)).asscalar())
        TP_s += tp_s
        FP_s += fp_s
        FN_s += fn_s

    P_s = TP_s / (TP_s + FP_s)
    R_s = TP_s / (TP_s + FN_s)
    F = (2 * P_s * R_s) / (P_s + R_s)

    logger.info("F:{}".format(F))
    return F


def predict(ch_bert, model, ch_vocab, test_dataiter, logger, ctx):
    example_ids = []
    sources = []
    pred_labels = []
    entities = []

    neg_example_ids = []
    neg_sources = []
    neg_pred_labels = []
    neg_entities = []

    for content, token_type, valid_len, source, entity, example_id in tqdm(test_dataiter):
        sources.extend(source)
        entities.extend(entity)

        content = content.as_in_context(ctx)
        token_type = token_type.as_in_context(ctx)
        valid_len = valid_len.as_in_context(ctx)

        outputs = model(content, token_type, valid_len)
        predicts = nd.argmax(nd.softmax(outputs, axis=-1), axis=-1).asnumpy().tolist()
        predicts = [int(label) for label in predicts]
        for label, eid, text, en in zip(predicts, example_id, source, entity):
            if label == 0:
                example_ids.append(eid)
                sources.append(text)
                pred_labels.append(label)
                entities.append(en)
            else:
                neg_example_ids.append(eid)
                neg_sources.append(text)
                neg_pred_labels.append(label)
                neg_entities.append(en)

    f = open("../results/result_only_neg.csv", "w", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["id", "nagative", "key_entity"])
    example_ids.extend(neg_example_ids)
    pred_labels.extend(neg_pred_labels)
    for ids, label in zip(example_ids, pred_labels):
        row = [ids, label, ""]
        writer.writerow(row)


def main(args):
    # init some setting
    # config logging
    log_path = os.path.join(args.log_root, '{}.log'.format(args.model_name))
    logger = config_logger(log_path)

    gpu_idx = args.gpu
    if not gpu_idx:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(gpu_idx - 1)
    logger.info("Using ctx: {}".format(ctx))

    # Loading vocab and model
    ch_bert, ch_vocab = gluonnlp.model.get_model(args.bert_model,
                                                 dataset_name=args.ch_bert_dataset,
                                                 pretrained=True,
                                                 ctx=ctx,
                                                 use_pooler=False,
                                                 use_decoder=False,
                                                 use_classifier=False)
    model = BertClass(bert=ch_bert,
                      max_seq_len=args.max_seq_len, ctx=ctx)
    logger.info("Model Creating Completed.")

    # init or load params for model
    if args.istrain:
        model.output_dense.initialize(init.Xavier(), ctx)
    else:
        model.load_parameters(args.model_params_path, ctx=ctx)
    logger.info("Parameters Initing and Loading Completed")

    model.hybridize()

    if args.istrain:
        # Loading dataloader
        assiant = DatasetAssiantTransformer(
            ch_vocab=ch_vocab, max_seq_len=args.max_seq_len)
        dataset = ClassDataset(args.train_file_path)
        train_dataset, dev_dataset = train_valid_split(dataset, valid_ratio=0.1)
        train_dataiter = ClassDataLoader(train_dataset, batch_size=args.batch_size,
                                         assiant=assiant, shuffle=True).dataiter
        dev_dataiter = ClassDataLoader(dev_dataset, batch_size=args.batch_size,
                                       assiant=assiant, shuffle=True).dataiter
        logger.info("Data Loading Completed")
    else:
        assiant = DatasetAssiantTransformer(
            ch_vocab=ch_vocab, max_seq_len=args.max_seq_len, istrain=args.istrain)
        test_dataset = ClassTestDataset(args.test_file_path)
        test_dataiter = ClassDataLoader(test_dataset, batch_size=args.batch_size,
                                        assiant=assiant, shuffle=True).dataiter

    # build trainer
    finetune_trainer = gluon.Trainer(ch_bert.collect_params(),
                                     args.optimizer, {"learning_rate": args.finetune_lr})
    trainer = gluon.Trainer(model.collect_params("dense*"), args.optimizer,
                            {"learning_rate": args.train_lr})

    loss_func = gloss.SoftmaxCELoss()

    if args.istrain:
        logger.info("## Trainning Start ##")
        train_and_valid(
            ch_bert=ch_bert, model=model, ch_vocab=ch_vocab,
            train_dataiter=train_dataiter, dev_dataiter=dev_dataiter, trainer=trainer, finetune_trainer=finetune_trainer, epochs=args.epochs,
            loss_func=loss_func, ctx=ctx, lr=args.train_lr, batch_size=args.batch_size, params_save_step=args.params_save_step,
            params_save_path_root=args.params_save_path_root, eval_step=args.eval_step, log_step=args.log_step, check_step=args.check_step,
            logger=logger, num_train_examples=len(train_dataset), warmup_ratio=args.warmup_ratio
        )
    else:
        predict(ch_bert=ch_bert, model=model, ch_vocab=ch_vocab,
                test_dataiter=test_dataiter, logger=logger, ctx=ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="new_entites_find")
    parser.add_argument("--train_file_path", type=str,
                        default="../data/Train_Data.csv")
    parser.add_argument("--test_file_path", type=str, default="../data/Test_Data.csv")
    parser.add_argument("--bert_model", type=str,
                        default="bert_12_768_12")
    parser.add_argument("--ch_bert_dataset", type=str,
                        default="wiki_cn_cased")
    parser.add_argument("--model_params_path", type=str,
                        default="../parameters/xxx.params")
    parser.add_argument("--istrain", type=bool,
                        default=True)
    parser.add_argument("--score", type=str,
                        default="0")
    parser.add_argument("--gpu", type=int,
                        default=1, help='which gpu to use for finetuning. CPU is used if set 0.')
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--bert_optimizer", type=str, default="bertadam")
    parser.add_argument("--train_lr", type=float, default=5e-5)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int,
                        default=64)
    parser.add_argument("--epochs", type=int,
                        default=3)
    parser.add_argument("--log_root", type=str, default="../logs/")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--eval_step", type=int, default=1000)
    parser.add_argument("--check_step", type=int, default=5)
    parser.add_argument("--params_save_step", type=int, default=300)
    parser.add_argument("--params_save_path_root", type=str, default="../parameters/")
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='ratio of warmup steps that linearly increase learning rate from '
                        '0 to target learning rate. default is 0.1')
    parser.add_argument("--max_seq_len", type=int,
                        default=256)
    # model parameters setting
    parser.add_argument("--model_dim", type=int,
                        default=768)
    parser.add_argument("--dropout", type=float,
                        default=0.1)

    args = parser.parse_args()

    main(args)
