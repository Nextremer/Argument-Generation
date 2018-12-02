# -*- coding: utf-8 -*-
# created by Tomohiko Abe
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, optimizers, backends, serializers

import numpy as np
import nltk
import collections
from collections import defaultdict
import argparse
import pickle
import time
import random

from utils import *



def get_train_test_idxs(path):
    train_idxs = []
    test_idxs = []
    with open(path, 'r') as f:
        for line in f:
            if 'TRAIN' in line.strip():
                s = line.strip().replace('"', '').split(';')
                train_idxs.append(int(s[0].replace('essay', ''))-1)
            elif 'TEST' in line.strip():
                s = line.strip().replace('"', '').split(';')
                test_idxs.append(int(s[0].replace('essay', ''))-1)
    return train_idxs, test_idxs



class Decoder(chainer.Chain):

    def __init__(self, n_layers, n_units, dropout):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, dropout=dropout)

    def __call__(self, h, c, xs):
        return self.decoder(h, c, xs)


class Embed(chainer.Chain):

    def __init__(self, w2id, id2w, w2vec, n_units):
        n_vocab = len(w2id)

        init_W = [w2vec[id2w[i]] if id2w[i] in w2vec.keys() \
                  else np.random.normal(scale=np.sqrt(2./n_units), size=(n_units, )) \
                  for i, w in id2w.items()]
        init_W = np.asarray(init_W, dtype=np.float32)

        super(Embed, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=init_W)

    def __call__(self, xs):
        return self.embed(xs)



class PretrainedModel(chainer.Chain):

    def __init__(self, w2id, id2w, w2vec, n_layers, n_units, dropout):

        n_vocab = len(w2id)

        super(PretrainedModel, self).__init__()
        with self.init_scope():
            #self.embed = L.EmbedID(n_vocab, n_units, initialW=init_W)
            self.embed = Embed(w2id, id2w, w2vec, n_units)
            #self.decoder = L.NStepLSTM(n_layers, n_units, n_units, dropout=dropout)
            self.decoder = Decoder(n_layers, n_units, dropout)
            self.W = L.Linear(n_units, n_vocab)

    def __call__(self, xs):
        batchsize = len(xs)
        concat_dhs, concat_xs_out = self.forward(xs)
        loss = F.sum(F.softmax_cross_entropy(self.W(concat_dhs), concat_xs_out, reduce='no'))/batchsize
        return loss

    def forward(self, xs):
        xs = [self.xp.array(x, dtype=self.xp.int32) for x in xs]

        xs_in = [x[:-1] for x in xs]
        xs_out = [x[1:] for x in xs]
        concat_xs_out = F.concat(xs_out, axis=0)

        exs = self.sequence_embed(self.embed, xs_in)

        _, _, dhs = self.decoder(None, None, exs)

        concat_dhs = F.concat(dhs, axis=0)

        return concat_dhs, concat_xs_out

    def sequence_embed(self, embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, axis=0)
        return exs

    def perplexity(self, xs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x_len = sum([len(x)-1 for x in xs])
            concat_dhs, concat_xs_out = self.forward(xs)
            log_perplexity = F.sum(F.softmax_cross_entropy(self.W(concat_dhs), concat_xs_out, reduce='no'))/x_len
            perplexity = F.exp(log_perplexity)

        return perplexity



def load_data(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d


def save_figs_(save_dir, current_epoch, train_mean_losses, train_mean_perplexitys, \
               dev_mean_losses, dev_mean_perplexitys):
    plt.switch_backend('agg')
    epoch = np.arange(1, current_epoch+1)
    fig = plt.figure(1)
    plt.title('mean losses')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.plot(epoch, train_mean_losses, label='loss(train)')
    plt.plot(epoch, dev_mean_losses, label='loss(dev)')
    plt.legend()
    plt.savefig(save_dir+'mean_losses.{}.png'.format(current_epoch))

    figs = plt.figure(2)
    plt.title('mean perplexitys')
    plt.xlabel('epoch')
    plt.ylabel('mean perplexity')
    plt.plot(epoch, train_mean_perplexitys, label='perplexity(train)')
    plt.plot(epoch, dev_mean_perplexitys, label='perplexity(dev)')
    plt.legend()
    plt.savefig(save_dir+'mean_perplexitys.{}.png'.format(current_epoch))



def _filter(xs, min_len):
    xs = [x for x in xs if len(x) >= min_len]
    return xs

def main(args):
    # w2vec(glove.6B)
    f = open(args.w2vec_path, 'r')
    w2vec = {line.strip().split(' ')[0]: np.array(line.strip().split(' ')[1:], dtype=np.float32) for line in f}

    # Monolingual language model(mlm) data(train, dev)
    if args.mlm_train_data_path and args.mlm_dev_data_path:
        with open(args.mlm_train_data_path, 'rb') as f:
            train_xs = pickle.load(f)
        with open(args.mlm_dev_data_path, 'rb') as f:
            dev_xs = pickle.load(f)

    # stab data(train, dev)
    if args.stab_data_dir:
        contexts = load_data(args.stab_data_dir+'contexts.pickle')    
    train_idxs, _ = get_train_test_idxs(args.idx_path)
    train_contexts = [contexts[idx] for idx in train_idxs[:args.stab_train_size]]
    dev_contexts = [contexts[idx] for idx in train_idxs[args.stab_train_size:]]

    random.seed(12345)

    # train, dev data
    train_xs.extend(train_contexts)
    random.shuffle(train_xs)
    dev_xs.extend(dev_contexts)

    train_xs_flatten = [w for x in train_xs for w in x]
    counter = collections.Counter(train_xs_flatten)
    count_words = counter.most_common()

    # vocab_size most frequent words
    freq_words = [i[0] for i in count_words[:args.vocab_size]]

    # initialize w2id
    w2id = defaultdict(lambda: len(w2id))

    # make dict from frequent words
    get_wid_seq(freq_words, w2id, is_make=True)

    w2id = dict(w2id)
    id2w = {v: k for k, v in w2id.items()}

    # get wid sequence
    train_xs = [get_wid_seq(x, w2id, is_make=False) for x in train_xs]
    dev_xs = [get_wid_seq(x, w2id, is_make=False) for x in dev_xs]

    train_xs = _filter(train_xs, args.min_len)

    # size
    train_size = len(train_xs)
    dev_size = len(dev_xs)

    # define model
    model = PretrainedModel(w2id, id2w, w2vec, 3, 200, args.dropout)

    # Use GPU
    if args.gpu >= 0:
        backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.threshold))
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.rate))

    # initialize reporter
    train_loss_reporter = ScoreReporter(args.mb_size, train_size)
    train_perplexity_reporter = ScoreReporter(args.mb_size, train_size)
    dev_loss_reporter = ScoreReporter(args.mb_size, dev_size)
    dev_perplexity_reporter = ScoreReporter(args.mb_size, dev_size)

    train_mean_losses = []
    train_mean_perplexitys = []
    dev_mean_losses = []
    dev_mean_perplexitys = []
    # train, dev loop
    for epoch in range(args.max_epoch):
        start = time.time()
        print('epoch: {}'.format(epoch+1))
        for mb in range(0, train_size, args.mb_size):
            train_xs_mb = train_xs[mb:mb+args.mb_size]
            model.cleargrads()
            loss = model(train_xs_mb)
            perplexity = model.perplexity(train_xs_mb)
            train_loss_reporter.add(backends.cuda.to_cpu(loss.data))
            train_perplexity_reporter.add(backends.cuda.to_cpu(perplexity.data))
            loss.backward()
            optimizer.update()

        train_mean_loss = train_loss_reporter.mean()
        train_mean_perplexity = train_perplexity_reporter.mean()
        train_mean_losses.append(train_mean_loss)
        train_mean_perplexitys.append(train_mean_perplexity)

        print('train mean loss: {}'.format(train_mean_loss))
        print('train mean perpleixity: {}'.format(train_mean_perplexity))

        for mb in range(0, dev_size, args.mb_size):
            dev_xs_mb = dev_xs[mb:mb+args.mb_size]
            dev_loss = model(dev_xs_mb)
            dev_perplexity = model.perplexity(dev_xs_mb)
            dev_loss_reporter.add(backends.cuda.to_cpu(dev_loss.data))
            dev_perplexity_reporter.add(backends.cuda.to_cpu(dev_perplexity.data))

        dev_mean_loss = dev_loss_reporter.mean()
        dev_mean_perplexity = dev_perplexity_reporter.mean()
        dev_mean_losses.append(dev_mean_loss)
        dev_mean_perplexitys.append(dev_mean_perplexity)
        print('dev mean loss: {}'.format(dev_mean_loss))
        print('dev mean perplexity: {}'.format(dev_mean_perplexity))

        train_loss_reporter = ScoreReporter(args.mb_size, train_size)
        train_perplexity_reporter = ScoreReporter(args.mb_size, train_size)
        dev_loss_reporter = ScoreReporter(args.mb_size, dev_size)
        dev_perplexity_reporter = ScoreReporter(args.mb_size, dev_size)

        if args.save_dir:
            serializers.save_npz(args.save_dir+str(epoch+1)+'decoder.model', model.decoder)
            serializers.save_npz(args.save_dir+str(epoch+1)+'embed.model', model.embed)
            serializers.save_npz(args.save_dir+str(epoch+1)+'pretrained.model', model)
            serializers.save_npz(args.save_dir+str(epoch+1)+'optimizer.model', optimizer)
            save_figs_(args.save_dir, epoch+1, train_mean_losses, train_mean_perplexitys, \
                       dev_mean_losses, dev_mean_perplexitys)
            np.savez(args.save_dir+'mean_perplexitys.npz', x=np.asarray(train_mean_perplexitys), y=np.asarray(dev_mean_perplexitys))
            np.savez(args.save_dir+'mean_losses.npz', x=np.asarray(train_mean_losses), y=np.asarray(dev_mean_losses))


        end = time.time()
        print('elapsed time per epoch: {}'.format(str(end-start)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pretrain decoder')
    parser.add_argument('--w2vec_path')
    parser.add_argument('--mlm_train_data_path')
    parser.add_argument('--mlm_dev_data_path')
    parser.add_argument('--stab_data_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--idx_path')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--mb_size', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--stab_train_size', type=int, default=292)
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=5.0)
    parser.add_argument('--rate', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    main(args)
    
    
    
    
    