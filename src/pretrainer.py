# -*- coding: utf-8 -*-
# created by Tomohiko Abe
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, optimizers, backends, serializers

import numpy as np
import nltk
from collections import defaultdict
import argparse
import pickle
from operator import add
from functools import reduce

from utils import *


EOS = 0


class Pretrainer(chainer.Chain):
    
    def __init__(self, w2id, id2w, w2vec, n_layers, n_units, dropout):
        
        n_vocab = len(w2id)
        
        init_W = [w2vec[id2w[i]] if id2w[i] in w2vec.keys() else np.random.normal(scale=np.sqrt(2./n_units), size=(n_units, )) \
                  for i, w in id2w.items()]
        init_W = np.asarray(init_W, dtype=np.float32)
        
        super(Pretrainer, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=init_W)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, dropout=dropout)
            self.W = L.Linear(n_units, n_vocab)
            
    def __call__(self, xs):
        batchsize = len(xs)
        concat_dhs, concat_xs_out = self.forward(xs)
        loss = F.sum(F.softmax_cross_entropy(self.W(concat_dhs), concat_xs_out, reduce='no'))/batchsize
        
        return loss
            
    def forward(self, xs):
        xs = [self.xp.array(x, dtype=self.xp.int32) for x in xs]
        eos = self.xp.array([EOS], dtype=self.xp.int32)
        
        xs_in = [F.concat([eos, x], axis=0) for x in xs]
        xs_out = [F.concat([x, eos], axis=0) for x in xs]
        assert len(xs_in) == len(xs_out)
        
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

def load_data(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d

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

        
def save_figs_(save_dir, current_epoch, train_mean_losses):
    plt.switch_backend('agg')
    epoch = np.arange(1, current_epoch+1)
    fig = plt.figure(1)
    plt.title('mean losses')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.plot(epoch, train_mean_losses, label='loss(train)')
    plt.legend()
    plt.savefig(save_dir+'mean_losses.{}.png'.format(current_epoch))
    
    

def main(args):
    # w2vec
    f = open(args.w2vec_path, 'r')
    w2vec = {line.strip().split(' ')[0]: np.array(line.strip().split(' ')[1:], dtype=np.float32) for line in f}
    
    # load pretrain data
    #f = open(args.data_path, 'rb')
    #xs = pickle.load(f)
    
    # load data
    topics = load_data(args.data_dir+'topics.pickle')
    contexts = load_data(args.data_dir+'contexts.pickle')
    
    # initialize w2id
    w2id = defaultdict(lambda: len(w2id))
    
    # get wid sequence
    #xs = [get_wid_seq(x, w2id, is_make=True) for x in xs]
    
    # train, test idxs
    train_idxs, _ = get_train_test_idxs(args.idx_path)
    train_topics = [get_wid_seq(topics[idx], w2id, is_make=True) for idx in train_idxs]
    train_contexts = [get_wid_seq(contexts[idx], w2id, is_make=True) for idx in train_idxs]
        
    id2w = {v: k for k, v in w2id.items()}
    
    train_size = len(train_contexts)
    
    # define model
    model = Pretrainer(w2id, id2w, w2vec, 3, 200, 0.8)
    
    # Use GPU
    if args.gpu >= 0:
        backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
        
    # optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    
    # initialize reporter
    train_loss_reporter = ScoreReporter(args.mb_size, train_size)
    
    train_mean_losses = []
    
    # train test loop
    for epoch in range(args.max_epoch):
        print('epoch: {}'.format(epoch+1))
        for mb in range(0, train_size, args.mb_size):
            train_xs = train_contexts[mb:mb+args.mb_size]
            
            model.cleargrads()
            loss = model(train_xs)
            train_loss_reporter.add(backends.cuda.to_cpu(loss.data))
            loss.backward()
            optimizer.update()
            
        train_mean_losses.append(train_loss_reporter.mean())
        print('train mean loss: {}'.format(train_loss_reporter.mean()))
            
        train_loss_reporter = ScoreReporter(args.mb_size, train_size)
    
    if args.save_dir:
        serializers.save_npz(args.save_dir+'pretrained.model', model)
        save_figs_(args.save_dir, args.max_epoch, train_mean_losses)
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pretrain decoder')
    parser.add_argument('--w2vec_path')
    parser.add_argument('--data_path')
    parser.add_argument('--data_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--idx_path')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--mb_size', type=int, default=16)
    args = parser.parse_args()
    
    main(args)
    
    
    
    
    