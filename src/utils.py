# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import json

def get_wid_seq(x, w2id, is_make=None):
    if is_make:
        w2id['<eos>']
        w2id['<unk>']
        wid_seq = np.array([w2id[w] for w in x], dtype=np.int32)
    else:
        wid_seq = np.array([w2id.get(w, w2id['<unk>']) for w in x], dtype=np.int32)
    return wid_seq


class ScoreReporter(object):
    def __init__(self, mb_size, data_size):
        self.score_sum = 0
        self.mb_size = mb_size
        self.data_size = data_size
        
    def add(self, score):
        self.score_sum += score
    
    def mean(self):
        n = len(list(range(0, self.data_size, self.mb_size)))
        return self.score_sum / float(n)

    
def save_figs(save_dir, current_epoch, train_mean_losses, train_mean_losses1, train_mean_losses23):
    plt.switch_backend('agg')
    epoch = np.arange(1, current_epoch+1)
    fig = plt.figure(1)
    plt.title('mean losses')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.plot(epoch, train_mean_losses, label='loss(total)')
    plt.plot(epoch, train_mean_losses1, label='loss(word)')
    plt.plot(epoch, train_mean_losses23, label='loss(labels)')
    plt.legend()
    plt.savefig(save_dir+'mean_losses.{}.png'.format(current_epoch))
    
    

def save_args(save_dir, *args):
    keys = ["n_layers", "n_units", "attn_n_units", "eta", "max_epoch", "mb_size", "dropout"]
    assert len(args) == len(keys)
    args_dict = {k: a for k, a in zip(keys, args)}
    
    f = open(save_dir+'args.json', 'w')
    json.dump(args_dict, f)
        
    
