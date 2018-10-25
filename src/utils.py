# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

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

    
def save_figs(save_dir, current_epoch, train_mean_losses, test_mean_losses, \
              train_token_level_accs, test_token_level_accs):
    plt.switch_backend('agg')
    epoch = np.arange(1, current_epoch+1)
    fig = plt.figure(1)
    plt.title('mean losses')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.plot(epoch, train_mean_losses, label='train')
    plt.plot(epoch, test_mean_losses, label='test')
    plt.legend()
    plt.savefig(save_dir+'mean_losses.{}.png'.format(current_epoch))
    
    fig = plt.figure(2)
    plt.title('token level accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(epoch, train_token_level_accs, label='train')
    plt.plot(epoch, test_token_level_accs, label='test')
    plt.legend()
    plt.savefig(save_dir+'mean_accuracies:{}.png'.format(current_epoch))



