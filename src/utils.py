# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import json



def get_wid_seq(x, w2id, is_make=None):
    if is_make:
        w2id['eos']
        w2id['unk']
        wid_seq = np.array([w2id[w] for w in x], dtype=np.int32)
    else:
        wid_seq = np.array([w2id.get(w, w2id['unk']) for w in x], dtype=np.int32)
    return wid_seq



def get_wvec_seq(x, w2vec):
    wvec_seq = np.array([w2vec.get(w, w2vec['unk']) for w in x], dtype=np.float32)
    return wvec_seq



def read_txt(data_dir):
    txt_files = glob.glob(data_dir+'*.txt')
    topics = []
    contexts = []
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            topic, context = f.read().split('\n\n')
            topics.append(nltk.word_tokenize(topic))
            contexts.append(nltk.word_tokenize(context))
    return topics, contexts



def read_sentence(data_dir):
    txt_files = glob.glob(data_dir+'*.txt')
    topics = []
    contexts = []
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            topic, context = f.read().split('\n\n')
            topics.append(nltk.tokenize.sent_tokenize(topic))
            contexts.append(nltk.tokenize.sent_tokenize(context))
    return topics, contexts



def read_ann(data_dir):
    ann_files = glob.glob(data_dir+'*.ann')
    anns = []
    for file in ann_files:
        ann = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                ann.append(nltk.word_tokenize(line))
        anns.append(ann)
    return anns



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



def save_figs(save_dir, current_epoch, train_mean_losses, train_mean_losses_w, train_mean_losses_label, \
              train_bleus, dev_bleus):
    plt.switch_backend('agg')
    epoch = np.arange(1, current_epoch+1)
    fig = plt.figure(1)
    plt.title('mean losses')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.plot(epoch, train_mean_losses, label='loss(total)')
    plt.plot(epoch, train_mean_losses_w, label='loss(word)')
    plt.plot(epoch, train_mean_losses_label, label='loss(labels)')
    plt.legend()
    plt.savefig(save_dir+'mean_losses.{}.png'.format(current_epoch))
    
    figs = plt.figure(2)
    plt.title('bleus')
    plt.xlabel('epoch')
    plt.ylabel('bleu')
    plt.plot(epoch, train_bleus, label='bleu(train)')
    plt.plot(epoch, dev_bleus, label='bleu(dev)')
    plt.legend()
    plt.savefig(save_dir+'bleus.{}.png'.format(current_epoch))



def save_args(save_dir, args):
    args_dict = {k: v for k, v in args._get_kwargs()}
    
    f = open(save_dir+'args.json', 'w')
    json.dump(args_dict, f)
        
    
