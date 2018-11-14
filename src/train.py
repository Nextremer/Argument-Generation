# -*- coding: utf-8 -*-
# created by Tomohiko Abe
# created at 2018-10-10
import pickle
import argparse
from collections import defaultdict
import time
from nltk.translate import bleu_score

from utils import *
from model import *


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


def calculateBleu(hypothesis, reference):
    references = [reference]
    list_of_references = [references]
    list_of_hypotheses = [hypothesis]
    bleu = bleu_score.corpus_bleu(list_of_references, list_of_hypotheses, \
                                  smoothing_function=bleu_score.SmoothingFunction().method1)

    return bleu
            

def main(args):
    # save args
    if args.save_dir:
        save_args(
            args.save_dir,
            args.n_layers, 
            args.n_units, 
            args.attn_n_units,
            args.eta,
            args.max_epoch,
            args.mb_size,
            args.dropout)
            
    # load data(tokenized word) 
    topics = load_data(args.data_dir+'topics.pickle')
    contexts = load_data(args.data_dir+'contexts.pickle')
    type_seqs = load_data(args.data_dir+'type_seqs.pickle')
    rel_seqs = load_data(args.data_dir+'rel_seqs.pickle')
    dist_seqs = load_data(args.data_dir+'dist_seqs.pickle')
    
    # train, test idxs
    train_idxs, test_idxs = get_train_test_idxs(args.idx_path)
    
    # train, test size
    train_size = len(train_idxs)
    test_size = len(test_idxs)

    # word2vec
    f = open(args.w2vec_path, 'r')
    w2vec = {line.strip().split(' ')[0]: np.array(line.strip().split(' ')[1:], dtype=np.float32) for line in f}
    
    # w2id
    w2id = defaultdict(lambda: len(w2id))

    # load pretrain data
    #f = open(args.data_path, 'rb')
    #xs = pickle.load(f)

    # get wid sequence of pretrain dataset
    #xs = [get_wid_seq(x, w2id, is_make=True) for x in xs]

    # train, test split
    train_topics = [get_wid_seq(topics[idx], w2id, is_make=True) for idx in train_idxs]
    train_contexts = [get_wid_seq(contexts[idx], w2id, is_make=True) for idx in train_idxs]
    train_type_seqs = [np.array(type_seqs[idx], dtype=np.int32) for idx in train_idxs]
    train_rel_seqs = [np.array(rel_seqs[idx], dtype=np.int32) for idx in train_idxs]
    train_dist_seqs = [np.array(dist_seqs[idx], dtype=np.int32) for idx in train_idxs]

    idxs = sorted(np.arange(0, train_size), key=lambda x: len(train_contexts[x]), reverse=True)
    train_topics = [train_topics[idx] for idx in idxs]
    train_contexts = [train_contexts[idx] for idx in idxs]
    train_type_seqs = [train_type_seqs[idx] for idx in idxs]
    train_rel_seqs = [train_rel_seqs[idx] for idx in idxs]
    train_dist_seqs = [train_dist_seqs[idx] for idx in idxs]

    test_topics = [topics[idx] for idx in test_idxs]
    test_contexts = [contexts[idx] for idx in test_idxs]
    test_type_seqs = [np.array(type_seqs[idx], dtype=np.int32) for idx in test_idxs]
    test_rel_seqs = [np.array(rel_seqs[idx], dtype=np.int32) for idx in test_idxs]
    test_dist_seqs = [np.array(dist_seqs[idx], dtype=np.int32) for idx in test_idxs]
    
    idxs = sorted(np.arange(0, test_size), key=lambda x: len(test_contexts[x]), reverse=True)
    test_topics = [test_topics[idx] for idx in idxs]
    test_contexts = [test_contexts[idx] for idx in idxs]
    test_type_seqs = [test_type_seqs[idx] for idx in idxs]
    test_rel_seqs = [test_rel_seqs[idx] for idx in idxs]
    test_dist_seqs = [test_dist_seqs[idx] for idx in idxs]
    
    # id2w
    id2w = {v: k for k, v in w2id.items()}
    
    # define model
    model = Model(
                w2id,
                id2w,
                w2vec,
                args.n_layers, 
                args.n_units, 
                args.attn_n_units, 
                args.eta,
                args.dropout)
    
    # Use GPU
    if args.gpu >= 0:
        backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    
    # initialize reporter
    train_loss_w_reporter = ScoreReporter(args.mb_size, train_size)
    train_loss_label_reporter = ScoreReporter(args.mb_size, train_size)
    train_loss_reporter = ScoreReporter(args.mb_size, train_size)
    train_bleu_reporter = ScoreReporter(args.mb_size, train_size)
    
    train_mean_losses_w = []
    train_mean_losses_label = []
    train_mean_losses = []
    train_mean_bleus = []
    test_mean_bleus = []
    
    # train test loop
    for epoch in range(args.max_epoch):
        print('epoch: {}'.format(epoch+1))
        start_time = time.time()
        for mb in range(0, train_size, args.mb_size):
            train_topic_mb = train_topics[mb:mb+args.mb_size]
            train_context_mb = train_contexts[mb:mb+args.mb_size]
            train_type_mb = train_type_seqs[mb:mb+args.mb_size]
            train_rel_mb = train_rel_seqs[mb:mb+args.mb_size]
            train_dist_mb = train_dist_seqs[mb:mb+args.mb_size]
            
            model.cleargrads()
            loss1, loss2, loss = model(train_topic_mb, train_context_mb, (train_type_mb, train_rel_mb, train_dist_mb))
            train_loss_w_reporter.add(backends.cuda.to_cpu(loss1.data))
            train_loss_label_reporter.add(backends.cuda.to_cpu(args.eta*loss2.data))
            train_loss_reporter.add(backends.cuda.to_cpu(loss.data))
            loss.backward()
            optimizer.update()
        
        train_mean_losses_w.append(train_loss_w_reporter.mean())
        train_mean_losses_label.append(train_loss_label_reporter.mean())
        train_mean_losses.append(train_loss_reporter.mean())
        print('train mean loss(word): {}'.format(train_loss_w_reporter.mean()))
        print('train mean loss(eta*loss2): {}'.format(train_loss_label_reporter.mean()))
        print('train mean loss: {}'.format(train_loss_reporter.mean()))
        
        train_mean_bleu = 0
        # generate argument(train)
        for mb in range(0, train_size, args.mb_size):
            topic = ''
            for w in topics[train_idxs[mb]]:
                topic += w
                topic += ' '
            train_idx_mb = train_idxs[mb:mb+args.mb_size]
            train_topics_mb = [topics[idx] for idx in train_idx_mb]
            arguments = model.generate(train_topics_mb, args.max_length)
            
            outs = []
            for i, argument in enumerate(arguments):
                bleu = calculateBleu(argument, contexts[train_idxs[mb+i]])
                train_mean_bleu += bleu
                out = ''
                for w in argument:
                    out += w
                    out += ' '  
                outs.append(out)
                
            print('-'*100)
            print('train')
            print('topic:')
            print(topic)
            print('generated argument:')
            print(outs[0])
            
            if args.save_dir:
                with open(args.save_dir+'arguments.txt', 'a') as f:
                    f.write('-'*50+'\n')
                    f.write('train\n')
                    f.write('epoch: '+str(epoch+1)+'\n')
                    f.write('topic: '+topic+'\n')
                    f.write('argument: '+outs[0]+'\n')
            
        train_mean_bleu /= float(train_size)
        print('train mean bleu: '+str(train_mean_bleu))
        train_mean_bleus.append(train_mean_bleu)
                    
        # initialize train reporter
        train_loss_w_reporter = ScoreReporter(args.mb_size, train_size)
        train_loss_label_reporter = ScoreReporter(args.mb_size, train_size)
        train_loss_reporter = ScoreReporter(args.mb_size, train_size)

        test_mean_bleu = 0
        # generate argument(test)
        for mb in range(0, test_size, args.mb_size):
            test_idx_mb = test_idxs[mb:mb+args.mb_size]
            test_topics_mb = [topics[idx] for idx in test_idx_mb]
            arguments = model.generate(test_topics_mb, args.max_length)

            outs = []
            for i, argument in enumerate(arguments):
                topic = ''
                for w in topics[test_idxs[mb+i]]:
                    topic += w
                    topic += ' '
                bleu = calculateBleu(argument, contexts[test_idxs[mb+i]])
                test_mean_bleu += bleu
                out = ''
                for w in argument:
                    out += w
                    out += ' '
                if args.save_dir:
                    with open(args.save_dir+'arguments.txt', 'a') as f:
                        f.write('-'*50+'\n')
                        f.write('test\n')
                        f.write('epoch: '+str(epoch+1)+'\n')
                        f.write('topic: '+topic+'\n')
                        f.write('argument: '+out+'\n')
                outs.append(out)

            print('-'*100)
            print('test')
            print('topic:')
            print(topic)
            print('generated argument:')
            print(outs[0])

        test_mean_bleu /= float(test_size)
        test_mean_bleus.append(test_mean_bleu)
        print('test mean bleu: '+str(test_mean_bleu))
                    
        if args.save_dir:
            # save figs
            save_figs(\
                args.save_dir, epoch+1, train_mean_losses, train_mean_losses_w, train_mean_losses_label, \
                train_mean_bleus, test_mean_bleus)
            
            np.savez(args.save_dir+'mean_bleus.npz', x = np.asarray(train_mean_bleus), y = np.asarray(test_mean_bleus))

        end_time = time.time()
        print('elapsed time:{}'.format(end_time-start_time))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train argument generator')
    parser.add_argument('--data_dir', help='data directory')
    parser.add_argument('--idx_path', help='train test idx file')
    parser.add_argument('--save_dir', help='save figures directory')
    parser.add_argument('--w2vec_path', help='w2vec path')
    parser.add_argument('--data_path', help='pretrain dataset')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_units', type=int, default=200)
    parser.add_argument('--attn_n_units', type=int, default=150)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--mb_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_length', type=int, default=100)
    args = parser.parse_args()
    
    main(args)
