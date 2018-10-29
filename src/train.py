# -*- coding: utf-8 -*-
# created by Tomohiko Abe
# created at 2018-10-10
import pickle
import argparse
from collections import defaultdict
from utils import *
from model import *
import time


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
    bio_seqs = load_data(args.data_dir+'bio_seqs.pickle')
    type_seqs = load_data(args.data_dir+'type_seqs.pickle')
    
    # train, test idxs
    train_idxs, test_idxs = get_train_test_idxs(args.idx_path)
    
    # train, test size
    train_size = len(train_idxs)
    test_size = len(test_idxs)
    
    # initialize dictionary
    source_w2id = defaultdict(lambda: len(source_w2id))
    target_w2id = defaultdict(lambda: len(target_w2id))
    
    # train, test split
    train_topics = [get_wid_seq(topics[idx], source_w2id, is_make=True) for idx in train_idxs]
    train_contexts = [get_wid_seq(contexts[idx], target_w2id, is_make=True) for idx in train_idxs]
    train_bio_seqs = [np.array(bio_seqs[idx], dtype=np.int32) for idx in train_idxs]
    train_type_seqs = [np.array(type_seqs[idx], dtype=np.int32) for idx in train_idxs]
    
    idxs = sorted(np.arange(0, train_size), key=lambda x: len(train_contexts[x]), reverse=True)
    train_topics = [train_topics[idx] for idx in idxs]
    train_contexts = [train_contexts[idx] for idx in idxs]
    train_bio_seqs = [train_bio_seqs[idx] for idx in idxs]
    train_type_seqs = [train_type_seqs[idx] for idx in idxs]
    
    test_topics = [get_wid_seq(topics[idx], source_w2id, is_make=False) for idx in test_idxs]
    test_contexts = [get_wid_seq(contexts[idx], target_w2id, is_make=False) for idx in test_idxs]
    test_bio_seqs = [np.array(bio_seqs[idx], dtype=np.int32) for idx in test_idxs]
    test_type_seqs = [np.array(type_seqs[idx], dtype=np.int32) for idx in test_idxs]
    
    idxs = sorted(np.arange(0, test_size), key=lambda x: len(test_contexts[x]), reverse=True)
    test_topics = [test_topics[idx] for idx in idxs]
    test_contexts = [test_contexts[idx] for idx in idxs]
    test_bio_seqs = [test_bio_seqs[idx] for idx in idxs]
    test_type_seqs = [test_type_seqs[idx] for idx in idxs]
    
    # id2w
    source_id2w = {v: k for k, v in source_w2id.items()}
    target_id2w = {v: k for k, v in target_w2id.items()}
    
    # define model
    model = Model(
                source_w2id, 
                target_w2id, 
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
    
    train_mean_losses_w = []
    train_mean_losses_label = []
    train_mean_losses = []
    
    # train test loop
    for epoch in range(args.max_epoch):
        print('epoch: {}'.format(epoch+1))
        start_time = time.time()
        for mb in range(0, train_size, args.mb_size):
            train_topic_mb = train_topics[mb:mb+args.mb_size]
            train_context_mb = train_contexts[mb:mb+args.mb_size]
            train_bio_mb = train_bio_seqs[mb:mb+args.mb_size]
            train_type_mb = train_type_seqs[mb:mb+args.mb_size]
            
            model.cleargrads()
            loss1, loss2, loss3, loss = model(train_topic_mb, train_context_mb, (train_bio_mb, train_type_mb))
            train_loss_w_reporter.add(backends.cuda.to_cpu(loss1.data))
            train_loss_label_reporter.add(backends.cuda.to_cpu(args.eta*(loss2.data+loss3.data)))
            train_loss_reporter.add(backends.cuda.to_cpu(loss.data))
            loss.backward()
            optimizer.update()
        
        train_mean_losses_w.append(train_loss_w_reporter.mean())
        train_mean_losses_label.append(train_loss_label_reporter.mean())
        train_mean_losses.append(train_loss_reporter.mean())
        print('train mean loss word: {}'.format(train_loss_w_reporter.mean()))
        print('train mean loss eta*(loss2 + loss3): {}'.format(train_loss_label_reporter.mean()))
        print('train mean loss: {}'.format(train_loss_reporter.mean()))
        
        # generate argument(train)
        idxs = [0, 50, 100, 150, 200, 250, 300]
        arguments = []
        for idx in idxs:
            print('-'*50)
            print('train')
            print('topic:')
            topic = ''
            for i in train_topics[idx]:
                topic += source_id2w[i]
                topic += ' '
            print(topic)
            print('generated argument:')
            argument = model.generate([train_topics[idx]], args.max_length)
            print(argument)
            arguments.append((topic, argument[0]))
            if args.save_dir:
                with open(args.save_dir+'arguments.txt', 'a') as f:
                    f.write('-'*50+'\n')
                    f.write('train\n')
                    f.write('epoch: '+str(epoch+1)+'\n')
                    f.write('topic: '+topic+'\n')
                    f.write('argument: '+argument[0]+'\n')
                    
        print('-'*50)
        
        # initialize train reporter
        train_loss_w_reporter = ScoreReporter(args.mb_size, train_size)
        train_loss_label_reporter = ScoreReporter(args.mb_size, train_size)
        train_loss_reporter = ScoreReporter(args.mb_size, train_size)
                
        # generate argument(test)
        idxs = [0, 5, 10, 15, 20, 25, 30]
        for idx in idxs:
            print('-'*50)
            print('test')
            print('topic:')
            topic = ''
            for i in test_topics[idx]:
                topic += source_id2w[i]
                topic += ' '
            print(topic)
            print('generated argument:')
            argument = model.generate([test_topics[idx]], args.max_length)
            print(argument)
            if args.save_dir:
                with open(args.save_dir+'arguments.txt', 'a') as f:
                    f.write('-'*50+'\n')
                    f.write('test\n')
                    f.write('epoch: '+str(epoch+1)+'\n')
                    f.write('topic: '+topic+'\n')
                    f.write('argument: '+argument[0]+'\n')
                    
        if args.save_dir:
            # save figs
            save_figs(
                args.save_dir, epoch+1, train_mean_losses, train_mean_losses_w, train_mean_losses_label)
            
        end_time = time.time()
        print('elapsed time:{}'.format(end_time-start_time))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train argument generator')
    parser.add_argument('--data_dir', help='data directory')
    parser.add_argument('--idx_path', help='train test idx file')
    parser.add_argument('--save_dir', help='save figures directory')
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
