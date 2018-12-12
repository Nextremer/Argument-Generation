# -*- coding: utf-8 -*-
# created by Tomohiko Abe
# created at 2018-10-10
import pickle
import argparse
import collections
from collections import defaultdict
import time
import random
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



def main(args):
    # save args
    if args.save_dir:
        save_args(args.save_dir, args)

    # w2vec(glove.6B)
    with open(args.w2vec_path, 'r') as f:
        w2vec = {line.strip().split(' ')[0]: np.array(line.strip().split(' ')[1:], dtype=np.float32) for line in f}

    # load data(tokenized word) 
    topics = load_data(args.data_dir+'topics.pickle')
    contexts = load_data(args.data_dir+'contexts.pickle')
    type_seqs = load_data(args.data_dir+'type_seqs.pickle')
    rel_seqs = load_data(args.data_dir+'rel_seqs.pickle')
    dist_seqs = load_data(args.data_dir+'dist_seqs.pickle')

    # train, test idxs
    train_idxs, test_idxs = get_train_test_idxs(args.idx_path)

    # train, dev idxs
    train_idxs, dev_idxs = train_idxs[:args.stab_train_size], train_idxs[args.stab_train_size:]

    # Monolingual language model(mlm) data(train)
    if args.mlm_train_data_path:
        with open(args.mlm_train_data_path, 'rb') as f:
            train_xs = pickle.load(f)

    train_contexts = [contexts[idx] for idx in train_idxs]
    train_xs.extend(train_contexts)

    train_xs_flatten = [w for x in train_xs for w in x]
    counter = collections.Counter(train_xs_flatten)
    count_words = counter.most_common()

    # vocab_size most frequent words
    freq_words = [i[0] for i in count_words[:args.vocab_size]]

    pretrain_w2id = defaultdict(lambda: len(pretrain_w2id))

    pretrain_w2id['bos']
    pretrain_w2id['eos']
    pretrain_w2id['unk']

    # make dict from frequent words
    get_wid_seq(freq_words, pretrain_w2id, is_make=True)

    pretrain_w2id = dict(pretrain_w2id)
    pretrain_id2w = {v: k for k, v in pretrain_w2id.items()}

    # w2id
    w2id = defaultdict(lambda: len(w2id))

    # shuffle train data
    random.seed(12345)
    random.shuffle(train_idxs)

    # train id sequence
    train_topics = [get_wid_seq(topics[idx], w2id, is_make=True) for idx in train_idxs]
    train_contexts = [get_wid_seq(contexts[idx], w2id, is_make=True) for idx in train_idxs]
    train_type_seqs = [np.array(type_seqs[idx], dtype=np.int32) for idx in train_idxs]
    train_rel_seqs = [np.array(rel_seqs[idx], dtype=np.int32) for idx in train_idxs]
    train_dist_seqs = [np.array(dist_seqs[idx], dtype=np.int32) for idx in train_idxs]

    # w2id, id2w
    w2id = dict(w2id)
    id2w = {v: k for k, v in w2id.items()}

    dev_topics = [topics[idx] for idx in dev_idxs]
    dev_contexts = [contexts[idx] for idx in dev_idxs]
    dev_type_seqs = [np.array(type_seqs[idx], dtype=np.int32) for idx in dev_idxs]
    dev_rel_seqs = [np.array(rel_seqs[idx], dtype=np.int32) for idx in dev_idxs]
    dev_dist_seqs = [np.array(dist_seqs[idx], dtype=np.int32) for idx in dev_idxs]

    test_topics = [topics[idx] for idx in test_idxs]
    test_contexts = [contexts[idx] for idx in test_idxs]
    test_type_seqs = [np.array(type_seqs[idx], dtype=np.int32) for idx in test_idxs]
    test_rel_seqs = [np.array(rel_seqs[idx], dtype=np.int32) for idx in test_idxs]
    test_dist_seqs = [np.array(dist_seqs[idx], dtype=np.int32) for idx in test_idxs]

    # train, dev, test size
    train_size = len(train_topics)
    dev_size = len(dev_topics)
    test_size = len(test_topics)

    assert len(train_topics) == len(train_contexts) == len(train_type_seqs) == len(train_rel_seqs) == len(train_dist_seqs)
    assert len(dev_topics) == len(dev_contexts) == len(dev_type_seqs) == len(dev_rel_seqs) == len(dev_dist_seqs)
    assert len(test_topics) == len(test_contexts) == len(test_type_seqs) == len(test_rel_seqs) == len(test_dist_seqs)

    # define model
    model = Model(args, w2id, id2w, pretrain_w2id, pretrain_id2w, w2vec)

    # Use GPU
    if args.gpu >= 0:
        backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.threshold))
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.rate))
    
    # resume training
    if args.resume:
        serializers.load_npz(args.saved_model_path, model)
        serializers.load_npz(args.saved_opt_path, optimizer)
        mean_losses = np.load(args.save_dir+'mean_losses.npz')
        bleus = np.load(args.save_dir+'bleus.npz')
        train_mean_losses, train_mean_losses_w, train_mean_losses_label = \
        list(mean_losses['x']), list(mean_losses['y']), list(mean_losses['z'])
        train_bleus, dev_bleus, test_bleus = list(bleus['x']), list(bleus['y']), list(bleus['z'])
    else:
        train_mean_losses_w = []
        train_mean_losses_label = []
        train_mean_losses = []
        
        train_bleus = []
        dev_bleus = []
        test_bleus = []

    # initialize reporter
    train_loss_w_reporter = ScoreReporter(args.mb_size, train_size)
    train_loss_label_reporter = ScoreReporter(args.mb_size, train_size)
    train_loss_reporter = ScoreReporter(args.mb_size, train_size)

    # train, dev, test loop
    for epoch in range(args.resume_epoch, args.max_epoch):
        print('epoch: {}'.format(epoch+1))
        start_time = time.time()
        for mb in range(0, train_size, args.mb_size):
            train_topic_mb = train_topics[mb:mb+args.mb_size]
            train_context_mb = train_contexts[mb:mb+args.mb_size]
            train_type_mb = train_type_seqs[mb:mb+args.mb_size]
            train_rel_mb = train_rel_seqs[mb:mb+args.mb_size]
            train_dist_mb = train_dist_seqs[mb:mb+args.mb_size]

            model.cleargrads()
            loss_w, loss_label, loss = model(train_topic_mb, train_context_mb, (train_type_mb, train_rel_mb, train_dist_mb))
            train_loss_w_reporter.add(backends.cuda.to_cpu(loss_w.data))
            train_loss_label_reporter.add(backends.cuda.to_cpu(loss_label.data))
            train_loss_reporter.add(backends.cuda.to_cpu(loss.data))
            loss.backward()
            optimizer.update()

        train_mean_loss_w = train_loss_w_reporter.mean()
        train_mean_loss_label = train_loss_label_reporter.mean()
        train_mean_loss = train_loss_reporter.mean()

        train_mean_losses_w.append(train_mean_loss_w)
        train_mean_losses_label.append(train_mean_loss_label)
        train_mean_losses.append(train_mean_loss)
        print('train mean loss(word): {}'.format(train_mean_loss_w))
        print('train mean loss(label): {}'.format(train_mean_loss_label))
        print('train mean loss(word+eta*label): {}'.format(train_mean_loss))

        # initialize train reporter
        train_loss_w_reporter = ScoreReporter(args.mb_size, train_size)
        train_loss_label_reporter = ScoreReporter(args.mb_size, train_size)
        train_loss_reporter = ScoreReporter(args.mb_size, train_size)

        # generate argument(train)
        hypothesis = []
        references = []
        for mb in range(0, train_size, args.mb_size):
            train_idx_mb = train_idxs[mb:mb+args.mb_size]
            train_topic_mb = [topics[idx] for idx in train_idx_mb]
            train_context_mb = [contexts[idx] for idx in train_idx_mb]

            arguments = model.generate(train_topic_mb, args.max_length)
            hypothesis.extend(arguments)
            references.extend([[ref] for ref in train_context_mb])

            topic = ' '.join(train_topic_mb[0])
            out = ' '.join(arguments[0])

            if args.save_dir:
                with open(args.save_dir+'arguments.txt', 'a') as f:
                    f.write('-'*100+'\n')
                    f.write('train\n')
                    f.write('epoch: '+str(epoch+1)+'\n')
                    f.write('topic:\n')
                    f.write(topic+'\n')
                    f.write('argument:\n')
                    f.write(out+'\n')

            print('-'*100)
            print('train')
            print('topic:')
            print(topic)
            print('generated argument:')
            print(out)

        train_bleu = bleu_score.corpus_bleu(references, hypothesis, \
                                            smoothing_function=bleu_score.SmoothingFunction().method1)

        train_bleus.append(train_bleu)

        # generate argument(dev)
        hypothesis = []
        references = []
        for mb in range(0, dev_size, args.mb_size):
            dev_idx_mb = dev_idxs[mb:mb+args.mb_size]
            dev_topic_mb = [topics[idx] for idx in dev_idx_mb]
            dev_context_mb = [contexts[idx] for idx in dev_idx_mb]

            arguments = model.generate(dev_topic_mb, args.max_length)
            hypothesis.extend(arguments)
            references.extend([[ref] for ref in dev_context_mb])

            for argument, dev_topic in zip(arguments, dev_topic_mb):
                topic = ' '.join(dev_topic)
                out = ' '.join(argument)

                if args.save_dir:
                    with open(args.save_dir+'arguments.txt', 'a') as f:
                        f.write('-'*100+'\n')
                        f.write('dev\n')
                        f.write('epoch: '+str(epoch+1)+'\n')
                        f.write('topic:\n')
                        f.write(topic+'\n')
                        f.write('argument:\n')
                        f.write(out+'\n')

            print('-'*100)
            print('dev')
            print('topic:')
            print(topic)
            print('generated argument:')
            print(out)

        dev_bleu = bleu_score.corpus_bleu(references, hypothesis, \
                                          smoothing_function=bleu_score.SmoothingFunction().method1)

        dev_bleus.append(dev_bleu)

        # generate argument(test)
        hypothesis = []
        references = []
        for mb in range(0, test_size, args.mb_size):
            test_idx_mb = test_idxs[mb:mb+args.mb_size]
            test_topic_mb = [topics[idx] for idx in test_idx_mb]
            test_context_mb = [contexts[idx] for idx in test_idx_mb]

            arguments = model.generate(test_topic_mb, args.max_length)
            hypothesis.extend(arguments)
            references.extend([[ref] for ref in test_context_mb])

            for argument, test_topic in zip(arguments, test_topic_mb):
                topic = ' '.join(test_topic)
                out = ' '.join(argument)

                if args.save_dir:
                    with open(args.save_dir+'arguments.txt', 'a') as f:
                        f.write('-'*100+'\n')
                        f.write('test\n')
                        f.write('epoch: '+str(epoch+1)+'\n')
                        f.write('topic:\n')
                        f.write(topic+'\n')
                        f.write('argument:\n')
                        f.write(out+'\n')

            print('-'*100)
            print('test')
            print('topic:')
            print(topic)
            print('generated argument:')
            print(out)

        test_bleu = bleu_score.corpus_bleu(references, hypothesis, \
                                           smoothing_function=bleu_score.SmoothingFunction().method1)
        print('train bleu: '+str(train_bleu))
        print('dev bleu: '+str(dev_bleu))
        print('test bleu: '+str(test_bleu))
        test_bleus.append(test_bleu)

        if args.save_dir:
            # save figs
            save_figs(\
                args.save_dir, epoch+1, train_mean_losses, train_mean_losses_w, train_mean_losses_label, \
                train_bleus, dev_bleus)
            serializers.save_npz(args.save_dir+'model.'+str(epoch+1)+'.model', model)
            serializers.save_npz(args.save_dir+'optimizer.'+str(epoch+1)+'.model', optimizer)
            np.savez(args.save_dir+'bleus.npz', x=np.asarray(train_bleus), y=np.asarray(dev_bleus), \
                     z=np.asarray(test_bleus))
            np.savez(args.save_dir+'mean_losses.npz', x=np.asarray(train_mean_losses), y=np.asarray(train_mean_losses_w), \
                     z=np.asarray(train_mean_losses_label))

        end_time = time.time()
        print('elapsed time per one iter:{}'.format(end_time-start_time))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train argument generator')
    parser.add_argument('--data_dir', help='data directory')
    parser.add_argument('--idx_path', help='train test idx file')
    parser.add_argument('--save_dir', help='save figures directory')
    parser.add_argument('--w2vec_path', help='w2vec path')
    parser.add_argument('--saved_model_path')
    parser.add_argument('--saved_opt_path')
    parser.add_argument('--mlm_train_data_path', help='monolingual language model train data path')
    parser.add_argument('--pretrained_decoder_path', help='pretrained decoder path')
    parser.add_argument('--pretrained_embed_path', help='pretrained embed path')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_layers2', type=int, default=3)
    parser.add_argument('--n_units', type=int, default=200)
    parser.add_argument('--attn_n_units', type=int, default=150)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--mb_size', type=int, default=16)
    parser.add_argument('--stab_train_size', type=int, default=292)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_length', type=int, default=120)
    parser.add_argument('--threshold', type=float, default=5.0)
    parser.add_argument('--rate', type=float, default=5e-4)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_label_in', action='store_true')
    parser.add_argument('--use_rnn3', action='store_true')
    args = parser.parse_args()

    main(args)
