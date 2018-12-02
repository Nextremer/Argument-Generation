# coding: utf-8
# creted by Tomohiko Abe
# created at 2018-12-01

from pretrain import *


def main(args):
    # w2vec(glove.6B)
    f = open(args.w2vec_path, 'r')
    w2vec = {line.strip().split(' ')[0]: np.array(line.strip().split(' ')[1:], dtype=np.float32) for line in f}
    
    if args.mlm_train_data_path and args.mlm_dev_data_path and args.mlm_test_data_path:
        with open(args.mlm_train_data_path, 'rb') as f:
            train_xs = pickle.load(f)
        with open(args.mlm_test_data_path, 'rb') as f:
            test_xs = pickle.load(f)
    
    # stab data(train, dev)
    if args.stab_data_dir:
        contexts = load_data(args.stab_data_dir+'contexts.pickle')    
    train_idxs, _ = get_train_test_idxs(args.idx_path)
    train_contexts = [contexts[idx] for idx in train_idxs[:args.stab_train_size]]
    
    # train, dev data
    train_xs.extend(train_contexts)
    
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
    
    test_size = len(test_xs)
    
    model = PretrainedModel(w2id, id2w, w2vec, 3, 200, args.dropout)
    serializers.load_npz(args.model_path, model)
    
    # Use GPU
    if args.gpu >= 0:
        backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    
    # test
    test_perplexity_reporter = ScoreReporter(args.mb_size, test_size)
    for mb in range(0, test_size, args.mb_size):
        test_xs_mb = test_xs[mb:mb+args.mb_size]
        test_perplexity = model.perplexity(test_xs_mb)
        test_perplexity_reporter.add(backends.cuda.to_cpu(test_perplexity.data))
    
    test_mean_perplexity = test_perplexity_reporter.mean()
    print('test mean perplexity: {}'.format(test_mean_perplexity))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pretrain decoder')
    parser.add_argument('--w2vec_path')
    parser.add_argument('--mlm_train_data_path')
    parser.add_argument('--mlm_test_data_path')
    parser.add_argument('--stab_data_dir')
    parser.add_argument('--idx_path')
    parser.add_argument('--model_path')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--mb_size', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--stab_train_size', type=int, default=292)
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()
    
    main(args)