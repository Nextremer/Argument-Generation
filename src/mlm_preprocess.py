# coding: utf-8
import argparse
import nltk
from bs4 import BeautifulSoup
import pickle

def main(args):
    
    train_data = []
    dev_data = []
    
    if args.mlm_train_data_path:
        with open(args.mlm_train_data_path, 'r') as f:
            for line in f:
                train_data.append(nltk.word_tokenize(line.strip()))
    
    if args.mlm_dev_data_path:
        with open(args.mlm_dev_data_path, 'r') as f:
            for line in f:
                dev_data.append(nltk.word_tokenize(line.strip()))
            
    if args.mlm_test_data_path:
        with open(args.mlm_test_data_path, 'r') as f:
            d = f.read()
    
    # 文長がmin_len未満の文は削除
    train_data = [s for s in train_data if len(s) >= args.min_len]
            
    soup = BeautifulSoup(d)
    test_data = soup.find_all('seg')
    test_data = [nltk.word_tokenize(s.string.strip()) for s in test_data]
    
    if args.use_lower:
        train_data = [[w.lower() for w in s] for s in train_data]
        dev_data = [[w.lower() for w in s] for s in dev_data]
        test_data = [[w.lower() for w in s] for s in test_data]
        
    if args.save_dir:
        with open(args.save_dir+'mlm_train.pickle', 'wb') as f:
            pickle.dump(train_data, f)
        with open(args.save_dir+'mlm_dev.pickle', 'wb') as f:
            pickle.dump(dev_data, f)
        with open(args.save_dir+'mlm_test.pickle', 'wb') as f:
            pickle.dump(test_data, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess monolingual language model \
    corpus(http://www.statmt.org/wmt11/translation-task.html)')
    parser.add_argument('--mlm_train_data_path', help='monolingual language model train data path')
    parser.add_argument('--mlm_dev_data_path', help='monolingual language model dev data path')
    parser.add_argument('--mlm_test_data_path', help='monolingual language model test data path')
    parser.add_argument('--save_dir')
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--use_lower', action='store_true')
    args = parser.parse_args()

    main(args)