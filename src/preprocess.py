# -*- coding:utf-8 -*-
import numpy as np
import json
import pickle
from collections import defaultdict
from utils import *
import argparse



def type_tagging(seq, match_idx, label, length):
    """
    'None': 0
    'B-Premise': 1
    'I-Premise':2
    'B-Claim':3
    'I-Claim':4
    'B-MajorClaim':5
    'I-MajorClaim':6
    """
    if label == 'MajorClaim':
        seq[match_idx] = 5
        seq[(match_idx+1):(match_idx+length)] = [6]*(length-1)
    elif label == 'Claim':
        seq[match_idx] = 3
        seq[(match_idx+1):(match_idx+length)] = [4]*(length-1)
    elif label == 'Premise':
        seq[match_idx] = 1
        seq[(match_idx+1):(match_idx+length)] = [2]*(length-1)
    return seq



def get_match_idx1(match_idxs, essay_num, ann):
    if len(match_idxs) == 1:
        match_idx = match_idxs[0]

    elif len(match_idxs) == 2:
        if essay_num == 153:
            match_idx = match_idxs[0]
        elif essay_num == 163 or essay_num == 166 or essay_num == 181:
            match_idx = match_idxs[1]
        elif essay_num == 263:
            if ann[0] == 'T2':
                match_idx = match_idxs[0]
            elif ann[0] == 'T14':
                match_idx = match_idxs[1]
        elif essay_num == 266:
            if ann[0] == 'T1':
                match_idx = match_idxs[0]
            elif ann[0] == 'T10':
                match_idx = match_idxs[1]
        elif essay_num == 285:
            if ann[0] == 'T4':
                match_idx = match_idxs[1]
            elif ann[0] == 'T14':
                match_idx = match_idxs[0]
                        
    elif len(match_idxs) == 3:
        if ann[0] == 'T1':
            match_idx = match_idxs[1]
        elif ann[0] == 'T3':
            match_idx = match_idxs[2]

    return match_idx



def get_match_idx2(match_idxs, essay_num, ann):
    if len(match_idxs) == 1:
        match_idx = match_idxs[0]

    elif len(match_idxs) == 2:
        if essay_num == 153:
            match_idx = match_idxs[0]
        elif essay_num == 163 or essay_num == 166 or essay_num == 181:
            match_idx = match_idxs[1]
        elif essay_num == 263:
            if ann[0] == 'T2':
                match_idx = match_idxs[0]
            elif ann[0] == 'T14':
                match_idx = match_idxs[1]
        elif essay_num == 285:
            if ann[0] == 'T4':
                match_idx = match_idxs[1]
            elif ann[0] == 'T14':
                match_idx = match_idxs[0]
                        
    elif len(match_idxs) == 3:
        if ann[0] == 'T1':
            match_idx = match_idxs[1]
        elif ann[0] == 'T3':
            match_idx = match_idxs[2]

    return match_idx



def get_labels(context, anns, essay_num, use_lower):
    """1つのcontextに対して、component typeのラベル系列を得る"""
    type_seq = [0]*len(context)
    for ann in anns:
        if ann[0].startswith('T'):
            pattern = ann[4:]
            if use_lower:
                pattern = [w.lower() for w in pattern]
            pattern_length = len(pattern)
            idxs = [idx for idx, w in enumerate(context) if w == pattern[0]]
            match_idxs = list(filter(lambda x: context[x:x+pattern_length] == pattern, idxs))
            assert len(match_idxs) > 0

            if use_lower:
                match_idx = get_match_idx1(match_idxs, essay_num, ann)
            else:
                match_idx = get_match_idx2(match_idxs, essay_num, ann)

        type_seq = type_tagging(type_seq, match_idx, ann[1], pattern_length)

    return type_seq


def main(args):
    # tokenized topics, tokenized contexts, annotations
    topics, contexts = read_txt(args.data_dir)
    anns = read_ann(args.data_dir)
    assert len(topics) == len(contexts) == len(anns)

    if args.use_lower:
        topics = [[w.lower() for w in topic] for topic in topics]
        contexts = [[w.lower() for w in context] for context in contexts]

    type_seqs = []
    essay_num = 0
    for context, ann in zip(contexts, anns):
        type_seq = get_labels(context, ann, essay_num, args.use_lower)
        type_seqs.append(type_seq)
        essay_num += 1

    # save tokenized topics, tokenized contexts, type sequences
    if args.save_dir:
        with open(args.save_dir+'topics.pickle', 'wb') as f:
            pickle.dump(topics, f)
        with open(args.save_dir+'contexts.pickle', 'wb') as f:
            pickle.dump(contexts, f)
        with open(args.save_dir+'type_seqs.pickle', 'wb') as f:
            pickle.dump(type_seqs, f)

    """
    d = []
    f = open(args.pretrain_data_path, 'r')
    for line in f:
        d.append(nltk.word_tokenize(line.strip()))

    if args.use_lower:
        d = [[w.lower() for w in s] for s in d]

    if args.save_dir:
        with open(args.save_dir+'news.2011.en.pickle', 'wb') as f:
            pickle.dump(d, f)
    """

    topics_sent, contexts_sent = read_sentence(args.data_dir)
    if args.use_lower:
        topics_sent = [[sent.lower() for sent in topic_sent] for topic_sent in topics_sent]
        contexts_sent = [[sent.lower() for sent in context_sent] for context_sent in contexts_sent]
        print(contexts_sent[0])

    if args.save_dir:
        with open(args.save_dir+'topics_sent.pickle', 'wb') as f:
            pickle.dump(topics_sent, f)
        with open(args.save_dir+'contexts_sent.pickle', 'wb') as f:
            pickle.dump(contexts_sent, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess corpus(Stab et al., 2017)')
    parser.add_argument('--data_dir', help='raw input data directory')
    parser.add_argument('--save_dir', help='save directory')
    parser.add_argument('--pretrain_data_path', help='data path for pretraining decoder')
    parser.add_argument('--use_lower', action='store_true')
    args = parser.parse_args()
    
    main(args)