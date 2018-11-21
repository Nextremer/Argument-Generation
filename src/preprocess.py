# -*- coding:utf-8 -*-
import numpy as np
import json
import pickle
from collections import defaultdict, Counter
import argparse
import wikipedia

from utils import *


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


def rel_tagging(seq, match_idx, label, length):
    """
    'None': 0
    'Support': 1
    'Attack': 2
    'For': 3
    'Against': 4
    """
    if label == 'supports':
        seq[match_idx:match_idx+length] = [1]*length
    elif label == 'attacks':
        seq[match_idx:match_idx+length] = [2]*length
    elif label == 'For':
        seq[match_idx:match_idx+length] = [3]*length
    elif label == 'Against':
        seq[match_idx:match_idx+length] = [4]*length

    return seq


def dist_tagging(seq, match_idx, label, length):
    """
    0: 'None'
    -11 ~ -1 => 1 ~ 11
    1 ~ 9 => 12 ~ 20
    """
    if -12 < label < 0:
        seq[match_idx:match_idx+length] = [label+12]*length
    elif 0 < label < 10:
        seq[match_idx:match_idx+length] = [label+11]*length

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
    rel_seq = [0]*len(context)
    dist_seq = [0]*len(context)
    comp_dict = defaultdict(lambda: len(comp_dict))
    for ann in anns:
        if ann[0].startswith('T'):
            label = ann[1]
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

            comp_dict[ann[0]] = (match_idx, pattern_length)

            type_seq = type_tagging(type_seq, match_idx, label, pattern_length)

        elif ann[0].startswith('A'):
            match_idx, pattern_length = comp_dict[ann[2]]
            label = ann[3]
            rel_seq = rel_tagging(rel_seq, match_idx, label, pattern_length)

        elif ann[0].startswith('R'):
            sorted_comp_dict = sorted(dict(comp_dict).items(), key=lambda x: x[1][0])
            sorted_comp = [i[0] for i in sorted_comp_dict]
            match_idx, pattern_length = comp_dict[ann[4]]
            label = ann[1]
            arg1 = ann[4]
            arg2 = ann[7]
            arg1_idx = sorted_comp.index(arg1)
            arg2_idx = sorted_comp.index(arg2)
            dist = arg2_idx - arg1_idx
            rel_seq = rel_tagging(rel_seq, match_idx, label, pattern_length)
            dist_seq = dist_tagging(dist_seq, match_idx, dist, pattern_length)

    return type_seq, rel_seq, dist_seq


def main(args):
    # tokenized topics, tokenized contexts, annotations
    topics, contexts = read_txt(args.data_dir)
    anns = read_ann(args.data_dir)
    assert len(topics) == len(contexts) == len(anns)
    
    if args.use_lower:
        topics = [[w.lower() for w in topic] for topic in topics]
        contexts = [[w.lower() for w in context] for context in contexts]

    type_seqs = []
    rel_seqs = []
    dist_seqs = []
    essay_num = 0
    for context, ann in zip(contexts, anns):
        type_seq, rel_seq, dist_seq = get_labels(context, ann, essay_num, args.use_lower)
        type_seqs.append(type_seq)
        rel_seqs.append(rel_seq)
        dist_seqs.append(dist_seq)
        essay_num += 1

    # save tokenized topics, tokenized contexts, type sequences
    if args.save_dir:
        with open(args.save_dir+'topics.pickle', 'wb') as f:
            pickle.dump(topics, f)
        with open(args.save_dir+'contexts.pickle', 'wb') as f:
            pickle.dump(contexts, f)
        with open(args.save_dir+'type_seqs.pickle', 'wb') as f:
            pickle.dump(type_seqs, f)
        with open(args.save_dir+'rel_seqs.pickle', 'wb') as f:
            pickle.dump(rel_seqs, f)
        with open(args.save_dir+'dist_seqs.pickle', 'wb') as f:
            pickle.dump(dist_seqs, f)

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

    if args.save_dir:
        with open(args.save_dir+'topics_sent.pickle', 'wb') as f:
            pickle.dump(topics_sent, f)
        with open(args.save_dir+'contexts_sent.pickle', 'wb') as f:
            pickle.dump(contexts_sent, f)


    topics, contexts_para = read_paragraph(args.data_dir)

    if args.use_lower:
        topics = [[w.lower() for w in topic] for topic in topics]
        contexts_para = [[[w.lower() for w in para] for para in context_para] for context_para in contexts_para]

    contexts_len = [np.cumsum([len(paragraph) for paragraph in context])[:-1] for context in contexts_para]

    # ラベル列(tyoe, rel, dist)をパラグラフごとに分割
    type_seqs = [np.split(type_seq, context_len) for type_seq, context_len in zip(type_seqs, contexts_len)]
    rel_seqs = [np.split(rel_seq, context_len) for rel_seq, context_len in zip(rel_seqs, contexts_len)]
    dist_seqs = [np.split(dist_seq, context_len) for dist_seq, context_len in zip(dist_seqs, contexts_len)]

    mc_idxs = [list(filter(lambda x: 5 in type_seq[x], range(len(type_seq))))[0] for type_seq in type_seqs]

    contexts_compressed = [np.concatenate(context_para[:mc_idx+1]) for mc_idx, context_para in zip(mc_idxs, contexts_para)]
    type_seqs_compressed = [np.concatenate(type_seq[:mc_idx+1]) for mc_idx, type_seq in zip(mc_idxs, type_seqs)]
    rel_seqs_compressed = [np.concatenate(rel_seq[:mc_idx+1]) for mc_idx, rel_seq in zip(mc_idxs, rel_seqs)]
    dist_seqs_compressed = [np.concatenate(dist_seq[:mc_idx+1]) for mc_idx, dist_seq in zip(mc_idxs, dist_seqs)]

    if args.save_dir:
        with open(args.save_dir+'contexts_compressed.pickle', 'wb') as f:
            pickle.dump(contexts_compressed, f)
        with open(args.save_dir+'type_seqs_compressed.pickle', 'wb') as f:
            pickle.dump(type_seqs_compressed, f)
        with open(args.save_dir+'rel_seqs_compressed.pickle', 'wb') as f:
            pickle.dump(rel_seqs_compressed, f)
        with open(args.save_dir+'dist_seqs_compressed.pickle', 'wb') as f:
            pickle.dump(dist_seqs_compressed, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess corpus(Stab et al., 2017)')
    parser.add_argument('--data_dir', help='raw input data directory')
    parser.add_argument('--save_dir', help='save directory')
    parser.add_argument('--pretrain_data_path', help='data path for pretraining decoder')
    parser.add_argument('--use_lower', action='store_true')
    args = parser.parse_args()
    
    main(args)