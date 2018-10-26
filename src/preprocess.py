# -*- coding:utf-8 -*-
import numpy as np
import json
import pickle
from collections import defaultdict
from utils import *
import argparse



def type_tagging(seq, match_idx, label, length):
    """component typeのラベル付け

    'None':0,
    'Premise':1,
    'Claim':2,
    'MajorClaim':3
    """
    if label == 'MajorClaim':
        seq[match_idx:match_idx+length] = [3]*length
    elif label == 'Claim':
        seq[match_idx:match_idx+length] = [2]*length
    elif label == 'Premise':
        seq[match_idx:match_idx+length] = [1]*length
    return seq


def bio_tagging(seq, match_idx, length):
    """BIOのラベル付け

    'O':0,
    'B':1,
    'I':2
    """
    seq[match_idx] = 1
    seq[match_idx+1:match_idx+length] = [2]*(length-1)
    return seq


def get_labels(context, anns, essay_num, use_lower):
    """1つのcontextに対して、BIOとcomponent typeのラベル系列を得る"""
    bio_seq = [0]*len(context)
    type_seq = [0]*len(context)
    for ann in anns:
        if ann[0].startswith('T'):
            pattern = ann[4:]
            if args.use_lower:
                pattern = [w.lower() for w in pattern]
            pattern_length = len(pattern)
            idxs = [idx for idx, w in enumerate(context) if w == pattern[0]]
            match_idxs = list(filter(lambda x: context[x:x+pattern_length] == pattern, idxs))
            
            if len(match_idxs) == 1:
                match_idx = match_idxs[0]
            elif len(match_idxs) == 2:
                if essay_num == 153:
                    match_idx = match_idxs[np.argmin(match_idxs)]
                elif essay_num == 163 or essay_num == 166 or essay_num == 181:
                    match_idx = match_idxs[np.argmax(match_idxs)]
                elif essay_num == 263:
                    if ann[0] == 'T2':
                        match_idx = match_idxs[np.argmin(match_idxs)]
                    elif ann[0] == 'T14':
                        match_idx = match_idxs[np.argmax(match_idxs)]
                elif essay_num == 266:
                    if ann[0] == 'T1':
                        match_idx = match_idxs[np.argmin(match_idxs)]
                elif essay_num == 285:
                    if ann[0] == 'T4':
                        match_idx = match_idxs[np.argmin(match_idxs)]
                    elif ann[0] == 'T14':
                        match_idx = match_idxs[np.argmax(match_idxs)]
            elif len(match_idxs) == 3:
                if ann[0] == 'T1':
                    match_idx = match_idxs[1]
                elif ann[0] == 'T3':
                    match_idx = match_idxs[np.argmax(match_idxs)]
        
        bio_seq = bio_tagging(bio_seq, match_idx, pattern_length)
        type_seq = type_tagging(type_seq, match_idx, ann[1], pattern_length)

    return bio_seq, type_seq


def main(args):
    # tokenized topics, tokenized contexts, annotations
    topics, contexts = read_txt(args.data_dir, args.use_lower)
    anns = read_ann(args.data_dir)
    assert len(topics) == len(contexts) == len(anns)

    # save tokenized topics, tokenized contexts
    with open(args.save_dir+'topics.pickle', 'wb') as f:
        pickle.dump(topics, f)
    with open(args.save_dir+'contexts.pickle', 'wb') as f:
        pickle.dump(contexts, f)

    bio_seqs = []
    type_seqs = []
    essay_num = 0
    
    for context, ann in zip(contexts, anns):
        bio_seq, type_seq = get_labels(context, ann, essay_num, args.use_lower)
        assert len(bio_seq) == len(type_seq)
        bio_seqs.append(bio_seq)
        type_seqs.append(type_seq)
        essay_num += 1
        
    # save bio_seqs, type_seqs
    with open(args.save_dir+'bio_seqs.pickle', 'wb') as f:
        pickle.dump(bio_seqs, f)
    with open(args.save_dir+'type_seqs.pickle', 'wb') as f:
        pickle.dump(type_seqs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess corpus(Stab et al., 2017)')
    parser.add_argument('--data_dir', help='raw input data directory')
    parser.add_argument('--save_dir', help='save directory')
    parser.add_argument('--use_lower', default=False)
    args = parser.parse_args()
    
    main(args)
