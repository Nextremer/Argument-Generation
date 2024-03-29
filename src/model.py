# -*- coding: utf-8 -*-
# created by Tomohiko Abe
# created at 2018/10/12

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, backends, Variable, serializers

import numpy as np

from pretrain import *


EOS = 0
UNK = 1
EOL1 = 7
EOL2 = 5
EOL3 = 21



class Scorer(chainer.Chain):
    #encoder hidden statesとdecoder hidden stateのスコア関数
    def __init__(self, enc_n_units, dec_n_units, attn_n_units):
        super(Scorer, self).__init__()
        with self.init_scope():
            self.W_a = L.Linear(dec_n_units, enc_n_units)
            self.W_e = L.Linear(enc_n_units, attn_n_units)
            self.W_d = L.Linear(dec_n_units, attn_n_units)
            self.v = L.Linear(attn_n_units, 1)
        
        self.attn_n_units = attn_n_units
            
    def dot(self, ehs, dhs):
        return F.matmul(ehs, dhs[:, None])
    
    def general(self, ehs, dhs):
        return F.matmul(ehs, self.W_a(dhs[None,:]).T)
        
    def concat(self, ehs, dhs):
        # ehs: (batchsize, enc_max_length, enc_hidden_units)
        # dhs: (batchsize, dec_max_length, dec_hidden_units)
        batchsize, enc_T, enc_H = ehs.shape
        batchsize, dec_T, dec_H = dhs.shape
        
        # d: (batchsize * dec_max_length, dec_hidden_units)
        dhs = F.concat(dhs, axis=0)
        # d: (batchsize * dec_max_length, attn_units)
        d = self.W_d(dhs)
        # d: (batchsize, dec_max_length, attn_units)
        d = F.split_axis(d, batchsize, axis=0)
        # d: (batchsize, dec_max_length, attn_units)
        d = F.pad_sequence(d)
        # d: (batchsize, dec_max_length, 1, attn_units)
        d = F.expand_dims(d, axis=2)
        # d: (batchsize, dec_max_length, enc_max_length, attn_units)
        d = F.broadcast_to(d, (batchsize, dec_T, enc_T, self.attn_n_units))
        
        # ehs: (batchsize * enc_max_length, attn_units)
        ehs = F.concat(ehs, axis=0)
        # e: (batchsize * enc_max_length, attn_units)
        e = self.W_e(ehs)
        # e: (batchsize, enc_max_length, attn_units)
        e = F.split_axis(e, batchsize, axis=0)
        # e: (batchsize, enc_max_length, attn_units)
        e = F.pad_sequence(e)
        # e: (batchsize, 1, enc_max_length, attn_units)
        e = F.expand_dims(e, axis=1)
        # e: (batchsize, dec_max_length, enc_max_length, attn_units)
        e = F.broadcast_to(e, (batchsize, dec_T, enc_T, self.attn_n_units))
        
        # h: (batchsize * dec_max_length, enc_max_length, attn_units)
        h = F.concat(e+d, axis=0)
        # h: (batchsize * dec_max_length * enc_max_length, attn_units)
        h = F.concat(h, axis=0)
        # score: (batchsize * dec_max_length * enc_max_length, 1)
        score = self.v(F.tanh(h))
        # score: (batchsize, dec_max_length * enc_max_length, 1)
        score = F.split_axis(score, batchsize, axis=0)
        # score: (batchsize, dec_max_length, enc_max_length)
        score = [F.pad_sequence(F.split_axis(s, dec_T, axis=0))[:,:,0] for s in score]
        
        return score
        


class Attention(chainer.Chain):
    
    def __init__(self, enc_n_units, dec_n_units, attn_n_units):
        super(Attention, self).__init__()
        with self.init_scope():
            self.W = L.Linear(enc_n_units+dec_n_units, dec_n_units)
            self.scorer = Scorer(enc_n_units, dec_n_units, attn_n_units)
            
    def __call__(self, ehs, dhs):
        # len(ehs) = batchsize
        # ehs[i]: (time step, enc_hidden_units)
        e_len = [len(eh) for eh in ehs]
        d_len = [len(dh) for dh in dhs]
        
        # ehs: (batchsize, enc_max_length, enc_hidden_units)
        # dhs: (batchsize, dec_max_length, dec_hidden_units)
        ehs = F.pad_sequence(ehs, padding=0)
        dhs = F.pad_sequence(dhs, padding=0)
        
        batchsize, enc_T, enc_H = ehs.shape
        batchsize, dec_T, dec_H = dhs.shape
        
        # len(score): batchsize
        # score[i]: (dec_max_length, enc_max_length)
        score = self.scorer.concat(ehs, dhs)
        for s, e_l, d_l in zip(score, e_len, d_len):
            s[d_l:].data -= 1e30
            s[:, e_l:].data -= 1e30
        score = F.pad_sequence(score)
        # alp: (batchsize, dec_max_length, enc_max_length)
        alp = F.softmax(score, axis=2)
        # ehs: (batchsize, 1, enc_max_length, enc_hidden_units)
        ehs = F.expand_dims(ehs, axis=1)
        # ehs: (batchsize, dec_max_length, enc_max_length, enc_hidden_units)
        ehs = F.broadcast_to(ehs, (batchsize, dec_T, enc_T, enc_H))
        # alp: (batchsize, dec_max_length, enc_max_length, 1)
        alp = F.expand_dims(alp, axis=3)
        # alp: (batchsize, dec_max_length, enc_max_length, enc_hidden_units)
        alp = F.broadcast_to(alp, (batchsize, dec_T, enc_T, enc_H))
        # c: (batchsize, dec_max_length, enc_hidden_units)
        c = F.sum(alp * ehs, axis=2)
        # h: (batchsize, dec_max_length, enc_hidden_units+dec_hidden_units)
        h = F.concat((c, dhs), axis=2)
        # h: (batchsize * dec_max_length, enc_hidden_units+dec_hidden_units)
        h = F.concat(h, axis=0)
        # h: (batchsize * dec_max_length, dec_hidden_units)
        h = F.tanh(self.W(h))
        # hs: (batchsize, dec_max_length, dec_hidden_units)
        hs = F.split_axis(h, batchsize, axis=0)
        
        attn_hs = []
        for h, d_l in zip(hs, d_len):
            attn_hs.append(h[:d_l])

        return attn_hs
    

    
class Model(chainer.Chain):

    def __init__(self, args, w2id, id2w, pretrain_w2id, pretrain_id2w, w2vec):
        n_layers = args.n_layers
        n_layers2 = args.n_layers2
        n_units = args.n_units
        attn_n_units = args.attn_n_units
        dropout = args.dropout
        
        n_vocab = len(w2id)
        n_label1 = 8
        n_label2 = 6
        n_label3 = 22
        l_n_units = 10
        
        if args.pretrained_embed_path:
            self.pretrained_embed = Embed(pretrain_w2id, pretrain_id2w, w2vec, n_units)
            serializers.load_npz(args.pretrained_embed_path, self.pretrained_embed)

            init_W = []
            for w in w2id.keys():
                if w in pretrain_w2id.keys():
                    init_W.extend(self.pretrained_embed(np.array([pretrain_w2id[w]], dtype=np.int32)).data)
                elif w not in pretrain_w2id.keys() and w in w2vec.keys():
                    init_W.append(w2vec[w])
                else:
                    init_W.append(np.random.normal(scale=np.sqrt(2./n_units), size=(n_units, )))
        else:            
            init_W = [w2vec[w] if w in w2vec.keys() \
                      else np.random.normal(scale=np.sqrt(2./n_units), size=(n_units, )) \
                      for w in w2id.keys()]
        init_W = np.asarray(init_W, dtype=np.float32)
        
        super(Model, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=init_W)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout=dropout)
            if args.use_label_in:
                self.embed_l1 = L.EmbedID(n_label1, l_n_units)
                self.embed_l2 = L.EmbedID(n_label2, l_n_units)
                self.embed_l3 = L.EmbedID(n_label3, l_n_units)
                self.W_in = L.Linear(n_units+3*l_n_units, n_units)
            if args.use_rnn3:
                self.decoder2 = L.NStepLSTM(n_layers2, n_units, n_units, dropout=dropout)
            self.W_y = L.Linear(n_units, n_vocab)
            self.W_l1 = L.Linear(n_units, n_label1)
            self.W_l2 = L.Linear(n_units, n_label2)
            self.W_l3 = L.Linear(n_units, n_label3)
            
            self.attention = Attention(n_units, n_units, attn_n_units)
        
            if args.pretrained_decoder_path:
                self.pretrained_decoder = Decoder(n_layers, n_units, dropout)
                serializers.load_npz(args.pretrained_decoder_path, self.pretrained_decoder)
            else:
                self.decoder1 = L.NStepLSTM(n_layers, n_units, n_units, dropout=dropout)                
                
        if args.pretrained_decoder_path and args.gpu >= 0:
            backends.cuda.get_device(args.gpu).use()
            self.pretrained_decoder.to_gpu(args.gpu)
        if args.pretrained_embed_path and args.gpu >= 0:
            backends.cuda.get_device(args.gpu).use()
            self.pretrained_embed.to_gpu(args.gpu)
            
        self.w2id = w2id
        self.id2w = id2w
        self.w2vec = w2vec
        
        self.pretrain_w2id = pretrain_w2id
        self.pretrain_id2w = pretrain_id2w
        
        self.eta = args.eta
        self.pretrained_decoder_path = args.pretrained_decoder_path
        self.pretrained_embed_path = args.pretrained_embed_path
        self.use_label_in = args.use_label_in
        self.use_rnn3 = args.use_rnn3
        self.dropout = args.dropout

    def __call__(self, xs, ys, ls):
        lhs, ls1_out, ls2_out, ls3_out, concat_yhs, concat_ys_out = self.forward(xs, ys, ls)
        if self.use_label_in:
            lhs = [lh[:-1] for lh in lhs]
            ls1_out = [ls[:-1] for ls in ls1_out]
            ls2_out = [ls[:-1] for ls in ls2_out]
            ls3_out = [ls[:-1] for ls in ls3_out]
        else:
            lhs = [lh[1:] for lh in lhs]
            ls1_out = [ls[1:] for ls in ls1_out]
            ls2_out = [ls[1:] for ls in ls2_out]
            ls3_out = [ls[1:] for ls in ls3_out]
        
        concat_lhs = F.concat(lhs, axis=0)
        concat_ls1_out = F.concat(ls1_out, axis=0)
        concat_ls2_out = F.concat(ls2_out, axis=0)
        concat_ls3_out = F.concat(ls3_out, axis=0)
        
        batchsize = len(xs)
        loss_w = F.sum(F.softmax_cross_entropy(self.W_y(concat_yhs), concat_ys_out, reduce='no'))/batchsize
        loss_l1 = F.sum(F.softmax_cross_entropy(self.W_l1(concat_lhs), concat_ls1_out, reduce='no'))/batchsize
        loss_l2 = F.sum(F.softmax_cross_entropy(self.W_l2(concat_lhs), concat_ls2_out, reduce='no'))/batchsize
        loss_l3 = F.sum(F.softmax_cross_entropy(self.W_l3(concat_lhs), concat_ls3_out, reduce='no'))/batchsize

        loss_l = loss_l1 + loss_l2 + loss_l3
        loss = loss_w + self.eta * loss_l
        
        return loss_w, loss_l, loss

    def forward(self, xs, ys, ls):
        batchsize = len(xs)
        ls1, ls2, ls3 = ls
        xs = [self.xp.array(x[::-1], dtype=self.xp.int32) for x in xs]
        ys = [self.xp.array(y, dtype=self.xp.int32) for y in ys]
        ls1 = [self.xp.array(l, dtype=self.xp.int32) for l in ls1]
        ls2 = [self.xp.array(l, dtype=self.xp.int32) for l in ls2]
        ls3 = [self.xp.array(l, dtype=self.xp.int32) for l in ls3]
        
        eos = self.xp.array([EOS], dtype=self.xp.int32)
        eol1 = self.xp.array([EOL1], dtype=self.xp.int32)
        eol2 = self.xp.array([EOL2], dtype=self.xp.int32)
        eol3 = self.xp.array([EOL3], dtype=self.xp.int32)

        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]
        concat_ys_out = F.concat(ys_out, axis=0)

        # 埋め込み
        exs = self.sequence_embed(self.embed, xs)
        eys = self.sequence_embed(self.embed, ys_in)

        if self.use_label_in:
            ls1_in = [F.concat([eol1, l], axis=0) for l in ls1]
            ls2_in = [F.concat([eol2, l], axis=0) for l in ls2]
            ls3_in = [F.concat([eol3, l], axis=0) for l in ls3]

            ls1_out = [F.concat([l, eol1], axis=0) for l in ls1]
            ls2_out = [F.concat([l, eol2], axis=0) for l in ls2]
            ls3_out = [F.concat([l, eol3], axis=0) for l in ls3]

            els1 = self.sequence_embed(self.embed_l1, ls1_in)
            els2 = self.sequence_embed(self.embed_l2, ls2_in)
            els3 = self.sequence_embed(self.embed_l3, ls3_in)
            eys = [F.concat([ey, el1, el2, el3], axis=1) for ey, el1, el2, el3 in zip(eys, els1, els2, els3)]

            eys_len = [len(ey) for ey in eys]
            eys_section = np.cumsum(eys_len[:-1])
            eys = F.concat(eys, axis=0)
            eys = self.W_in(eys)
            eys = F.split_axis(eys, eys_section, axis=0)

        else:
            ls1_out = [F.concat([eol1, l], axis=0) for l in ls1]
            ls2_out = [F.concat([eol2, l], axis=0) for l in ls2]
            ls3_out = [F.concat([eol3, l], axis=0) for l in ls3]

        h, c, ehs = self.encoder(None, None, exs)

        if self.pretrained_decoder_path:
            _, _, dhs = self.pretrained_decoder(h, c, eys)
        else:
            _, _, dhs = self.decoder1(h, c, eys)

        yhs = self.attention(ehs, dhs)
        concat_yhs = F.concat(yhs, axis=0)
        
        if self.use_rnn3:
            _, _, lhs = self.decoder2(None, None, dhs)
            return lhs, ls1_out, ls2_out, ls3_out, concat_yhs, concat_ys_out
        elif self.use_label_in:
            return yhs, ls1_out, ls2_out, ls3_out, concat_yhs, concat_ys_out
        else:
            return dhs, ls1_out, ls2_out, ls3_out, concat_yhs, concat_ys_out
        
        
    def generate(self, xs, max_length):
        batchsize = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            if self.pretrained_embed_path:
                exs = []
                for x in xs:
                    ex = []
                    for w in x[::-1]:
                        if w not in self.w2id.keys() and w not in self.pretrain_w2id.keys() and w in self.w2vec.keys():
                            ex.append(F.expand_dims(self.xp.array(self.w2vec[w], dtype=self.xp.float32), axis=0))
                        elif w not in self.w2id.keys() and w in self.pretrain_w2id.keys():
                            wid = F.concat([self.xp.array([self.pretrain_w2id[w]], dtype=self.xp.int32)], axis=0)
                            ex.append(self.pretrained_embed(wid))
                        else:
                            ex.append(self.embed(self.xp.array([self.w2id.get(w, UNK)], dtype=self.xp.int32)))
                    exs.append(ex)
            else:
                exs = [[F.expand_dims(self.xp.array(self.w2vec[w], dtype=self.xp.float32), axis=0)\
                        if w not in self.w2id.keys() and w in self.w2vec.keys() \
                        else self.embed(self.xp.array([self.w2id.get(w, UNK)], dtype=self.xp.int32)) \
                        for w in x[::-1]] for x in xs]
            exs = [F.concat(x, axis=0) for x in exs]

            h, c, ehs = self.encoder(None, None, exs)

            ys = self.xp.full(batchsize, EOS, dtype=self.xp.int32)
            if self.use_label_in:
                ls1 = self.xp.full(batchsize, EOL1, dtype=self.xp.int32)
                ls2 = self.xp.full(batchsize, EOL2, dtype=self.xp.int32)
                ls3 = self.xp.full(batchsize, EOL3, dtype=self.xp.int32)
            
            h_l, c_l = None, None
            result = []
            for i in range(max_length):
                eys = self.embed(ys)
                if self.use_label_in:
                    els1 = self.embed_l1(ls1)
                    els2 = self.embed_l2(ls2)
                    els3 = self.embed_l3(ls3)

                    eys = [F.concat([ey, el1, el2, el3], axis=0) for ey, el1, el2, el3 in zip(eys, els1, els2, els3)]
                    eys = [F.expand_dims(ey, axis=0) for ey in eys]
                    eys = F.concat(eys, axis=0)
                    eys = self.W_in(eys)
                    eys = F.split_axis(eys, batchsize, axis=0)
                else:
                    eys = F.split_axis(eys, batchsize, axis=0)
                    
                if self.pretrained_decoder_path:
                    h, c, dhs = self.pretrained_decoder(h, c, eys)
                else:
                    h, c, dhs = self.decoder1(h, c, eys)
                yhs = self.attention(ehs, dhs)
                concat_yhs = F.concat(yhs, axis=0)
                wy = self.W_y(concat_yhs)
                ys = self.xp.argmax(wy.data, axis=1).astype(self.xp.int32)
                if self.use_rnn3:
                    h_l, c_l, lhs = self.decoder2(h_l, c_l, dhs)
                    concat_lhs = F.concat(lhs, axis=0)
                elif self.use_label_in:
                    concat_lhs = concat_yhs
                else:
                    concat_lhs = F.concat(dhs, axis=0)
                ls1 = self.xp.argmax(self.W_l1(concat_lhs).data, axis=1).astype(self.xp.int32)
                ls2 = self.xp.argmax(self.W_l2(concat_lhs).data, axis=1).astype(self.xp.int32)
                ls3 = self.xp.argmax(self.W_l3(concat_lhs).data, axis=1).astype(self.xp.int32)

                result.append(ys)
                
        result = backends.cuda.to_cpu(self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)
        
        # EOSを除去
        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            s = []
            for i in y:
                s.append(self.id2w.get(i, 'unk'))
            outs.append(s)
                
        return outs
    
    def sequence_embed(self, embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, axis=0)
        return exs
    
    def get_accuracy(self, xs, ys, ls):
        lhs, ls_out, _, _ = self.forward(xs, ys, ls)
        lhs = [lh[:-1] for lh in lhs]
        ls_out = [ls[:-1] for ls in ls_out]
        
        concat_lhs = F.concat(lhs, axis=0)
        concat_ls_out = F.concat(ls_out, axis=0)
        
        ls_pred = F.argmax(self.W_l(concat_lhs), axis=1)
        
        acc = np.sum(backends.cuda.to_cpu(ls_pred.data) == backends.cuda.to_cpu(concat_ls_out.data))/float(len(concat_ls_out))
        return acc
    
