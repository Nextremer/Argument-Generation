from model import *
from train import *
import numpy as np
import unittest
import random
import chainer
from chainer.cuda import cupy as cp
from chainer import cuda

np.random.seed(1234)
random.seed(1234)
if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(1234)

class AttentionTest(unittest.TestCase):
    
    def test_attention(self):
        attention = Attention(400, 200, 150)
        backends.cuda.get_device(0).use()
        attention.to_gpu(0)
        ehs = [cp.random.randn(10, 400).astype(cp.float32), cp.random.randn(9, 400).astype(cp.float32)]
        dhs = [cp.random.randn(8, 200).astype(cp.float32), cp.random.randn(7, 200).astype(cp.float32)]
        result = attention(ehs, dhs)
        print(result[0].data[0][:3])
        print([-0.52674127, -0.12444207 , 0.7915659 ])
        print(result[1].data[-1][-3:])
        print([-0.2755296,   0.47057515,  0.1896592 ])
        self.assertTrue(all(np.isclose(cuda.to_cpu(result[0].data[0][:3]), [-0.52674127, -0.12444207 , 0.7915659 ])))
        self.assertTrue(all(np.isclose(cuda.to_cpu(result[1].data[-1][-3:]), [-0.2755296,   0.47057515,  0.1896592 ])))


    def test_bleu(self):
        hypothesis = ['This', 'is', 'cat']
        reference = ['This', 'is', 'a', 'cat']
        references = [reference]
        list_of_references = [references]
        list_of_hypotheses = [hypothesis]
        bleu = bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)


if __name__ == '__main__':
    unittest.main()
