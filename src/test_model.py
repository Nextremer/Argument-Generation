from model import *
import numpy as np
import unittest

np.random.seed(1234)

class AttentionTest(unittest.TestCase):
    
    def test_attention(self):
        attention = Attention(200, 150)
        ehs = [np.random.randn(10, 400).astype(np.float32), np.random.randn(9, 400).astype(np.float32)]
        dhs = [np.random.randn(8, 200).astype(np.float32), np.random.randn(7, 200).astype(np.float32)]
        result = attention(ehs, dhs)
        self.assertTrue(all(np.isclose(result[0].data[0][:3], [0.2097658,  -0.13039063,  0.36767963])))
        self.assertTrue(all(np.isclose(result[1].data[-1][-3:], [0.20084219, -0.43477857,  0.7479435])))

        
        
if __name__ == '__main__':
    unittest.main()
