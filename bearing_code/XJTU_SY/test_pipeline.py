"""Integrity tests; all fabricated inputs here are tests, never experimental results."""
import unittest
import tempfile
import copy
from pathlib import Path
import numpy as np
import torch
from data import config_load,proxy_labels,preprocess_record
from model import PrognosticGATBiLSTM,hypergraph

class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config_load(Path(__file__).with_name('config.json'))
        self.cfg.update({'encoder_channels':4,'feature_dim':8,'hidden_dim':8,
                         'heads':2,'dropout':0.,'encoder_chunk_size':16})
        torch.set_num_threads(2);torch.manual_seed(4)

    def test_no_future_in_proxy(self):
        rms=np.r_[np.linspace(1,1.1,10),np.linspace(1.1,5,20)]
        original=proxy_labels(rms,self.cfg)
        rms[20:]=10000
        changed=proxy_labels(rms,self.cfg)
        for i in [0,1,2]:np.testing.assert_array_equal(original[i][:20],changed[i][:20])

    def test_hyperedge_dedup_and_isolation(self):
        x=torch.tensor([[[1.,0.],[1.,0.],[-1.,0.]]])
        h,a=hypergraph(x,.7)
        self.assertEqual(int((h.sum(1)>0).sum()),1)
        self.assertEqual(float(h[0,2].sum()),0)
        self.assertTrue(a[0].diagonal().all())

    def test_block_matches_independent_histories_and_backward(self):
        for task in ['stage','rul']:
            model=PrognosticGATBiLSTM(self.cfg,task).eval()
            t=torch.randn(6,2,32);f=torch.randn(6,2,16);hist=3
            block=model.forward_block(t,f,hist)
            separate=torch.cat([model.forward_block(t[i:i+hist],f[i:i+hist],hist) for i in range(4)])
            torch.testing.assert_close(block,separate,rtol=1e-4,atol=1e-5)
            block.sum().backward()
            self.assertIsNotNone(model.time_conv.weight.grad)
            self.assertIsNotNone(model.neighbor_linear.weight.grad)
            self.assertTrue(torch.isfinite(model.time_conv.weight.grad).all())

    def test_future_records_do_not_change_past_prediction(self):
        model=PrognosticGATBiLSTM(self.cfg,'rul').eval()
        t=torch.randn(7,2,32);f=torch.randn(7,2,16)
        original=model.forward_block(t,f,3)
        t[5:]=1000;f[5:]=1000
        changed=model.forward_block(t,f,3)
        torch.testing.assert_close(original[:3],changed[:3])

    def test_preprocessing_shape_and_rms(self):
        a=np.sin(np.arange(32768)*2*np.pi/32)
        t,f,r=preprocess_record(a,1024)
        self.assertEqual(t.shape,(32,1024));self.assertEqual(f.shape,(32,512))
        self.assertAlmostEqual(r,2**-.5,places=6)
        self.assertEqual(int(f[0].argmax()),32)

if __name__=='__main__':unittest.main(verbosity=2)
