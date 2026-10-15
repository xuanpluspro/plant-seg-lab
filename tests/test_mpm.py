import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from model import OSSSeg
from model.mpm import MPM
from utils.runtime import FORMAT, load_model
from train_medical import parse_args


class MPMContracts(unittest.TestCase):
    def test_shapes_branch_sharing_and_gradients(self):
        torch.set_num_threads(2)
        model = MPM(32)
        inputs, fused, cam = [], [], []
        handles = [model.conv5.register_forward_hook(lambda m,i,o: inputs.append(i[0].shape)),
                   model.Conv.register_forward_pre_hook(lambda m,i: fused.append(i[0].shape)),
                   model.ca.register_forward_pre_hook(lambda m,i: cam.append(i[0].shape))]
        x = torch.randn(2,32,16,24,requires_grad=True)
        y = model(x)
        for h in handles: h.remove()
        self.assertEqual(inputs,[torch.Size([2,8,16,24])]*2)
        self.assertEqual(fused,[torch.Size([2,64,16,24])])
        self.assertEqual(cam,[x.shape])
        self.assertEqual(y.shape,x.shape)
        self.assertEqual(model.conv5[0].groups,1)
        self.assertEqual(model.conv5[0].kernel_size,(5,5))
        self.assertEqual(model.conv3[0].kernel_size,(3,3))
        self.assertEqual(model.conv1[0].kernel_size,(1,1))
        self.assertEqual(model.Conv.act.weight.shape,(32,1,7,7))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_default_checkpoint(self):
        model = OSSSeg(width=16).eval()
        self.assertIsInstance(model.mpm,MPM)
        x = torch.randn(1,3,64,96)
        with torch.no_grad(): expected = model(x)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mpm.pth'
            torch.save(dict(format=FORMAT,model_config=model.model_config,model=model.state_dict()),path)
            restored,_=load_model(path,torch.device('cpu'))
            with torch.no_grad(): torch.testing.assert_close(expected,restored(x),rtol=0,atol=0)

    def test_cli_has_no_model_variants(self):
        with patch('sys.argv',['train_medical.py','--from-scratch']):
            args=parse_args()
        self.assertFalse(hasattr(args,'cssp_module'))
        self.assertFalse(hasattr(args,'mpm_variant'))
        self.assertFalse(hasattr(args,'adpm'))
