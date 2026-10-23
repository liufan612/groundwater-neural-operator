import unittest
import torch
from test_local_pde import fixture
from gwno.physics.local_water_pde import LocalWaterPDE


class LocalWaterTests(unittest.TestCase):
    def test_static_and_opposite_local_errors(self):
        g,p,_=fixture((2,1,1))
        loss=LocalWaterPDE(g,[p,p],1.,'cpu')
        h=torch.ones(2,dtype=torch.float64)
        self.assertEqual(float(loss(h,h,1,180.)),0.)
        h=torch.tensor([2.,1.],dtype=torch.float64,requires_grad=True)
        value=loss(h,h,1,180.)
        self.assertGreater(float(value.detach()),0.)
        value.backward()
        self.assertGreater(float(h.grad[0]),0.)
        self.assertLess(float(h.grad[1]),0.)

    def test_storage_balanced_by_injection(self):
        g,p,_=fixture((1,1,1));g['storage_capacity'].fill_(2.)
        p['wel']=dict(cells=torch.tensor([0]),q=torch.tensor([2.],dtype=torch.float64),
                      concentration=torch.tensor([0.],dtype=torch.float64))
        callback=LocalWaterPDE(g,[p,p],1.,'cpu')
        value=callback(torch.tensor([0.]),torch.tensor([1.]),1,1.)
        self.assertLess(float(value),1e-25)

    def test_known_head_cells_excluded_but_drive_neighbor(self):
        g,p,_=fixture((1,1,2))
        p['chd']=dict(cells=torch.tensor([0]),head=torch.tensor([2.],dtype=torch.float64))
        callback=LocalWaterPDE(g,[p,p],1.,'cpu')
        h=torch.tensor([17.,1.],dtype=torch.float64,requires_grad=True)
        loss=callback(h,h,1,180.);loss.backward()
        self.assertEqual(float(h.grad[0]),0.)
        self.assertLess(float(h.grad[1]),0.)


if __name__=='__main__':unittest.main()
