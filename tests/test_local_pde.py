import unittest
from unittest.mock import patch
import numpy as np
import torch
from gwno.physics.water import rectangular_water_geometry
from gwno.physics.local_pde import local_terms,local_residual_loss,LocalTransportPDE


def fixture(shape=(2,2,3)):
    active=np.ones(shape);bottom=-np.arange(1,shape[0]+1)[:,None,None]*np.ones(shape)
    g=rectangular_water_geometry(np.zeros(shape[1:]),bottom,active,np.ones(shape[2]),np.ones(shape[1]),active,active,active,np.zeros(shape),np.zeros(shape))
    for key in ['dx','dy','porosity']:g[key]=torch.ones(active.size,dtype=torch.float64)
    empty=torch.empty(0,dtype=torch.float64);ids=torch.empty(0,dtype=torch.long)
    packet=dict(chd=dict(cells=ids,head=empty),wel=dict(cells=ids,q=empty,concentration=empty),
        riv=dict(cells=ids,cond=empty,stage=empty,rbot=empty),rch=dict(cells=ids,q=empty),
        evt=dict(cells=ids,rate=empty,area=empty,surface=empty,depth=empty))
    mat={key:torch.zeros(active.size,dtype=torch.float64) for key in ['alh','alv','ath1','ath2','atv','diffc']}
    mat['diffc'].fill_(1.)
    return g,packet,mat


class LocalPDETests(unittest.TestCase):
    def test_opposite_cell_errors_do_not_cancel_in_loss(self):
        r=torch.tensor([1.,-1.],requires_grad=True)
        loss=local_residual_loss(r,torch.ones(2),torch.ones(2,dtype=torch.bool),1.,1.)
        self.assertEqual(float(r.sum()),0.);self.assertEqual(float(loss),1.)
        loss.backward();torch.testing.assert_close(r.grad,torch.tensor([1.,-1.]))

    def test_constant_state_and_vertical_diffusion(self):
        g,p,m=fixture((2,1,1));h=torch.zeros(2,dtype=torch.float64)
        c=torch.ones(2,dtype=torch.float64)
        terms=local_terms(c,c,h,h,g,m,p,1.)
        torch.testing.assert_close(terms['residual'],torch.zeros_like(c))
        c=torch.tensor([2.,0.],dtype=torch.float64,requires_grad=True)
        terms=local_terms(c,c,h,h,g,m,p,1.)
        torch.testing.assert_close(terms['dispersion'],torch.tensor([2.,-2.],dtype=torch.float64))
        terms['residual'].square().sum().backward();self.assertTrue(torch.isfinite(c.grad).all())

    def test_tensor_cross_terms_and_internal_cancellation(self):
        g,p,m=fixture();h=torch.zeros(12,dtype=torch.float64)
        c=torch.arange(12,dtype=torch.float64).square()
        diagonal=torch.eye(3,dtype=torch.float64).expand(12,3,3).clone()
        full=diagonal.clone();full[:,0,1]=.3;full[:,1,0]=.3
        with patch('gwno.physics.local_pde.modflow_dispersion_tensor',return_value=diagonal):
            a=local_terms(c,c,h,h,g,m,p,1.)['dispersion']
        with patch('gwno.physics.local_pde.modflow_dispersion_tensor',return_value=full):
            b=local_terms(c,c,h,h,g,m,p,1.)['dispersion']
        self.assertFalse(torch.allclose(a,b));self.assertAlmostEqual(float(b.sum()),0.,places=10)

    def test_injection_storage_sign_and_callback_gradients(self):
        g,p,m=fixture((1,1,1));h=torch.zeros(1,dtype=torch.float64)
        p['wel']=dict(cells=torch.tensor([0]),q=torch.tensor([2.],dtype=torch.float64),concentration=torch.tensor([3.],dtype=torch.float64))
        c0=torch.zeros(1,dtype=torch.float64);c1=torch.tensor([6.],dtype=torch.float64)
        terms=local_terms(c0,c1,h,h,g,m,p,1.)
        torch.testing.assert_close(terms['residual'],torch.zeros_like(c0))
        c1=torch.tensor([5.],dtype=torch.float64,requires_grad=True)
        callback=LocalTransportPDE(dict(geometry=g,transport=m,packets=[p,p]),1.,'cpu')
        loss=callback(c0,c1,h,h,1,1.);self.assertAlmostEqual(float(loss),1.,places=8)
        loss.backward();self.assertLess(float(c1.grad),0.)


    def test_manufactured_quadratic_anisotropic_interior(self):
        g,p,m=fixture((1,5,5));y,x=torch.meshgrid(torch.arange(5,dtype=torch.float64),torch.arange(5,dtype=torch.float64),indexing='ij')
        c=(x*x+x*y+y*y).reshape(-1);h=torch.zeros_like(c)
        tensor=torch.eye(3,dtype=torch.float64).expand(25,3,3).clone()
        tensor[:,0,1]=.3;tensor[:,1,0]=.3
        with patch('gwno.physics.local_pde.modflow_dispersion_tensor',return_value=tensor):
            terms=local_terms(c,c,h,h,g,m,p,1.)
        self.assertAlmostEqual(float(terms['residual'][12]),-4.6,places=10)

    def test_vectorized_substeps_equal_reference_with_changing_saturation(self):
        g,p,m=fixture();g['convertible'][:6]=True
        g['storage_capacity'].fill_(.01)
        callback=LocalTransportPDE(dict(geometry=g,transport=m,packets=[p,p]),2.,'cpu')
        torch.manual_seed(9)
        c0=torch.rand(12,dtype=torch.float64)
        c1=torch.randn(12,dtype=torch.float64,requires_grad=True)
        h0=torch.full((12,),-.8,dtype=torch.float64);h1=torch.full((12,),.5,dtype=torch.float64)
        reference=callback.reference(c0,c1,h0,h1,1,180.)
        before=torch.autograd.grad(reference,c1)[0]
        fast=callback(c0,c1,h0,h1,1,180.)
        after=torch.autograd.grad(fast,c1)[0]
        torch.testing.assert_close(fast,reference,rtol=1e-12,atol=1e-10)
        torch.testing.assert_close(after,before,rtol=1e-12,atol=1e-10)


if __name__=='__main__':unittest.main()
