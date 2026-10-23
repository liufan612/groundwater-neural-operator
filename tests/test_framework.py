"""Code-only checks of channel contracts, recurrent gradients and frozen heads."""
import unittest
import torch
import run_nonnegative_oneway as N
import run_e1_five_models as F


@unittest.skipUnless(torch.cuda.is_available(), 'Original training loops require CUDA')
class FrameworkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        # Six forecast updates cross a five-step truncated-BPTT boundary.
        self.d = dict(static=torch.randn(1, 2, 6, 30, 50, device='cuda'),
                      forcing=torch.randn(7, 1, 2, 9, 30, 50, device='cuda'),
                      active_mask=torch.ones(1, 2, 30, 50, device='cuda'),
                      head=torch.randn(7, 1, 2, 30, 50, device='cuda'),
                      concentration=torch.rand(7, 1, 2, 30, 50, device='cuda'),
                      chd_mask=torch.zeros(7, 1, 2, 30, 50, dtype=torch.bool, device='cuda'),
                      chd_values=torch.zeros(7, 1, 2, 30, 50, device='cuda'),
                      times_days=torch.arange(7, device='cuda')[:, None] * 180.,
                      scenario_index=torch.tensor([0], device='cuda'))
        self.norm = {'head': {'mean': [0.], 'std': [1.]},
                     'concentration': {'mean': [1.], 'std': [2.]}}

    def test_five_backbone_concentration_updates(self):
        for name in ['unet', 'fno', 'afno', 'uno', 'deeponet']:
            with self.subTest(model=name):
                net = N.build_operator(name, 38, 2, N.C.CAPACITIES[name]).cuda()
                opt = torch.optim.AdamW(net.parameters(), lr=.0003)
                before = next(net.parameters()).detach().clone()
                paths, losses, _ = F.hard_rollout(net, self.d, self.d['head'], self.norm, True)
                self.assertTrue(torch.isfinite(losses).all())
                self.assertGreaterEqual(float((paths * 2 + 1).min()), 0.)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
                opt.step()
                self.assertFalse(torch.equal(before, next(net.parameters())))

    def test_two_stage_freezes_head_and_dual_updates_both(self):
        for route in ['sequential', 'dual']:
            with self.subTest(route=route):
                N.SCHEDULE = route
                h, c = N.networks(42)
                if route == 'sequential':
                    h.requires_grad_(False)
                    c = N.build_operator('afno', 38, 2, N.C.CAPACITIES['afno']).cuda()
                    N.HEADS = self.d['head'].transpose(0, 1).clone()
                else:
                    N.HEADS = None
                _, losses = N.trajectory(h, c, self.d, self.norm, backward=True)
                self.assertTrue(torch.isfinite(losses).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters()))
                if route == 'sequential':
                    self.assertTrue(all(p.grad is None for p in h.parameters()))
                else:
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))


if __name__ == '__main__':
    unittest.main()
