import copy
import unittest
from gwno.training.trainer import early_stop_update,validate_resume_config

class EarlyStoppingTests(unittest.TestCase):
    def test_small_improvements_accumulate_until_delta(self):
        reference,wait=1.,0
        reference,wait=early_stop_update(.99995,reference,wait,.0001)
        self.assertEqual(wait,1)
        reference,wait=early_stop_update(.99989,reference,wait,.0001)
        self.assertEqual(wait,0)
        for _ in range(100): reference,wait=early_stop_update(1.,reference,wait,.0001)
        self.assertEqual(wait,100)

    def test_resume_extends_budget_but_keeps_optimizer(self):
        old={'training':{'epochs':100,'lr':.0003,'batch_size':64,'early_stopping_patience':15}}
        new=copy.deepcopy(old);new['training'].update(epochs=2000,resume=True,early_stopping_patience=100,early_stopping_min_delta=.0001)
        validate_resume_config(old,new)
        new['training']['lr']=.01
        with self.assertRaises(ValueError):validate_resume_config(old,new)
