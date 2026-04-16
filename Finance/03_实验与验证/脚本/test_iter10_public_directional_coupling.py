from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / 'Finance' / '02_核心代码' / '源代码'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from khaos.模型定义.kan import KHAOS_KAN
from khaos.核心引擎.physics import PHYSICS_FEATURE_NAMES


class Iter10PublicDirectionalCouplingTests(unittest.TestCase):
    def test_reversion_logits_respect_directional_floor(self):
        torch.manual_seed(7)
        model = KHAOS_KAN(
            input_dim=len(PHYSICS_FEATURE_NAMES),
            hidden_dim=32,
            output_dim=2,
            layers=3,
            grid_size=10,
            arch_version='iterA4_multiscale',
            horizon_count=1,
            horizon_family_mode='legacy',
        )
        x = torch.randn(16, 20, len(PHYSICS_FEATURE_NAMES))
        main_pred, aux_pred, info = model(x, return_aux=True, return_debug=True)
        self.assertEqual(main_pred.shape[1], 2)
        self.assertIn('directional_floor', info)
        directional_floor = info['directional_floor']
        self.assertEqual(directional_floor.shape[0], main_pred.shape[0])
        reversion_logits = main_pred[:, 1:2]
        self.assertTrue(torch.all(reversion_logits >= directional_floor + 0.10 - 1e-6).item())


if __name__ == '__main__':
    unittest.main()
