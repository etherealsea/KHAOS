import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

try:
    from khaos.optimization.pine_replica import run_khaos_simulation
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Mock Data
df = pd.DataFrame({'close': np.random.rand(100) * 100 + 1000})

params = {
    'alpha': 0.1,
    'hurst_len': 30,
    'calib_len': 50,
    'sigma_trig': 1.5,
    'sigma_exit': 0.5,
    'pe_len': 30
}

weights = {
    'q': np.random.rand(6),
    'r': np.random.rand(6),
    'm': np.random.rand(6)
}

print("Running simulation...")
try:
    sig_top, sig_bot, gf, ekf = run_khaos_simulation(df, params, weights)
    print("Simulation successful")
    print("Signals:", np.sum(sig_top), np.sum(sig_bot))
except Exception as e:
    print(f"Simulation failed: {e}")
    import traceback
    traceback.print_exc()
