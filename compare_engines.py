import sys
import os
import pandas as pd
import numpy as np

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from tests.property.test_gpu_engine_properties import _make_parity_df, _make_engines

def run_comparison():
    n = 12
    entry = 100.0
    tp = 102.0
    sl = 98.0
    mid = 100.5
    
    # Building a deterministic dataframe with a mix of TP/SL outcomes
    # Row 0: TP
    # Row 1: SL
    # Row 2: Close (mid)
    # Row 3: TP
    # Row 4: SL
    # Row 5: Close
    # Row 6: TP
    # Row 7: SL
    # Row 8: Close
    # Row 9: TP
    # Row 10: SL
    # Row 11: Close
    
    label_max = [tp, mid, mid, tp, mid, mid, tp, mid, mid, tp, mid, mid]
    label_min = [mid, sl, mid, mid, sl, mid, mid, sl, mid, mid, sl, mid]
    label_close = [tp, sl, mid, tp, sl, mid, tp, sl, mid, tp, sl, mid]
    mbm = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    
    df = _make_parity_df(n, label_max, label_min, label_close, mbm, entry=entry)
    direction = "long"
    gpu_eng, cpu_eng = _make_engines(df, direction, max_hold_candles=3)
    
    # Binary chromosome [1] matches "feat_binary" IS 1
    chromosome = [1]
    rule_set = [("[feat_binary] IS Active (1)", "AND")]
    
    gpu_results = gpu_eng.simulate_rule_batch(np.array([chromosome]))[0]
    cpu_results = cpu_eng.simulate_rule_set(rule_set)
    
    metrics = ["executed_trades", "total_return_pct", "sortino_ratio", "max_drawdown_pct", "win_rate", "profit_factor"]
    
    print(f"{'Metric':<20} | {'CPU':<15} | {'GPU':<15}")
    print("-" * 55)
    for m in metrics:
        print(f"{m:<20} | {cpu_results[m]:<15.6f} | {gpu_results[m]:<15.6f}")
    
    cpu_sortino = cpu_results['sortino_ratio']
    gpu_sortino = gpu_results['sortino_ratio']
    
    if not np.isclose(cpu_sortino, gpu_sortino, rtol=1e-4):
        diff = abs(cpu_sortino - gpu_sortino) / (abs(cpu_sortino) + 1e-9) * 100
        print(f"\nSortino Ratio Difference: {diff:.6f}%")

if __name__ == "__main__":
    run_comparison()
