import os
import tempfile
from gpu_fuzzy_trader.reporting.reporter import Reporter

def test():
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            reporter = Reporter()
            history = [
                {'generation': 0, 'mean_sortino_ratio': 0.5, 'best_sortino_ratio': 0.8},
                {'generation': 1, 'mean_sortino_ratio': 0.6, 'best_sortino_ratio': 0.9}
            ]
            # Added "long" as the direction argument
            path = reporter.plot_phase2_pnl(history, direction="long")
            print(f"Generated path: {path}")
            assert os.path.exists(path), f"Path {path} does not exist"
            assert path.endswith("phase2_long_pnl.png"), f"Path {path} does not end with phase2_long_pnl.png"
            print("Success: Phase 2 PnL plot generated correctly.")
        finally:
            os.chdir(old_cwd)

if __name__ == "__main__":
    test()
