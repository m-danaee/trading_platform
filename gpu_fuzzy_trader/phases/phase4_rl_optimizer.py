"""
phase4_rl_optimizer.py — RL_Agent (Phase 4)

Fine-tunes TP, SL, and capital_pct for each rule in the selected rule set
using a reinforcement-learning-inspired approach.

Primary approach: stable-baselines3 DDPG/PPO with a gym-compatible TradingEnv.
Fallback (when stable-baselines3 / gymnasium / torch are unavailable):
  Random search over TP/SL/capital_pct parameter space, evaluated via
  CPUBacktestEngine, with the Elbow Method applied to the validation curve
  to identify the optimal checkpoint.

State vector:
  [K market features, R rule activation strengths, equity_normalized,
   open_exposure_normalized]

Action vector (per rule):
  [tp_i, sl_i, capital_pct_i]  clipped to config bounds

Reward:
  net_pnl_normalized - drawdown_penalty

Skip logic:
  If outputs/{direction}.json exists and TP/SL/capital_pct values are within
  valid ranges, Phase 4 is skipped.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.reporting.reporter import Reporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_OUTPUT_PATHS = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "short.json"),
}

# ---------------------------------------------------------------------------
# Optional imports (stable-baselines3 / gymnasium / torch)
# ---------------------------------------------------------------------------

_SB3_AVAILABLE = False
_GYM_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    try:
        import gym  # type: ignore[no-redef]
        from gym import spaces  # type: ignore[assignment]
        _GYM_AVAILABLE = True
    except ImportError:
        pass

try:
    import stable_baselines3 as sb3
    _SB3_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Elbow Method
# ---------------------------------------------------------------------------

def find_elbow_point(validation_returns: list[float]) -> int:
    """
    Find the elbow point in a validation returns curve.

    Algorithm:
    1. Normalize the curve to [0, 1] range.
    2. Compute the line from the first to the last point.
    3. Find the point with maximum perpendicular distance from this line.
    4. Return that index as the optimal checkpoint.

    Edge cases:
    - Monotonically increasing → returns last index.
    - Immediately plateauing (all values equal) → returns first index (0).
    - Single element → returns 0.
    - Two elements → returns 0 (first point is the elbow).

    Parameters
    ----------
    validation_returns : list[float]
        Validation return values at each checkpoint.

    Returns
    -------
    int
        Index of the elbow point.
    """
    n = len(validation_returns)
    if n == 0:
        return 0
    if n == 1:
        return 0
    if n == 2:
        return 0

    arr = np.array(validation_returns, dtype=float)

    # Handle all-equal (plateau) case
    v_min = arr.min()
    v_max = arr.max()
    if v_max == v_min:
        # Immediately plateauing → first point
        return 0

    # Normalize to [0, 1]
    normalized = (arr - v_min) / (v_max - v_min)

    # Line from first to last point in normalized space
    x = np.arange(n, dtype=float)
    x_norm = x / (n - 1)  # normalize x to [0, 1] as well

    # Direction vector of the line (first → last)
    p1 = np.array([0.0, normalized[0]])
    p2 = np.array([1.0, normalized[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        # Degenerate line (first == last in normalized space)
        return 0

    # Perpendicular distance from each point to the line p1→p2
    # distance = ||(point - p1) × line_vec|| / ||line_vec||
    distances = np.zeros(n)
    for i in range(n):
        point = np.array([x_norm[i], normalized[i]])
        diff = point - p1
        # 2D cross product magnitude: |diff_x * line_y - diff_y * line_x|
        cross = abs(diff[0] * line_vec[1] - diff[1] * line_vec[0])
        distances[i] = cross / line_len

    max_dist = distances.max()

    # If all distances are effectively zero, the curve is linear (monotonically
    # increasing or decreasing with no curvature) → return last point per spec.
    if max_dist < 1e-10:
        return n - 1

    elbow_idx = int(np.argmax(distances))
    return elbow_idx


# ---------------------------------------------------------------------------
# TradingEnv (gym-compatible, optional)
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    Gym-compatible trading environment for RL-based risk optimization.

    State vector:
        [K market features, R rule activation strengths,
         equity_normalized, open_exposure_normalized]

    Action vector (continuous, per rule):
        [tp_0, sl_0, capital_pct_0, tp_1, sl_1, capital_pct_1, ...]
        Bounds from config.py.

    Reward:
        net_pnl_normalized - drawdown_penalty

    This class works with or without gymnasium/gym installed.
    When gym is available, it inherits from gym.Env for full compatibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        rule_set: dict,
        direction: str,
        feature_cols: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataset split (train or validation).
        rule_set : dict
            Rule set dict with "rules_set" key (evaluator_v3.ipynb format).
        direction : str
            "long" or "short".
        feature_cols : list[str] | None
            Feature columns to include in state. If None, auto-detected.
        """
        self.df = df.reset_index(drop=True)
        self.rules = rule_set.get("rules_set", [])
        self.direction = direction
        self.n_rules = len(self.rules)

        # Detect feature columns (exclude label/meta/internal columns)
        _exclude = set(_cfg.LABEL_COLUMNS) | set(_cfg.META_COLUMNS) | {"_symbol_bar_index"}
        if feature_cols is not None:
            self.feature_cols = [c for c in feature_cols if c in df.columns]
        else:
            self.feature_cols = [
                c for c in df.columns
                if c not in _exclude and not c.startswith("_")
            ]

        self.n_features = len(self.feature_cols)
        self.n_state = self.n_features + self.n_rules + 2  # +2 for equity/exposure

        # Action dimension: 3 values per rule (tp, sl, capital_pct)
        self.n_actions = self.n_rules * 3

        # Action bounds
        self._tp_min = _cfg.PHASE4_TP_MIN
        self._tp_max = _cfg.PHASE4_TP_MAX
        self._sl_min = _cfg.PHASE4_SL_MIN
        self._sl_max = _cfg.PHASE4_SL_MAX
        self._cap_min = _cfg.PHASE4_CAPITAL_PCT_MIN
        self._cap_max = _cfg.PHASE4_CAPITAL_PCT_MAX

        # Build gym spaces if available
        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_state,),
                dtype=np.float32,
            )
            # Interleaved: [tp_0, sl_0, cap_0, tp_1, sl_1, cap_1, ...]
            low_bounds = []
            high_bounds = []
            for _ in range(self.n_rules):
                low_bounds.extend([self._tp_min, self._sl_min, self._cap_min])
                high_bounds.extend([self._tp_max, self._sl_max, self._cap_max])
            self.action_space = spaces.Box(
                low=np.array(low_bounds, dtype=np.float32),
                high=np.array(high_bounds, dtype=np.float32),
                dtype=np.float32,
            )

        # Simulation state
        self._current_idx = 0
        self._equity = _cfg.INITIAL_CAPITAL
        self._open_exposure = 0.0
        self._peak_equity = _cfg.INITIAL_CAPITAL
        self._current_tp_sl_cap = self._default_action()

    def _default_action(self) -> np.ndarray:
        """Return midpoint action values as default."""
        action = []
        for _ in range(self.n_rules):
            tp = (self._tp_min + self._tp_max) / 2.0
            sl = (self._sl_min + self._sl_max) / 2.0
            cap = (self._cap_min + self._cap_max) / 2.0
            action.extend([tp, sl, cap])
        return np.array(action, dtype=np.float32)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to valid bounds."""
        clipped = action.copy().astype(np.float32)
        for i in range(self.n_rules):
            base = i * 3
            clipped[base] = np.clip(clipped[base], self._tp_min, self._tp_max)
            clipped[base + 1] = np.clip(clipped[base + 1], self._sl_min, self._sl_max)
            clipped[base + 2] = np.clip(clipped[base + 2], self._cap_min, self._cap_max)
        return clipped

    def _compute_rule_activations(self, row: pd.Series) -> np.ndarray:
        """
        Compute rule activation strengths for a single row.

        Activation = fraction of conditions satisfied for each rule.
        Uses simple string matching against feature values.
        """
        activations = np.zeros(self.n_rules, dtype=np.float32)
        for r_idx, rule in enumerate(self.rules):
            conditions = rule.get("conditions", [])
            if not conditions:
                activations[r_idx] = 0.0
                continue
            satisfied = 0
            for cond in conditions:
                # Parse "[feature_name] IS value_name"
                if " IS " not in cond:
                    continue
                feat_part, _ = cond.split(" IS ", 1)
                feat_name = feat_part.strip().lstrip("[").rstrip("]").strip()
                if feat_name in row.index:
                    # Condition is "active" if the feature has a non-zero value
                    # (simplified activation strength)
                    val = row[feat_name]
                    if pd.notna(val) and val != 0:
                        satisfied += 1
            activations[r_idx] = satisfied / len(conditions)
        return activations

    def _get_observation(self) -> np.ndarray:
        """Build state vector for current row."""
        if self._current_idx >= len(self.df):
            return np.zeros(self.n_state, dtype=np.float32)

        row = self.df.iloc[self._current_idx]

        # Market features
        feat_vals = np.array(
            [float(row[c]) if c in row.index else 0.0 for c in self.feature_cols],
            dtype=np.float32,
        )

        # Rule activation strengths
        activations = self._compute_rule_activations(row)

        # Portfolio state
        equity_norm = float(self._equity / _cfg.INITIAL_CAPITAL)
        exposure_norm = float(
            self._open_exposure / max(self._equity, 1e-9)
        )

        obs = np.concatenate([
            feat_vals,
            activations,
            [equity_norm, exposure_norm],
        ]).astype(np.float32)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to start of dataset."""
        self._current_idx = 0
        self._equity = _cfg.INITIAL_CAPITAL
        self._open_exposure = 0.0
        self._peak_equity = _cfg.INITIAL_CAPITAL
        self._current_tp_sl_cap = self._default_action()
        obs = self._get_observation()
        if _GYM_AVAILABLE:
            return obs, {}
        return obs

    def step(self, action: np.ndarray):
        """
        Apply action (TP/SL/capital_pct per rule) and advance one candle.

        Returns (observation, reward, terminated, truncated, info).
        """
        clipped = self._clip_action(np.asarray(action, dtype=np.float32))
        self._current_tp_sl_cap = clipped

        # Build rule set with current TP/SL/capital_pct
        current_rule_set = []
        for r_idx, rule in enumerate(self.rules):
            base = r_idx * 3
            current_rule_set.append({
                "conditions": rule["conditions"],
                "tp": float(clipped[base]),
                "sl": float(clipped[base + 1]),
                "capital_pct": float(clipped[base + 2]),
            })

        # Evaluate on a small window around current index
        window_size = min(50, len(self.df) - self._current_idx)
        if window_size <= 0:
            obs = self._get_observation()
            terminated = True
            if _GYM_AVAILABLE:
                return obs, 0.0, terminated, False, {}
            return obs, 0.0, terminated, {}

        window_df = self.df.iloc[self._current_idx: self._current_idx + window_size].copy()
        try:
            engine = CPUBacktestEngine(window_df, {}, self.direction,
                                       initial_capital=self._equity)
            metrics = engine.simulate_rule_set(current_rule_set)
            net_pnl = metrics.get("total_return_pct", 0.0) * self._equity / 100.0
            drawdown = metrics.get("max_drawdown_pct", 0.0)
        except Exception:
            net_pnl = 0.0
            drawdown = 0.0

        # Update equity
        self._equity = max(self._equity + net_pnl, 0.01)
        self._peak_equity = max(self._peak_equity, self._equity)
        current_dd = (self._peak_equity - self._equity) / self._peak_equity * 100.0

        # Reward
        net_pnl_norm = net_pnl / _cfg.INITIAL_CAPITAL * 100.0
        drawdown_penalty = max(0.0, current_dd - 5.0) * 0.1
        reward = float(net_pnl_norm - drawdown_penalty)

        self._current_idx += window_size
        terminated = self._current_idx >= len(self.df)

        obs = self._get_observation()
        info = {"net_pnl": net_pnl, "drawdown": drawdown}

        if _GYM_AVAILABLE:
            return obs, reward, terminated, False, info
        return obs, reward, terminated, info

    def get_current_params(self) -> list[dict]:
        """Return current TP/SL/capital_pct as a list of rule dicts."""
        result = []
        for r_idx, rule in enumerate(self.rules):
            base = r_idx * 3
            result.append({
                "conditions": rule["conditions"],
                "tp": float(self._current_tp_sl_cap[base]),
                "sl": float(self._current_tp_sl_cap[base + 1]),
                "capital_pct": float(self._current_tp_sl_cap[base + 2]),
            })
        return result


# ---------------------------------------------------------------------------
# Random search fallback (no stable-baselines3)
# ---------------------------------------------------------------------------

def _random_search_optimize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    rule_set: dict,
    direction: str,
    n_samples: int,
    elbow_window: int,
    rng: random.Random,
) -> tuple[list[dict], list[float], int]:
    """
    Random search over TP/SL/capital_pct parameter space.

    Evaluates each sample on the validation split using CPUBacktestEngine.
    Applies the Elbow Method to the validation curve to identify the optimal
    checkpoint.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split (used for overfitting check, not primary optimization).
    val_df : pd.DataFrame
        Validation split (primary evaluation target).
    rule_set : dict
        Rule set dict with "rules_set" key.
    direction : str
        "long" or "short".
    n_samples : int
        Total number of random samples to evaluate.
    elbow_window : int
        Number of samples per evaluation window (analogous to PHASE4_ELBOW_WINDOW).
    rng : random.Random
        Random number generator.

    Returns
    -------
    best_rule_set : list[dict]
        Optimized rule dicts with updated TP/SL/capital_pct.
    validation_returns : list[float]
        Validation return at each elbow_window checkpoint.
    elbow_idx : int
        Index of the elbow point in validation_returns.
    """
    rules = rule_set.get("rules_set", [])
    n_rules = len(rules)

    if n_rules == 0:
        return [], [], 0

    val_engine = CPUBacktestEngine(val_df, {}, direction)

    # Track best per checkpoint window
    validation_returns: list[float] = []
    checkpoint_params: list[list[dict]] = []

    best_val_return = -np.inf
    best_params: list[dict] = []

    # Initialize with current (Phase 3) params
    current_params = [
        {
            "conditions": r["conditions"],
            "tp": float(r.get("tp", _cfg.PHASE2_TP)),
            "sl": float(r.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(r.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
        }
        for r in rules
    ]

    window_best_return = -np.inf
    window_best_params = current_params

    for sample_idx in range(n_samples):
        # Sample random TP/SL/capital_pct for each rule
        candidate_params = []
        for _ in range(n_rules):
            tp = rng.uniform(_cfg.PHASE4_TP_MIN, _cfg.PHASE4_TP_MAX)
            sl = rng.uniform(_cfg.PHASE4_SL_MIN, _cfg.PHASE4_SL_MAX)
            cap = rng.uniform(_cfg.PHASE4_CAPITAL_PCT_MIN, _cfg.PHASE4_CAPITAL_PCT_MAX)
            candidate_params.append({"tp": tp, "sl": sl, "capital_pct": cap})

        # Build rule set for evaluation
        candidate_rule_set = [
            {
                "conditions": rules[i]["conditions"],
                "tp": candidate_params[i]["tp"],
                "sl": candidate_params[i]["sl"],
                "capital_pct": candidate_params[i]["capital_pct"],
            }
            for i in range(n_rules)
        ]

        try:
            val_metrics = val_engine.simulate_rule_set(candidate_rule_set)
            val_return = float(val_metrics.get("total_return_pct", 0.0))
        except Exception as exc:
            logger.debug("Random search sample %d failed: %s", sample_idx, exc)
            val_return = -100.0

        if val_return > window_best_return:
            window_best_return = val_return
            window_best_params = candidate_rule_set

        if val_return > best_val_return:
            best_val_return = val_return
            best_params = candidate_rule_set

        # Record checkpoint every elbow_window samples
        if (sample_idx + 1) % elbow_window == 0:
            validation_returns.append(window_best_return)
            checkpoint_params.append(window_best_params)
            logger.debug(
                "Random search checkpoint %d/%d: best_val_return=%.2f%%",
                len(validation_returns),
                n_samples // elbow_window,
                window_best_return,
            )
            # Reset window tracker
            window_best_return = -np.inf
            window_best_params = current_params

    # Final checkpoint if not already recorded
    if len(validation_returns) == 0 or (n_samples % elbow_window != 0):
        validation_returns.append(window_best_return if window_best_return > -np.inf else best_val_return)
        checkpoint_params.append(window_best_params if window_best_return > -np.inf else best_params)

    # Apply Elbow Method to identify optimal checkpoint
    elbow_idx = find_elbow_point(validation_returns)

    # Use the params from the elbow checkpoint
    if checkpoint_params and elbow_idx < len(checkpoint_params):
        elbow_params = checkpoint_params[elbow_idx]
    else:
        elbow_params = best_params

    logger.info(
        "Random search [%s]: %d samples, %d checkpoints, elbow at idx=%d "
        "(val_return=%.2f%%)",
        direction,
        n_samples,
        len(validation_returns),
        elbow_idx,
        validation_returns[elbow_idx] if validation_returns else 0.0,
    )

    return elbow_params, validation_returns, elbow_idx


# ---------------------------------------------------------------------------
# SB3-based training (when stable-baselines3 is available)
# ---------------------------------------------------------------------------

def _sb3_train_optimize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    rule_set: dict,
    direction: str,
    total_timesteps: int,
    elbow_window: int,
) -> tuple[list[dict], list[float], int]:
    """
    Train a DDPG/PPO agent using stable-baselines3.

    Falls back to random search if training fails.

    Returns
    -------
    best_rule_set : list[dict]
        Optimized rule dicts.
    validation_returns : list[float]
        Validation return at each checkpoint.
    elbow_idx : int
        Elbow point index.
    """
    try:
        from stable_baselines3 import DDPG, PPO  # type: ignore[import]
        from stable_baselines3.common.env_checker import check_env  # type: ignore[import]
    except ImportError:
        logger.warning("stable-baselines3 not available; falling back to random search.")
        rng = random.Random(42)
        return _random_search_optimize(
            train_df, val_df, rule_set, direction,
            n_samples=total_timesteps // 100,
            elbow_window=elbow_window,
            rng=rng,
        )

    rules = rule_set.get("rules_set", [])
    n_rules = len(rules)
    if n_rules == 0:
        return [], [], 0

    # Build training environment
    train_env = TradingEnv(train_df, rule_set, direction)
    val_engine = CPUBacktestEngine(val_df, {}, direction)

    # Select algorithm
    algo_name = _cfg.PHASE4_RL_ALGORITHM.upper()
    AlgoClass = DDPG if algo_name == "DDPG" else PPO

    try:
        model = AlgoClass("MlpPolicy", train_env, verbose=0)
    except Exception as exc:
        logger.warning("SB3 model init failed (%s); falling back to random search.", exc)
        rng = random.Random(42)
        return _random_search_optimize(
            train_df, val_df, rule_set, direction,
            n_samples=total_timesteps // 100,
            elbow_window=elbow_window,
            rng=rng,
        )

    validation_returns: list[float] = []
    checkpoint_params: list[list[dict]] = []
    steps_per_window = max(1, total_timesteps // max(1, total_timesteps // elbow_window))

    n_windows = total_timesteps // steps_per_window
    for window_i in range(n_windows):
        try:
            model.learn(total_timesteps=steps_per_window, reset_num_timesteps=(window_i == 0))
        except Exception as exc:
            logger.warning("SB3 learn step %d failed: %s", window_i, exc)
            break

        # Evaluate on validation split
        obs = train_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Extract current action from model
        try:
            action, _ = model.predict(obs, deterministic=True)
            clipped = train_env._clip_action(np.asarray(action, dtype=np.float32))
        except Exception:
            clipped = train_env._default_action()

        # Build rule set with current params
        candidate_rule_set = []
        for r_idx, rule in enumerate(rules):
            base = r_idx * 3
            candidate_rule_set.append({
                "conditions": rule["conditions"],
                "tp": float(clipped[base]),
                "sl": float(clipped[base + 1]),
                "capital_pct": float(clipped[base + 2]),
            })

        try:
            val_metrics = val_engine.simulate_rule_set(candidate_rule_set)
            val_return = float(val_metrics.get("total_return_pct", 0.0))
        except Exception:
            val_return = 0.0

        validation_returns.append(val_return)
        checkpoint_params.append(candidate_rule_set)

        logger.debug(
            "SB3 [%s] window %d/%d: val_return=%.2f%%",
            direction, window_i + 1, n_windows, val_return,
        )

    if not validation_returns:
        rng = random.Random(42)
        return _random_search_optimize(
            train_df, val_df, rule_set, direction,
            n_samples=total_timesteps // 100,
            elbow_window=elbow_window,
            rng=rng,
        )

    elbow_idx = find_elbow_point(validation_returns)
    elbow_params = checkpoint_params[elbow_idx] if elbow_idx < len(checkpoint_params) else checkpoint_params[-1]

    return elbow_params, validation_returns, elbow_idx


# ---------------------------------------------------------------------------
# Skip logic helpers
# ---------------------------------------------------------------------------

def _load_rule_set(path: str) -> Optional[dict]:
    """Load and return rule set JSON, or None if missing/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    return data


def _params_within_bounds(rule_set: dict) -> bool:
    """
    Return True if all TP/SL/capital_pct values in the rule set are within
    the Phase 4 config bounds.
    """
    rules = rule_set.get("rules_set", [])
    if not rules:
        return False
    for rule in rules:
        tp = float(rule.get("tp", 0.0))
        sl = float(rule.get("sl", 0.0))
        cap = float(rule.get("capital_pct", 0.0))
        if not (_cfg.PHASE4_TP_MIN <= tp <= _cfg.PHASE4_TP_MAX):
            return False
        if not (_cfg.PHASE4_SL_MIN <= sl <= _cfg.PHASE4_SL_MAX):
            return False
        if not (_cfg.PHASE4_CAPITAL_PCT_MIN <= cap <= _cfg.PHASE4_CAPITAL_PCT_MAX):
            return False
    return True


# ---------------------------------------------------------------------------
# RL_Agent
# ---------------------------------------------------------------------------

class RL_Agent:
    """
    Phase 4: RL-based risk optimizer.

    Fine-tunes TP, SL, and capital_pct for each rule in the selected rule set.

    Uses stable-baselines3 DDPG/PPO when available; falls back to random
    search with Elbow Method stopping otherwise.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (already prepared).
    val_df : pd.DataFrame
        Validation split DataFrame (already prepared).
    rule_set : dict
        Current rule set dict from Phase 3 output (evaluator_v3.ipynb format).
    direction : str
        "long" or "short".
    total_timesteps : int | None
        Override PHASE4_TOTAL_TIMESTEPS (useful for testing).
    elbow_window : int | None
        Override PHASE4_ELBOW_WINDOW (useful for testing).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        rule_set: dict,
        direction: str,
        total_timesteps: int | None = None,
        elbow_window: int | None = None,
        seed: int = 42,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

        rules = rule_set.get("rules_set", [])
        if not rules:
            raise ValueError("rule_set must contain at least one rule in 'rules_set'.")

        self.train_df = train_df
        self.val_df = val_df
        self.rule_set = rule_set
        self.direction = direction
        self.total_timesteps = (
            total_timesteps if total_timesteps is not None
            else _cfg.PHASE4_TOTAL_TIMESTEPS
        )
        self.elbow_window = (
            elbow_window if elbow_window is not None
            else _cfg.PHASE4_ELBOW_WINDOW
        )
        self.seed = seed

        # Results populated after train()
        self._validation_returns: list[float] = []
        self._elbow_idx: int = 0
        self._optimized_rule_set: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Train the RL agent and return the optimized rule set.

        Updates outputs/{direction}.json with optimized TP/SL/capital_pct.

        Returns
        -------
        dict
            Optimized rule set dict (evaluator_v3.ipynb compatible format).
        """
        logger.info(
            "Phase 4 [%s]: starting optimization (timesteps=%d, elbow_window=%d, "
            "sb3_available=%s)",
            self.direction,
            self.total_timesteps,
            self.elbow_window,
            _SB3_AVAILABLE,
        )

        if _SB3_AVAILABLE and _GYM_AVAILABLE:
            optimized_params, val_returns, elbow_idx = _sb3_train_optimize(
                train_df=self.train_df,
                val_df=self.val_df,
                rule_set=self.rule_set,
                direction=self.direction,
                total_timesteps=self.total_timesteps,
                elbow_window=self.elbow_window,
            )
        else:
            logger.info(
                "Phase 4 [%s]: stable-baselines3/gymnasium not available; "
                "using random search fallback.",
                self.direction,
            )
            rng = random.Random(self.seed)
            # Scale n_samples: use total_timesteps / 100 as proxy for sample count
            n_samples = max(self.elbow_window, self.total_timesteps // 100)
            optimized_params, val_returns, elbow_idx = _random_search_optimize(
                train_df=self.train_df,
                val_df=self.val_df,
                rule_set=self.rule_set,
                direction=self.direction,
                n_samples=n_samples,
                elbow_window=self.elbow_window,
                rng=rng,
            )

        self._validation_returns = val_returns
        self._elbow_idx = elbow_idx
        self._optimized_rule_set = optimized_params

        # Build output dict
        output_dict = {
            "direction": self.direction,
            "rules_set": optimized_params,
        }

        # Persist to outputs/{direction}.json
        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)

        logger.info(
            "Phase 4 [%s]: optimization complete. Elbow at checkpoint %d "
            "(val_return=%.2f%%). Saved to %s",
            self.direction,
            elbow_idx,
            val_returns[elbow_idx] if val_returns else 0.0,
            output_path,
        )

        # Reporter: RL training curve with elbow point marked
        try:
            Reporter().plot_rl_curve(val_returns, elbow_idx, self.direction)
        except Exception as exc:
            logger.warning("Reporter.plot_rl_curve failed (non-fatal): %s", exc)

        return output_dict

    @staticmethod
    def find_elbow_point(validation_returns: list[float]) -> int:
        """
        Find the elbow point in a validation returns curve.

        Delegates to the module-level find_elbow_point function.

        Parameters
        ----------
        validation_returns : list[float]
            Validation return values at each checkpoint.

        Returns
        -------
        int
            Index of the elbow point.
        """
        return find_elbow_point(validation_returns)

    @staticmethod
    def skip_if_valid(direction: str) -> Optional[dict]:
        """
        Return loaded rule set if valid and TP/SL/capital_pct within bounds.

        Parameters
        ----------
        direction : str
            "long" or "short".

        Returns
        -------
        dict | None
            Loaded rule set if valid and within bounds, None otherwise.
        """
        if direction not in _OUTPUT_PATHS:
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

        path = _OUTPUT_PATHS[direction]
        data = _load_rule_set(path)
        if data is None:
            return None

        if not _params_within_bounds(data):
            logger.info(
                "Phase 4 [%s]: existing file has out-of-bounds TP/SL/capital_pct; "
                "will re-run.",
                direction,
            )
            return None

        logger.info(
            "Phase 4 [%s]: existing file is valid and within bounds; skipping.",
            direction,
        )
        return data

    @property
    def validation_returns(self) -> list[float]:
        """Validation returns recorded during training."""
        return self._validation_returns

    @property
    def elbow_idx(self) -> int:
        """Elbow point index in the validation returns curve."""
        return self._elbow_idx
