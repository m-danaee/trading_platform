"""Joint long/short portfolio simulation.

Phase 2 and RB score each direction independently so that a weak short search
cannot suppress a useful long rule.  Deployment, however, owns one account:
there is one net position per symbol, repeated same-side signals are ignored,
and an opposite signal closes/reverses at the next tradable open.  This module
provides that final portfolio view while reusing the CPU engine's exact rule
matching and first-touch outcome contract.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    _build_entries_from_rule_set,
    _build_entries_from_signal_masks,
    _safe_profit_factor,
    _sortino_ratio_from_returns,
    compute_entry_time_priority,
)
from gpu_fuzzy_trader.backtest.symbol_conditions import get_normalized_symbol_array


class JointPortfolioEngine:
    """Evaluate long and short rule books in one net-position account."""

    def __init__(self, df: pd.DataFrame, **constants) -> None:
        self.df = df
        self.constants = constants
        self.long_engine = CPUBacktestEngine(
            df, {}, "long", **constants,
        )
        self.short_engine = CPUBacktestEngine(
            df, {}, "short", **constants,
        )
        self.initial_capital = float(
            constants.get("initial_capital", _cfg.INITIAL_CAPITAL)
        )
        self.leverage = float(
            constants.get("leverage", _cfg.LEVERAGE)
        )
        self.fee_pct = float(constants.get("fee_pct", _cfg.FEE_PCT))
        self.fee_rate = self.fee_pct / 100.0
        self.spread_bps = float(constants.get("spread_bps", _cfg.SPREAD_BPS))
        self.slippage_bps = float(constants.get("slippage_bps", _cfg.SLIPPAGE_BPS))
        self.effective_fee_rate = (
            self.fee_rate
            + (self.spread_bps / 10000.0)
            + (self.slippage_bps / 10000.0)
        )
        self.max_hold_candles = int(
            constants.get("max_hold_candles", _cfg.MAX_HOLD_CANDLES)
        )
        self.max_total_exposure_rate = float(
            constants.get(
                "max_total_exposure_pct", _cfg.MAX_TOTAL_EXPOSURE_PCT,
            )
        ) / 100.0
        self.min_position_notional = float(
            constants.get("min_position_notional", _cfg.MIN_POSITION_NOTIONAL)
        )
        self.symbols = self.long_engine.symbols
        self.datetimes = self.long_engine.datetimes
        self.entry_datetimes = self.long_engine.entry_datetimes
        self.symbol_bar_index = self.long_engine.symbol_bar_index
        self.entry_price = self.long_engine.entry_price
        self.entry_time_priority = compute_entry_time_priority(
            self.datetimes, len(df),
        )

    @staticmethod
    def _empty_metrics(initial_capital: float) -> dict:
        return {
            "direction": "joint",
            "total_return_pct": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "account_ruined": False,
            "loss_count": 0,
            "time_closed_count": 0,
            "raw_signal_count": 0,
            "executed_trades": 0,
            "final_equity": initial_capital,
            "profit_factor": 0.0,
            "avg_position_notional": 0.0,
            "skipped_min_notional_count": 0,
            "skipped_same_side_count": 0,
            "reversal_count": 0,
            "max_simultaneous_positions": 0,
            "max_total_open_exposure": 0.0,
            "per_symbol_metrics": {},
            "per_symbol_metrics_available": True,
        }

    def _entries(
        self,
        direction: str,
        strategy: dict,
        history_df: pd.DataFrame | None = None,
    ) -> list[dict]:
        engine = self.long_engine if direction == "long" else self.short_engine
        if isinstance(strategy, dict) and "mtf_candidate" in strategy:
            from gpu_fuzzy_trader.mtf.candidate import (
                HierarchicalStrategyCandidate,
            )
            from gpu_fuzzy_trader.mtf.runtime import evaluate_candidate_rule_masks

            candidate = HierarchicalStrategyCandidate.from_dict(
                strategy["mtf_candidate"]
            )
            signal_masks, _stats, _audit = evaluate_candidate_rule_masks(
                candidate,
                self.df,
                history_df=history_df,
            )
            entries = _build_entries_from_signal_masks(
                self.df,
                signal_masks,
                candidate.lwc_rules,
                row_priority=self.entry_time_priority,
                context_mask=engine._context_mask,
            )
            for entry in entries:
                entry["direction"] = direction
            return entries
        rules = strategy.get("rules_set", []) if isinstance(strategy, dict) else []
        normalized = None
        if any(
            isinstance(rule, dict)
            and (
                rule.get("symbols")
                or any(
                    str(c).strip().lower().startswith(
                        ("symbol is ", "[symbol] is ")
                    )
                    for c in rule.get("conditions", [])
                )
            )
            for rule in rules
        ):
            normalized = get_normalized_symbol_array(self.df)
        entries = _build_entries_from_rule_set(
            self.df,
            rules,
            engine._condition_mask_cache,
            row_priority=self.entry_time_priority,
            normalized_symbols=normalized,
            context_mask=engine._context_mask,
        )
        for entry in entries:
            entry["direction"] = direction
        return entries

    def _close_position(
        self,
        pos: dict,
        *,
        equity: float,
        peak_equity: float,
        max_drawdown_pct: float,
        logs: list[dict],
        forced_return_pct: float | None = None,
        forced_index: int | None = None,
    ) -> tuple[float, float, float]:
        if forced_return_pct is not None:
            pos["price_return_pct"] = float(forced_return_pct)
            pos["gross_pnl"] = pos["position_notional"] * (
                float(forced_return_pct) / 100.0
            )
            pos["net_pnl"] = pos["gross_pnl"] - pos["fee"]
            pos["exit_reason"] = "Reverse"
            pos["release_index"] = int(forced_index)
        equity_before = equity
        equity += float(pos["net_pnl"])
        peak_equity = max(peak_equity, equity)
        drawdown = (
            (peak_equity - equity) / peak_equity * 100.0
            if peak_equity > 0.0 else 100.0
        )
        max_drawdown_pct = max(max_drawdown_pct, drawdown)
        log_index = pos.get("log_index")
        if log_index is not None and 0 <= int(log_index) < len(logs):
            release_idx = int(pos["release_index"])
            close_time = (
                self.datetimes[release_idx]
                if release_idx < len(self.datetimes)
                else (self.datetimes[-1] if len(self.datetimes) else np.nan)
            )
            logs[int(log_index)].update({
                "Release_Index": release_idx,
                "Close_Time": close_time,
                "Exit_Reason": pos["exit_reason"],
                "Price_Return_Pct": float(pos["price_return_pct"]),
                "Gross_PnL": float(pos["gross_pnl"]),
                "Net_PnL": float(pos["net_pnl"]),
                "Equity_Before_Close": equity_before,
                "Equity_After": equity,
                "Account_Return_Pct": (
                    equity / self.initial_capital - 1.0
                ) * 100.0,
                "High_Water_Mark": peak_equity,
                "Drawdown_Pct": drawdown,
                "Account_Status": (
                    "ACCOUNT_RUINED" if equity <= 0.0 else "ACTIVE"
                ),
                "Realized": True,
            })
        return equity, peak_equity, max_drawdown_pct

    def simulate(
        self,
        strategies: dict[str, dict],
        *,
        return_logs: bool = True,
        history_df: pd.DataFrame | None = None,
    ) -> tuple[dict, pd.DataFrame]:
        entries = self._entries(
            "long", strategies.get("long", {}), history_df=history_df
        )
        entries.extend(
            self._entries(
                "short", strategies.get("short", {}), history_df=history_df
            )
        )
        entries.sort(
            key=lambda row: (
                int(row.get("entry_priority", 0)),
                int(row.get("idx", 0)),
                0 if row.get("direction") == "long" else 1,
                int(row.get("rule_index", 0)),
                int(row.get("symbol_priority", 0)),
            )
        )

        metrics = self._empty_metrics(self.initial_capital)
        metrics["raw_signal_count"] = len(entries)
        if not entries:
            return metrics, pd.DataFrame()

        equity = self.initial_capital
        peak_equity = equity
        max_drawdown = 0.0
        open_positions: list[dict] = []
        by_symbol: dict[str, dict] = {}
        logs: list[dict] = []
        returns_for_sortino: list[float] = []
        gross_wins = 0.0
        gross_losses = 0.0
        position_sum = 0.0
        symbol_stats: dict[str, dict] = {}

        def realize(pos: dict, forced_return: float | None = None, at: int | None = None):
            nonlocal equity, peak_equity, max_drawdown, gross_wins, gross_losses
            equity, peak_equity, max_drawdown = self._close_position(
                pos,
                equity=equity,
                peak_equity=peak_equity,
                max_drawdown_pct=max_drawdown,
                logs=logs,
                forced_return_pct=forced_return,
                forced_index=at,
            )
            pnl = float(pos["net_pnl"])
            if pnl > 0:
                metrics["win_rate"] += 1
                gross_wins += pnl
            elif pnl < 0:
                metrics["loss_count"] += 1
                gross_losses += abs(pnl)
            if pos.get("exit_reason") == f"Time_{self.max_hold_candles}":
                metrics["time_closed_count"] += 1
            returns_for_sortino.append(
                pnl / max(equity - pnl, 1e-9)
            )
            sym = str(pos["symbol"])
            stat = symbol_stats.setdefault(
                sym, {"trade_count": 0, "win_count": 0, "net_pnl": 0.0}
            )
            stat["net_pnl"] += pnl
            if pnl > 0:
                stat["win_count"] += 1

        def release_due(current_index: int, current_time_priority: int | None = None):
            nonlocal open_positions
            still_open: list[dict] = []
            for pos in open_positions:
                is_due = (
                    pos.get("release_time_priority", pos["release_index"]) <= current_time_priority
                    if current_time_priority is not None and "release_time_priority" in pos
                    else int(pos["release_index"]) <= current_index
                )
                if is_due:
                    realize(pos)
                    symbol = str(pos["symbol"])
                    by_symbol.pop(symbol, None)
                else:
                    still_open.append(pos)
            open_positions = still_open

        for entry in entries:
            idx = int(entry["idx"])
            entry_priority = int(entry.get("entry_priority", idx))
            release_due(idx, current_time_priority=entry_priority)
            symbol = str(self.symbols[idx])
            direction = str(entry["direction"])
            current = by_symbol.get(symbol)
            if current is not None:
                if current["direction"] == direction:
                    metrics["skipped_same_side_count"] += 1
                    continue
                # Opposite signal: close at the next tradable open represented
                # by label_open_next at this signal row, then reverse.
                old_entry = float(current["entry_price"])
                new_entry = float(self.entry_price[idx])
                if old_entry > 0.0 and np.isfinite(new_entry):
                    forced = (
                        (new_entry - old_entry) / old_entry * 100.0
                        if current["direction"] == "long"
                        else (old_entry - new_entry) / old_entry * 100.0
                    )
                    realize(current, forced_return=forced, at=idx)
                open_positions.remove(current)
                by_symbol.pop(symbol, None)
                metrics["reversal_count"] += 1

            engine = (
                self.long_engine if direction == "long" else self.short_engine
            )
            tp = float(entry["tp"])
            sl = float(entry["sl"])
            capital_pct = float(entry["capital_pct"])
            target = equity * capital_pct / 100.0 * self.leverage
            open_exposure = sum(
                float(pos["position_notional"]) for pos in open_positions
            )
            remaining = max(
                0.0,
                equity * self.max_total_exposure_rate * self.leverage
                - open_exposure,
            )
            notional = min(target, remaining)
            if notional < self.min_position_notional:
                metrics["skipped_min_notional_count"] += 1
                continue

            returns, _offsets, releases = engine._get_trade_bundle(tp, sl)
            price_return = float(returns[idx])
            if not np.isfinite(price_return):
                continue
            _result, exit_reason = engine._build_trade_outcome_single(
                idx, tp, sl,
            )
            gross = notional * price_return / 100.0
            fee = notional * self.effective_fee_rate
            net = gross - fee
            release_idx = int(releases[idx])
            release_time_priority = (
                int(self.entry_time_priority[release_idx])
                if release_idx < len(self.entry_time_priority)
                else int(np.max(self.entry_time_priority) + 1 if len(self.entry_time_priority) else len(self.df))
            )
            position = {
                "symbol": symbol,
                "direction": direction,
                "entry_price": float(self.entry_price[idx]),
                "entry_index": idx,
                "release_index": release_idx,
                "release_time_priority": release_time_priority,
                "position_notional": notional,
                "price_return_pct": price_return,
                "exit_reason": exit_reason,
                "gross_pnl": gross,
                "fee": fee,
                "net_pnl": net,
                "log_index": len(logs) if return_logs else None,
            }
            if return_logs:
                logs.append({
                    "Trade_Number": len(logs) + 1,
                    "Direction": direction,
                    "Rule_Index": int(entry["rule_index"]),
                    "Rule_TP": tp,
                    "Rule_SL": sl,
                    "Symbol": symbol,
                    "Entry_Time": self.entry_datetimes[idx],
                    "Entry_Index": idx,
                    "Symbol_Bar_Index": int(self.symbol_bar_index[idx]),
                    "Entry_Price": float(self.entry_price[idx]),
                    "Release_Index": release_idx,
                    "Close_Time": None,
                    "Exit_Reason": exit_reason,
                    "Price_Return_Pct": price_return,
                    "Position_Notional": notional,
                    "Gross_PnL": gross,
                    "Fee": fee,
                    "Net_PnL": net,
                    "Account_Status": "OPEN",
                    "Realized": False,
                })
            open_positions.append(position)
            by_symbol[symbol] = position
            symbol_stats.setdefault(
                symbol, {"trade_count": 0, "win_count": 0, "net_pnl": 0.0},
            )["trade_count"] += 1
            metrics["executed_trades"] += 1
            position_sum += notional
            metrics["max_simultaneous_positions"] = max(
                metrics["max_simultaneous_positions"], len(open_positions)
            )
            metrics["max_total_open_exposure"] = max(
                metrics["max_total_open_exposure"],
                sum(float(pos["position_notional"]) for pos in open_positions),
            )

        release_due(
            len(self.df),
            current_time_priority=int(
                np.max(self.entry_time_priority) + 1
                if len(self.entry_time_priority)
                else len(self.df)
            ),
        )
        metrics["final_equity"] = equity
        metrics["total_return_pct"] = (
            equity / self.initial_capital - 1.0
        ) * 100.0
        metrics["max_drawdown_pct"] = max_drawdown
        metrics["win_rate"] = (
            float(metrics["win_rate"]) / metrics["executed_trades"] * 100.0
            if metrics["executed_trades"] else 0.0
        )
        metrics["profit_factor"] = _safe_profit_factor(
            gross_wins, gross_losses,
        )
        metrics["sortino_ratio"] = _sortino_ratio_from_returns(
            returns_for_sortino, scale_by_trades=True,
        )
        metrics["avg_position_notional"] = (
            position_sum / metrics["executed_trades"]
            if metrics["executed_trades"] else 0.0
        )
        metrics["account_ruined"] = bool(equity <= 0.0)
        metrics["per_symbol_metrics"] = {
            symbol: {
                "trade_count": int(value["trade_count"]),
                "win_rate": (
                    value["win_count"] / value["trade_count"] * 100.0
                    if value["trade_count"] else 0.0
                ),
                "net_pnl": float(value["net_pnl"]),
            }
            for symbol, value in symbol_stats.items()
        }
        return metrics, pd.DataFrame(logs)
