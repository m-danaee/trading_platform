from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12
KEY_COLS = ["datetime", "symbol"]

REQUIRED_RAW_COLS = ["datetime", "symbol", "open", "high", "low", "close", "volume"]

SAFE27_FEATURES = [
    'atr_pct_14',
    'band_bb_percB_20_2',
    'channel_pos_20',
    'close_location_value',
    'dmi_balance_14',
    'dollar_vol_rel_20',
    'downside_semivol_20',
    'efficiency_ratio_20',
    'ema_gap_atr_20',
    'ema_slope_atr_20_5',
    'macd_hist_atr',
    'parkinson_vol_20',
    'percent_b_20',
    'range_compression_20_100',
    'realized_vol_20',
    'ret_autocorr_1_30',
    'return_skew_30',
    'roc_10',
    'rsi_centered_14',
    'tr_to_atr_14',
    'up_close_ratio_5',
    'upside_semivol_20',
    'vol_over_ema20',
    'vol_over_median20',
    'mom_stoch_rsi_14_14_3',
    'vol_ratio_20_100',
    'log_range_over_vol_100',
]


def safe_numeric(x: Any) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.to_numeric(x, errors="coerce")


def safe_div(a: Any, b: Any) -> pd.Series:
    aa = safe_numeric(a)
    bb = safe_numeric(b).replace(0, np.nan)
    return aa / bb


def ema(s: pd.Series, span: int) -> pd.Series:
    return safe_numeric(s).ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(g: pd.DataFrame) -> pd.Series:
    h = safe_numeric(g["high"])
    l = safe_numeric(g["low"])
    prev_close = safe_numeric(g["close"]).shift(1)
    return pd.concat(
        [h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    close = safe_numeric(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_RAW_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw input is missing required columns: {missing}")

    out = raw.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
    out["datetime"] = out["datetime"].dt.tz_localize(None)
    if out["datetime"].isna().any():
        raise ValueError(f"Invalid datetime rows: {int(out['datetime'].isna().sum())}")

    out["symbol"] = pd.to_numeric(out["symbol"], errors="coerce")
    if out["symbol"].isna().any():
        raise ValueError(f"Invalid symbol rows: {int(out['symbol'].isna().sum())}")
    out["symbol"] = out["symbol"].astype(int)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[["open", "high", "low", "close", "volume"]].isna().any().any():
        bad = out[["open", "high", "low", "close", "volume"]].isna().sum()
        raise ValueError(f"Raw OHLCV contains missing/non-numeric values:\n{bad}")

    dup = int(out.duplicated(KEY_COLS).sum())
    if dup:
        raise ValueError(f"Raw input has {dup} duplicate (datetime, symbol) rows")

    return out.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def compute_raw_candidates_one_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("datetime").reset_index(drop=True)
    o = safe_numeric(g["open"])
    h = safe_numeric(g["high"])
    l = safe_numeric(g["low"])
    c = safe_numeric(g["close"])
    v = safe_numeric(g["volume"])

    ret = c.pct_change()
    logc = np.log(c.replace(0, np.nan))
    logret = logc.diff()
    tr = true_range(g)
    atr14_ema = ema(tr, 14)

    body = c - o
    body_abs = body.abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    sma20 = c.rolling(20, min_periods=20).mean()
    std20_samp = c.rolling(20, min_periods=20).std(ddof=1)
    bb_upper_samp = sma20 + 2 * std20_samp
    bb_lower_samp = sma20 - 2 * std20_samp
    bb_width_samp = bb_upper_samp - bb_lower_samp

    rolling_high20 = h.rolling(20, min_periods=20).max()
    rolling_low20 = l.rolling(20, min_periods=20).min()
    rolling_high100 = h.rolling(100, min_periods=100).max()
    rolling_low100 = l.rolling(100, min_periods=100).min()

    ema20 = ema(c, 20)
    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd = ema12 - ema26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal

    rsi14 = rsi_wilder(c, 14)
    rsi_min14 = rsi14.rolling(14, min_periods=14).min()
    rsi_max14 = rsi14.rolling(14, min_periods=14).max()
    stoch_rsi = safe_div(rsi14 - rsi_min14, rsi_max14 - rsi_min14)

    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_wilder = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        / tr_wilder.replace(0, np.nan)
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        / tr_wilder.replace(0, np.nan)
    )

    dollar_vol = c * v
    roll_close_hi100 = c.rolling(100, min_periods=100).max()
    roll_close_lo100 = c.rolling(100, min_periods=100).min()
    close_roll_log_range = np.log(safe_div(roll_close_hi100, roll_close_lo100))
    rv100 = logret.rolling(100, min_periods=100).std(ddof=0)

    out = pd.DataFrame(
        {
            "datetime": g["datetime"].to_numpy(),
            "symbol": g["symbol"].to_numpy(),
            "atr_pct_14_ema": safe_div(atr14_ema, c),
            "percent_b_20_samp_signed": 2
            * safe_div(c - bb_lower_samp, bb_width_samp)
            - 1,
            "cand_body_signed_to_tr": safe_div(body, tr),
            "body_to_tr": safe_div(body_abs, tr),
            "channel_pos_20_hl": safe_div(
                c - rolling_low20, rolling_high20 - rolling_low20
            ),
            "close_location_signed_hl": 2 * safe_div(c - l, h - l) - 1,
            "dmi_balance_wilder_14": safe_div(
                plus_di - minus_di, plus_di + minus_di
            ),
            "dollar_vol_rel_20_ema": safe_div(dollar_vol, ema(dollar_vol, 20)),
            "downside_semivol_20_logret": logret.clip(upper=0)
            .pow(2)
            .rolling(20, min_periods=20)
            .mean()
            .pow(0.5),
            "efficiency_ratio_20": safe_div(
                (c - c.shift(20)).abs(),
                c.diff().abs().rolling(20, min_periods=20).sum(),
            ),
            "ema_gap_atr20_ema": safe_div(c - ema20, atr14_ema),
            "ema_slope_atr20_5_ema": safe_div(ema20 - ema20.shift(5), atr14_ema),
            "cand_lower_wick_to_tr": safe_div(lower_wick, tr),
            "macd_hist_atr_ema": safe_div(macd_hist, atr14_ema),
            "parkinson_vol_20": (
                np.log(safe_div(h, l)).pow(2).rolling(20, min_periods=20).mean()
                / (4 * np.log(2))
            ).pow(0.5),
            "range_compression_20_100": safe_div(
                rolling_high20 - rolling_low20,
                rolling_high100 - rolling_low100,
            ),
            "realized_vol_20_logret": logret.rolling(
                20, min_periods=20
            ).std(ddof=0),
            "logret_autocorr_1_30": logret.rolling(
                30, min_periods=30
            ).corr(logret.shift(1)),
            "return_skew_30_logret": logret.rolling(
                30, min_periods=30
            ).skew(),
            "roc_10_pct": c.pct_change(10),
            "rsi_wilder_centered": (rsi14 - 50) / 50,
            "tr_to_atr14_ema": safe_div(tr, atr14_ema),
            "up_close_ratio_5": (c.diff() > 0)
            .astype(float)
            .rolling(5, min_periods=5)
            .mean(),
            "cand_upper_wick_to_tr": safe_div(upper_wick, tr),
            "upside_semivol_20_logret": logret.clip(lower=0)
            .pow(2)
            .rolling(20, min_periods=20)
            .mean()
            .pow(0.5),
            "vol_over_ema20_extra": safe_div(v, ema(v, 20)),
            "vol_over_median20_extra": safe_div(
                v, v.rolling(20, min_periods=20).median()
            ),
            "stoch_rsi_wilder_14_sma3": stoch_rsi.rolling(
                3, min_periods=3
            ).mean(),
            "vol_ratio_ema_20_100": safe_div(ema(v, 20), ema(v, 100)),
            "log_close_rollrange_over_rv_100": safe_div(
                close_roll_log_range, rv100
            ),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def build_candidate_frame(raw: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sym, g in raw.groupby("symbol", sort=True):
        print(f"Computing safe27 raw candidates: symbol={sym}, rows={len(g)}")
        parts.append(compute_raw_candidates_one_symbol(g))
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["symbol", "datetime"])
        .reset_index(drop=True)
    )


def shift_by_symbol(x: pd.Series, symbols: pd.Series, lag: int) -> pd.Series:
    xx = safe_numeric(x).reset_index(drop=True)
    ss = safe_numeric(symbols).reset_index(drop=True)
    if lag == 0:
        return xx
    return xx.groupby(ss, sort=False).shift(lag)


def apply_pre_transform(x: pd.Series, name: str | None) -> pd.Series:
    x = safe_numeric(x)
    name = "x" if name is None else str(name)
    if name == "x":
        return x
    if name == "-x":
        return -x
    if name == "abs_x":
        return x.abs()
    if name == "-abs_x":
        return -x.abs()
    if name == "one_minus_x":
        return 1 - x
    if name == "neg_one_minus_x":
        return -(1 - x)
    if name == "sqrt_abs_x":
        return np.sqrt(x.abs())
    if name == "log1p_abs_x":
        return np.log1p(x.abs())
    raise ValueError(f"Unsupported pre_transform: {name}")


def apply_transform(
    x: pd.Series, symbols: pd.Series, spec: dict[str, Any]
) -> pd.Series:
    family = spec["family"]
    transform_spec = spec["transform_spec"]
    method = transform_spec["method"]
    scope = transform_spec.get("scope", "per_symbol")
    x = safe_numeric(x).reset_index(drop=True)
    symbols = safe_numeric(symbols).reset_index(drop=True).astype(int)
    out = pd.Series(np.nan, index=x.index, dtype="float64")

    def finish(z: pd.Series) -> pd.Series:
        if family == "positive":
            return z.clip(0, 1)
        if family == "signed":
            return (2 * z - 1).clip(-1, 1)
        raise ValueError(f"Unsupported family {family!r}")

    if method == "fit_interior":
        params = transform_spec.get("params", {})
        if scope != "per_symbol":
            p = transform_spec
            z = (x - float(p["low"])) / float(p["width"])
            return finish(z)
        for sym in sorted(symbols.unique()):
            idx = symbols == sym
            p = params.get(str(int(sym)))
            if p is None:
                raise KeyError(f"No scaler parameters for symbol {sym}")
            z = (x.loc[idx] - float(p["low"])) / float(p["width"])
            out.loc[idx] = finish(z).to_numpy()
        return out

    if method.startswith("q"):
        params = transform_spec.get("params", {})
        if scope != "per_symbol":
            low = float(transform_spec["low"])
            high = float(transform_spec["high"])
            return finish((x - low) / (high - low))
        for sym in sorted(symbols.unique()):
            idx = symbols == sym
            p = params.get(str(int(sym)))
            if p is None:
                raise KeyError(f"No quantile parameters for symbol {sym}")
            low = float(p["low"])
            high = float(p["high"])
            out.loc[idx] = finish((x.loc[idx] - low) / (high - low)).to_numpy()
        return out

    if method == "linear_fit_interior_clip":
        params = transform_spec.get("params", {})
        for sym in sorted(symbols.unique()):
            idx = symbols == sym
            p = params.get(str(int(sym)))
            if p is None:
                raise KeyError(f"No linear parameters for symbol {sym}")
            y = float(p["a"]) * x.loc[idx] + float(p["b"])
            out.loc[idx] = y.clip(0, 1).to_numpy()
        return out

    raise ValueError(f"Unsupported transform method: {method}")


def build_safe27(raw: pd.DataFrame, specs: dict[str, Any]) -> pd.DataFrame:
    raw = _prepare_raw(raw)
    cand = build_candidate_frame(raw)
    if not raw[KEY_COLS].equals(cand[KEY_COLS]):
        raise RuntimeError("Candidate frame key alignment failed")

    symbols = raw["symbol"]
    out = raw[KEY_COLS].copy()

    for i, feature in enumerate(SAFE27_FEATURES, start=1):
        if feature not in specs:
            raise KeyError(f"Missing spec for {feature}")
        spec = specs[feature]
        candidate = spec["candidate"]
        if candidate not in cand.columns:
            raise KeyError(f"Candidate {candidate!r} required by {feature!r} was not built")

        print(f"Transforming feature {i:02d}/27: {feature}")
        x = shift_by_symbol(cand[candidate], symbols, int(spec.get("lag", 0)))
        x = apply_pre_transform(x, spec.get("pre_transform", "x"))
        values = apply_transform(x, symbols, spec)
        out[feature] = values

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 27 validated features from continuous raw 5-minute OHLCV."
    )
    parser.add_argument("--raw", required=True, help="Continuous raw OHLCV CSV")
    parser.add_argument(
        "--spec", default="specs/safe27_specs.json", help="Safe27 spec JSON"
    )
    parser.add_argument("--out", required=True, help="Output CSV with datetime, symbol, 27 features")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    specs = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if list(specs.keys()) != SAFE27_FEATURES:
        missing = [x for x in SAFE27_FEATURES if x not in specs]
        extra = [x for x in specs if x not in SAFE27_FEATURES]
        if missing or extra:
            raise ValueError(f"Spec mismatch. missing={missing}, extra={extra}")

    result = build_safe27(raw, specs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved safe27 features: {out_path}")
    print(f"Shape: {result.shape}")
    print(f"Date range: {result['datetime'].min()} -> {result['datetime'].max()}")
    print("Missing counts (largest first):")
    print(result[SAFE27_FEATURES].isna().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
