# Phase 4 — Walk-Forward Risk Optimization (Optuna)

> **ماژول:** `gpu_fuzzy_trader/phases/phase4_wf_optimizer.py` → `WalkForwardRiskOptimizer`  
> **ورودی:** `validation_25.parquet`, rule set از Phase 3  
> **خروجی:** `outputs/long.json`, `outputs/short.json` (با TP/SL/capital_pct بهینه‌شده)

---

## ۱. هدف Phase 4

بهینه‌سازی پارامترهای ریسک (TP، SL، capital_pct) برای هر rule در rule set. **شرایط rule ها ثابت می‌مانند** — فقط risk parameters تغییر می‌کنند.

**چرا جدا از Phase 2؟** در Phase 2 می‌خواستیم "alpha" (توانایی پیش‌بینی جهت) را پیدا کنیم. در Phase 4 می‌خواهیم بهترین risk/reward ratio را برای آن alpha پیدا کنیم. این جداسازی باعث می‌شود هر phase کار خودش را بهتر انجام دهد.

---

## ۲. Walk-Forward Validation

### الگوریتم تقسیم

```python
def split_validation_walk_forward(val_df, k):
    for symbol, group in val_df.groupby("symbol"):
        g = group.sort_values("datetime")
        n = len(g)
        indices = np.array_split(np.arange(n), k)
        symbol_chunks[symbol] = [g.iloc[idx] for idx in indices]
    
    windows = []
    for i in range(k):
        parts = [symbol_chunks[sym][i] for sym in symbols]
        windows.append(pd.concat(parts))
    
    return windows
```

**مثال با k=2:**
```
validation data (per symbol):
Symbol A: [ردیف 0 تا 249]
Symbol B: [ردیف 0 تا 249]

Window 1: Symbol A [0-124] + Symbol B [0-124]
Window 2: Symbol A [125-249] + Symbol B [125-249]
```

### چرا Walk-Forward؟

بهینه‌سازی روی کل validation ممکن است به پارامترهایی برسد که فقط برای آن دوره زمانی خاص خوب هستند. Walk-forward تضمین می‌کند که پارامترها در **بدترین** window هم قابل قبول باشند.

---

## ۳. Objective Function

```python
def objective(trial):
    # ۱. پیشنهاد پارامترها برای هر rule
    for i in range(n_rules):
        tp = trial.suggest_float(f"tp_{i}", PHASE4_TP_MIN, PHASE4_TP_MAX, step=PHASE4_TP_STEP)
        sl = trial.suggest_float(f"sl_{i}", PHASE4_SL_MIN, PHASE4_SL_MAX, step=PHASE4_SL_STEP)
        cap = trial.suggest_float(f"capital_pct_{i}", PHASE4_CAPITAL_PCT_MIN, PHASE4_CAPITAL_PCT_MAX, step=PHASE4_CAPITAL_STEP)
    
    # ۲. ارزیابی روی هر window
    for split_df in val_splits:
        engine = CPUBacktestEngine(split_df, {}, direction)
        metrics = engine.simulate_rule_set(candidate_rule_set)
        split_sortinos.append(metrics["sortino_ratio"])
        split_drawdowns.append(metrics["max_drawdown_pct"])
    
    # ۳. محاسبه worst-case
    penalty = overalloc_penalty(params_list)
    worst_sortino = min(split_sortinos) - penalty
    worst_drawdown = max(split_drawdowns) + penalty
    
    return worst_sortino, worst_drawdown  # maximize, minimize
```

**Worst-case strategy:** به جای میانگین، بدترین window را در نظر می‌گیریم. این باعث می‌شود پارامترهای انتخاب‌شده در همه شرایط بازار قابل قبول باشند.

---

## ۴. Overallocation Penalty

```python
def _overalloc_penalty(params_list):
    total_cap = sum(p["capital_pct"] for p in params_list)
    return max(0.0, total_cap - 100.0) / 100.0 × PHASE4_TOTAL_CAP_PENALTY
```

اگر مجموع capital_pct همه rule ها از ۱۰۰٪ بیشتر شود، جریمه می‌شود.

---

## ۵. Sampler: NSGA-II vs TPE

### NSGA-II (پیش‌فرض)

```python
sampler = optuna.samplers.NSGAIISampler(seed=seed)
```

**مناسب برای:** بهینه‌سازی چندهدفه. NSGA-II یک Pareto front از پارامترها می‌سازد.

**مزیت:** تنوع بیشتر در Pareto front، کشف trade-off های مختلف بین sortino و drawdown.

### TPE (Tree-structured Parzen Estimator)

```python
sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
```

**مناسب برای:** بهینه‌سازی تک‌هدفه یا وقتی می‌خواهید سریع‌تر به یک نقطه خوب برسید.

**مزیت:** معمولاً سریع‌تر converge می‌کند.

---

## ۶. انتخاب از Pareto Front

```python
def _select_pareto_trial(study, max_worst_dd_pct):
    pareto = study.best_trials
    
    # فیلتر بر اساس max drawdown
    candidates = [t for t in pareto if t.values[1] <= max_worst_dd_pct]
    
    if candidates:
        # از بین کاندیداها، بیشترین worst sortino را انتخاب کن
        return max(candidates, key=lambda t: t.values[0])
    else:
        # fallback: کمترین drawdown، سپس بیشترین sortino
        min_dd = min(t.values[1] for t in pareto)
        tied = [t for t in pareto if t.values[1] == min_dd]
        return max(tied, key=lambda t: t.values[0])
```

**منطق انتخاب:**
1. ابتدا trial هایی که worst drawdown ≤ `PHASE4_MAX_WORST_DRAWDOWN_PCT` دارند
2. از بین آن‌ها، بیشترین worst sortino

---

## ۷. Hard Cap Normalization

```python
def _normalize_capital_pct(rules_set):
    if not PHASE4_HARD_CAP_NORMALIZE:
        return rules_set
    
    total_cap = sum(p["capital_pct"] for p in rules_set)
    if total_cap > MAX_TOTAL_EXPOSURE_PCT:
        scale = MAX_TOTAL_EXPOSURE_PCT / total_cap
        for p in rules_set:
            p["capital_pct"] *= scale
    
    return rules_set
```

**چرا؟** حتی اگر Optuna پارامترهایی پیشنهاد دهد که مجموع capital_pct از ۱۰۰٪ بیشتر است، این normalization تضمین می‌کند که مجموع نهایی ≤ `MAX_TOTAL_EXPOSURE_PCT` باشد.

---

## ۸. Hyperparameters Phase 4

### Search Space

| پارامتر | پیش‌فرض | توضیح |
|---------|---------|-------|
| `PHASE4_TP_MIN` | `2.0` | حداقل Take Profit (%) |
| `PHASE4_TP_MAX` | `4.0` | حداکثر Take Profit (%) |
| `PHASE4_TP_STEP` | `0.2` | گام جستجو برای TP |
| `PHASE4_SL_MIN` | `1.0` | حداقل Stop Loss (%) |
| `PHASE4_SL_MAX` | `2.0` | حداکثر Stop Loss (%) |
| `PHASE4_SL_STEP` | `0.2` | گام جستجو برای SL |
| `PHASE4_CAPITAL_PCT_MIN` | `10.0` | حداقل capital allocation (%) |
| `PHASE4_CAPITAL_PCT_MAX` | `50.0` | حداکثر capital allocation (%) |
| `PHASE4_CAPITAL_STEP` | `5.0` | گام جستجو برای capital |

**اندازه search space:**
```
TP: (4.0 - 2.0) / 0.2 + 1 = 11 مقدار
SL: (2.0 - 1.0) / 0.2 + 1 = 6 مقدار
Capital: (50.0 - 10.0) / 5.0 + 1 = 9 مقدار

برای ۲ rule: 11 × 6 × 9 × 11 × 6 × 9 = 356,994 ترکیب ممکن
```

### پارامترهای اصلی

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE4_N_TRIALS` | `1000` | جستجوی دقیق‌تر، کندتر | سریع‌تر اما ممکن است بهینه نباشد |
| `PHASE4_WF_SPLITS` | `2` | ارزیابی روی window های بیشتر، robust‌تر | سریع‌تر |
| `PHASE4_MAX_WORST_DRAWDOWN_PCT` | `15.0` | پارامترهای با drawdown بیشتر هم قبول می‌شوند | فقط پارامترهای محافظه‌کارانه‌تر |
| `PHASE4_SAMPLER` | `"nsga2"` | `"tpe"` = سریع‌تر converge | - |
| `PHASE4_SEED` | `42` | reproducibility | - |
| `PHASE4_N_JOBS` | `1` | parallel trials (با احتیاط) | - |

### پارامترهای دیگر

| پارامتر | پیش‌فرض | تأثیر |
|---------|---------|-------|
| `PHASE4_TOTAL_CAP_PENALTY` | `2.0` | جریمه overallocation |
| `PHASE4_HARD_CAP_NORMALIZE` | `True` | normalization نهایی capital |

---

## ۹. نکات عملی

### اگر Phase 4 پارامترهای خوبی پیدا نمی‌کند:
- `PHASE4_N_TRIALS` را افزایش دهید (مثلاً 2000)
- `PHASE4_WF_SPLITS` را کاهش دهید (مثلاً 1) تا search space کمتر محدود شود
- `PHASE4_MAX_WORST_DRAWDOWN_PCT` را افزایش دهید (مثلاً 20.0)

### اگر می‌خواهید TP/SL بزرگ‌تر را هم بررسی کنید:
- `PHASE4_TP_MAX` را افزایش دهید (مثلاً 6.0)
- `PHASE4_SL_MAX` را افزایش دهید (مثلاً 3.0)

### اگر Phase 4 خیلی کند است:
- `PHASE4_SAMPLER = "tpe"` را امتحان کنید
- `PHASE4_N_TRIALS` را کاهش دهید
- `PHASE4_N_JOBS` را افزایش دهید (اگر CPU کافی دارید)

### درباره TP/SL Ratio:
- TP/SL ratio پیش‌فرض: 3.0/1.5 = 2.0 (در Phase 2)
- در Phase 4: می‌تواند از 2.0/1.0 = 2.0 تا 4.0/2.0 = 2.0 باشد
- ratio بالاتر = win rate پایین‌تر اما سود بیشتر per trade
- ratio پایین‌تر = win rate بالاتر اما سود کمتر per trade
