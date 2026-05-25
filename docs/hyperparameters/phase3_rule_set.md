# Phase 3 — Rule Set Selection (Greedy + NSGA-II Refinement)

> **ماژول:** `gpu_fuzzy_trader/phases/phase3_rule_set.py` → `Rule_Set_Selector`  
> **ورودی:** `train_75.parquet`, `validation_25.parquet`, pool از Phase 2  
> **خروجی:** `outputs/long.json`, `outputs/short.json`

---

## ۱. هدف Phase 3

انتخاب بهترین **ترکیب** از ۲ تا ۳ rule از pool Phase 2. در Phase 2 هر rule به تنهایی ارزیابی شد؛ در Phase 3 می‌خواهیم بدانیم کدام ترکیب از rule ها با هم بهترین عملکرد را دارند.

**چرا ترکیب مهم است؟** دو rule که هر کدام به تنهایی خوب هستند، ممکن است با هم بد باشند (مثلاً هر دو روی یک نوع بازار کار می‌کنند). یا دو rule که به تنهایی متوسط هستند، با هم عالی باشند (coverage مکمل).

---

## ۲. Search Space

### اندازه search space

اگر pool دارای N rule باشد و می‌خواهیم ترکیب‌های ۲ تا ۳ rule انتخاب کنیم:

```
C(N, 2) + C(N, 3) = N(N-1)/2 + N(N-1)(N-2)/6
```

برای N=500: تقریباً ۲۰ میلیون ترکیب ممکن — exhaustive search غیرممکن است.

### محدودیت‌ها

- **بدون تکرار:** دو rule با condition set یکسان نمی‌توانند در یک rule set باشند
- **Order-independent:** rule set {A, B} و {B, A} یکی هستند (اما در backtest، ترتیب اهمیت دارد — priority-based)

---

## ۳. الگوریتم دو مرحله‌ای

### مرحله ۱: Greedy Construction

```python
greedy_set, n_greedy_evals = greedy_rule_set_search(
    pool=pool,
    val_engine=val_engine,
    train_engine=train_engine,
    min_rules=PHASE3_MIN_RULES,
    max_rules=PHASE3_MAX_RULES,
    ...
)
```

**الگوریتم greedy:**
1. با بهترین rule از pool شروع کن (بر اساس validation sortino)
2. در هر مرحله، rule ای را اضافه کن که بیشترین بهبود را در validation sortino ایجاد می‌کند
3. تا رسیدن به `PHASE3_MAX_RULES` ادامه بده

**مزیت:** سریع است و یک نقطه شروع خوب برای مرحله بعد فراهم می‌کند.

### مرحله ۲: NSGA-II Refinement

```python
initial_pop = _seed_population_from_greedy(greedy_set, pool, pop_size, ...)
pareto_rule_sets, history = _run_nsga2_combinatorial(
    pool=pool,
    val_engine=val_engine,
    train_engine=train_engine,
    pop_size=PHASE3_REFINE_POP_SIZE,
    n_generations=PHASE3_REFINE_GENERATIONS,
    initial_population=initial_pop,
    ...
)
```

Population با greedy solution seed می‌شود، سپس NSGA-II آن را بهبود می‌دهد.

---

## ۴. Fitness Function

### سه Objective (همه minimize می‌شوند)

```python
f1 = -validation_sortino_ratio
f2 = validation_max_drawdown_pct
f3 = -validation_win_rate
```

**چرا validation؟** می‌خواهیم rule set هایی انتخاب کنیم که روی داده‌های ندیده خوب عمل کنند.

### Penalties

#### Coverage Penalty
```python
symbols_with_trades = count_symbols_with_at_least_one_trade(val_metrics)
if symbols_with_trades < PHASE3_MIN_SYMBOL_COVERAGE:
    coverage_penalty = (PHASE3_MIN_SYMBOL_COVERAGE - symbols_with_trades) × large_value
```

**چرا؟** یک rule set که فقط روی ۲ از ۱۰ symbol trade می‌کند، diversification کافی ندارد.

#### Overfitting Penalty
```python
overfitting_penalty = |train_return - val_return| / max(|train_return|, 1.0)
```

**چرا؟** اگر train return خیلی بیشتر از val return باشد، rule set overfit شده است.

#### Duplicate Rule Penalty
```python
if any two rules have identical condition sets:
    duplicate_penalty = large_value
```

#### Val Gate Penalties (مهم‌ترین)

```python
# اگر val_sortino < ratio × train_sortino
if val_sortino < PHASE3_VAL_SORTINO_RATIO_GATE × train_sortino:
    objectives += PHASE3_VAL_GATE_PENALTY

# اگر val_drawdown > ratio × train_drawdown
if val_drawdown > PHASE3_VAL_DRAWDOWN_RATIO_GATE × train_drawdown:
    objectives += PHASE3_VAL_GATE_PENALTY
```

---

## ۵. Eval Cache

```python
cache = build_phase3_eval_cache(pool, train_df, val_df, val_engine)
```

**چه کاری می‌کند؟** برای هر rule در pool، signal mask (کدام ردیف‌ها با این rule match می‌کنند) را از قبل محاسبه می‌کند. این باعث می‌شود ارزیابی ترکیب‌های مختلف بسیار سریع‌تر باشد.

**مزیت:** به جای اجرای `_apply_dynamic_rule` برای هر ترکیب، فقط mask های از پیش محاسبه‌شده را ترکیب می‌کنیم.

---

## ۶. Crossover و Mutation در Rule Set Space

### Crossover
```python
def _crossover_rule_sets(parent_a, parent_b, pool, rng, min_rules, max_rules):
    combined = list(parent_a) + list(parent_b)
    rng.shuffle(combined)
    # deduplicate و trim به max_rules
    # pad به min_rules اگر لازم است
```

### Mutation
```python
def _mutate_rule_set(rule_set, pool, rng, min_rules, max_rules, mutation_rate=0.3):
    # با احتمال mutation_rate:
    # - یک rule را با rule دیگری از pool جایگزین کن
    # - یک rule اضافه کن
    # - یک rule حذف کن
```

**mutation_rate = 0.3:** نسبتاً بالا است تا exploration کافی داشته باشیم.

---

## ۷. Parallel Batch Evaluation

```python
if PHASE3_USE_PARALLEL_BATCH:
    with ProcessPoolExecutor(max_workers=PHASE3_BATCH_WORKERS) as pool:
        results = pool.map(evaluate_rule_set, rule_sets)
```

**چرا ProcessPool؟** هر ارزیابی CPU-bound است. ProcessPool از چند core استفاده می‌کند.

**PHASE3_BATCH_WORKERS:** پیش‌فرض `min(32, cpu_count)`. روی ماشین‌های با CPU زیاد، این می‌تواند Phase 3 را چند برابر سریع‌تر کند.

---

## ۸. انتخاب بهترین Rule Set از Pareto Front

```python
def _select_best_from_pareto(pareto_rule_sets, val_engine, train_engine, cache):
    # rule set با کمترین f1 (بیشترین validation sortino) انتخاب می‌شود
    best_idx = argmin(f1 for each rule_set in pareto_rule_sets)
    return pareto_rule_sets[best_idx]
```

---

## ۹. Hyperparameters Phase 3

### پارامترهای اصلی

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE3_MIN_RULES` | `2` | حداقل rule در هر rule set | - |
| `PHASE3_MAX_RULES` | `3` | حداکثر rule در هر rule set | search space کوچک‌تر |
| `PHASE3_MIN_SYMBOL_COVERAGE` | `7` | rule set باید روی ۷ از ۱۰ symbol trade کند | coverage کمتر مجاز است |
| `PHASE3_REFINE_POP_SIZE` | `100` | جستجوی گسترده‌تر، کندتر | سریع‌تر |
| `PHASE3_REFINE_GENERATIONS` | `80` | بهینه‌سازی بیشتر، کندتر | سریع‌تر |

### پارامترهای Parallel

| پارامتر | پیش‌فرض | تأثیر |
|---------|---------|-------|
| `PHASE3_USE_PARALLEL_BATCH` | `True` | فعال/غیرفعال کردن parallel evaluation |
| `PHASE3_BATCH_WORKERS` | `min(32, cpu_count)` | تعداد worker های parallel |
| `PHASE3_USE_GPU` | `False` | استفاده از JAX GPU برای evaluation |
| `PHASE3_NUMBA_ENABLED` | `True` | استفاده از Numba برای NSGA-II sort |

### پارامترهای Overfitting Control

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE3_USE_TRAIN_TARGET` | `True` | train به عنوان target، val به عنوان gate | - |
| `PHASE3_VAL_SORTINO_RATIO_GATE` | `0.5` | val sortino باید حداقل ۵۰٪ train sortino باشد | gate سخت‌تر |
| `PHASE3_VAL_DRAWDOWN_RATIO_GATE` | `1.5` | val drawdown نباید بیشتر از ۱.۵× train drawdown باشد | gate سخت‌تر |
| `PHASE3_VAL_GATE_PENALTY` | `75.0` | جریمه بیشتر برای نقض gate | جریمه کمتر |
| `PHASE3_PER_RULE_MIN_VAL_TRADES_PER_SYMBOL` | `5` | هر rule باید حداقل ۵ trade per symbol در val داشته باشد | کمتر |

### پارامترهای Greedy Weights

```python
PHASE3_GREEDY_WEIGHTS = (1.0, 0.7, 0.5)  # sortino, drawdown, win_rate
```

در مرحله greedy، این وزن‌ها برای ترکیب سه objective استفاده می‌شوند:
- sortino با وزن ۱.۰ (مهم‌ترین)
- drawdown با وزن ۰.۷
- win_rate با وزن ۰.۵

### پارامترهای Consistency

| پارامتر | پیش‌فرض | تأثیر |
|---------|---------|-------|
| `PHASE3_SYMBOL_CONSISTENCY_WEIGHT` | `10.0` | وزن penalty برای inconsistency بین symbol ها |
| `PHASE3_TRAIN_VAL_CORR_WEIGHT` | `5.0` | وزن penalty برای correlation پایین بین train و val per-symbol PnL |

---

## ۱۰. نکات عملی

### اگر Phase 3 rule set خوبی پیدا نمی‌کند:
- `PHASE3_REFINE_GENERATIONS` را افزایش دهید (مثلاً 150)
- `PHASE3_REFINE_POP_SIZE` را افزایش دهید (مثلاً 200)
- `PHASE3_MIN_SYMBOL_COVERAGE` را کاهش دهید (مثلاً 5)

### اگر rule set در test خوب عمل نمی‌کند (overfitting):
- `PHASE3_VAL_SORTINO_RATIO_GATE` را افزایش دهید (مثلاً 0.7)
- `PHASE3_VAL_DRAWDOWN_RATIO_GATE` را کاهش دهید (مثلاً 1.2)
- `PHASE3_VAL_GATE_PENALTY` را افزایش دهید (مثلاً 150.0)

### اگر Phase 3 خیلی کند است:
- `PHASE3_BATCH_WORKERS` را افزایش دهید
- `PHASE3_REFINE_GENERATIONS` را کاهش دهید
- `PHASE3_REFINE_POP_SIZE` را کاهش دهید
