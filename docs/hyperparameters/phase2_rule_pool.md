# Phase 2 — Rule Pool Generation (NSGA-III Evolutionary Search)

> **ماژول:** `gpu_fuzzy_trader/phases/phase2_rule_pool.py` → `Rule_Pool_Generator`  
> **ورودی:** `train_75.parquet`, `validation_25.parquet`, feature lists از Phase 1  
> **خروجی:** `outputs/phase2_long_pool.json`, `outputs/phase2_short_pool.json`

---

## ۱. هدف Phase 2

پیدا کردن یک pool بزرگ و متنوع از قوانین fuzzy که هر کدام به تنهایی عملکرد خوبی دارند. این pool در Phase 3 برای ساخت یک "تیم" از قوانین استفاده می‌شود.

**تفاوت کلیدی با Phase 3:** در Phase 2 هر rule به تنهایی ارزیابی می‌شود. در Phase 3 ترکیب چند rule با هم ارزیابی می‌شود.

---

## ۲. Chromosome Encoding

### ساختار

```
chromosome = [gene_0, gene_1, ..., gene_{K-1}]
```

- `K` = تعداد feature های انتخاب‌شده در Phase 1 (مثلاً ۲۰)
- هر `gene_i` یک مقدار صحیح است

### مقادیر مجاز هر gene

```
gene_i ∈ {0, 1, ..., num_classes_i - 1, dont_care_i}
```

| Mode | num_classes | dont_care | مقادیر مجاز |
|------|------------|-----------|------------|
| binary | 2 | 2 | {0, 1, 2} |
| ternary | 3 | 3 | {0, 1, 2, 3} |
| positive/sparse_positive/sparse_signed | 5 | 5 | {0, 1, 2, 3, 4, 5} |
| signed | 10 | 10 | {0, 1, ..., 9, 10} |

**dont_care = num_classes:** وقتی gene برابر dont_care است، آن feature در rule غیرفعال است (شرطی برای آن feature وجود ندارد).

### مثال

فرض کنید ۳ feature داریم:
```
feature_0: mode=positive, dont_care=5
feature_1: mode=binary, dont_care=2
feature_2: mode=signed, dont_care=10

chromosome = [3, 2, 7]
→ feature_0: gene=3 → "High"
→ feature_1: gene=2 → dont_care (غیرفعال)
→ feature_2: gene=7 → "Bullish"

Rule: [feature_0] IS High AND [feature_2] IS Bullish
```

### تعداد شرایط فعال

```python
active_conditions = sum(chromosome[i] != dont_care[i] for i in range(K))
```

باید بین `MIN_CONDITIONS` و `MAX_CONDITIONS` باشد.

---

## ۳. Search Space

### اندازه search space

برای K=20 feature با mode های مختلف:
- هر gene می‌تواند ۳ تا ۱۱ مقدار داشته باشد
- کل search space: تقریباً `6^20 ≈ 3.6 × 10^15` حالت ممکن

این فضا بسیار بزرگ است و exhaustive search غیرممکن است. NSGA-III برای جستجوی هوشمند در این فضا استفاده می‌شود.

---

## ۴. الگوریتم NSGA-III

### چرا NSGA-III؟

NSGA-III (Non-dominated Sorting Genetic Algorithm III) برای بهینه‌سازی چندهدفه طراحی شده است. برخلاف NSGA-II که از crowding distance استفاده می‌کند، NSGA-III از **reference vectors** برای حفظ تنوع در فضای objective استفاده می‌کند.

### سه Objective (همه minimize می‌شوند)

```python
f1 = -sortino_ratio      # می‌خواهیم sortino را maximize کنیم
f2 = max_drawdown_pct    # می‌خواهیم drawdown را minimize کنیم
f3 = -win_rate           # می‌خواهیم win_rate را maximize کنیم
```

**چرا سه objective؟** یک rule ممکن است sortino بالا اما drawdown بالا هم داشته باشد. با سه objective، می‌توانیم rule هایی با trade-off های مختلف پیدا کنیم.

### جریان یک نسل

```
1. ارزیابی population با backtest engine
2. Non-dominated sorting → Pareto fronts
3. Tournament selection از Pareto front
4. Crossover (uniform) → offspring
5. Mutation → offspring
6. Merge parents + offspring
7. NSGA-III selection → نسل بعدی
```

---

## ۵. Fitness Evaluation

### تابع ارزیابی

```python
def _evaluate_chromosome(chromosome, dont_cares, engine, pareto_front, val_engine=None):
    # ۱. شمارش شرایط فعال
    active = count_active_conditions(chromosome, dont_cares)
    
    # ۲. جریمه تعداد شرایط
    if active < MIN_CONDITIONS:
        cond_penalty = (MIN_CONDITIONS - active) × 10.0
    elif active > MAX_CONDITIONS:
        cond_penalty = (active - MAX_CONDITIONS) × 10.0
    
    # ۳. اجرای backtest روی train
    metrics = engine.simulate_rule_batch(chromosomes, tp=PHASE2_TP, sl=PHASE2_SL, capital_pct=PHASE2_CAPITAL_PCT)
    
    # ۴. اگر PHASE2_JOINT_TRAIN_VAL فعال است، روی val هم اجرا می‌شود
    if PHASE2_JOINT_TRAIN_VAL and val_engine:
        val_metrics = val_engine.simulate_rule_batch(...)
        sortino_for_obj = min(sortino_train, sortino_val)  # بدترین حالت
    
    # ۵. جریمه support
    support_penalty = compute_support_penalty(metrics, regime_fractions)
    
    # ۶. جریمه diversity
    if min_hamming_to_pareto <= PHASE2_DIVERSITY_HAMMING_THRESHOLD:
        diversity_penalty = PHASE2_DIVERSITY_PENALTY
    
    # ۷. محاسبه objectives
    f1 = -sortino_for_obj + support_penalty + diversity_penalty + cond_penalty
    f2 = max_dd + support_penalty + diversity_penalty + cond_penalty
    f3 = -win_rate + support_penalty + diversity_penalty + cond_penalty
```

### Saturating Sortino

```python
def _saturating_sortino(raw):
    return tanh(raw / SORTINO_SCALE) × SORTINO_CAP
```

**چرا tanh؟** بدون saturation، یک rule با sortino=100 و یک rule با sortino=5 در نسل اول فاصله زیادی دارند و NSGA-III نمی‌تواند به خوبی بین آن‌ها تمایز قائل شود. tanh این مقادیر را به بازه [-SORTINO_CAP, +SORTINO_CAP] فشرده می‌کند.

---

## ۶. Penalties

### Support Penalty

```python
if executed_trades < MIN_TRADE_SUPPORT:
    penalty = SUPPORT_PENALTY_MAX × (1 - executed_trades / MIN_TRADE_SUPPORT)
```

**چرا؟** یک rule که فقط ۵ trade داشته، sortino بالایی ممکن است داشته باشد اما این sortino آماری معنادار نیست. Support penalty این rule ها را جریمه می‌کند.

**MIN_TRADE_POOL_FLOOR:** حداقل مطلق trade برای ورود به pool. rule هایی با کمتر از این تعداد trade اصلاً وارد pool نمی‌شوند.

### Diversity Penalty (Hamming Distance)

```python
min_hamming = min(hamming_distance(chromosome, pf) for pf in pareto_front)
if min_hamming <= PHASE2_DIVERSITY_HAMMING_THRESHOLD:
    diversity_penalty = PHASE2_DIVERSITY_PENALTY
```

**Hamming Distance:** تعداد gene هایی که بین دو chromosome متفاوت هستند.

**چرا؟** بدون این penalty، NSGA-III ممکن است به یک ناحیه کوچک از search space همگرا شود و rule های مشابه زیادی تولید کند. این penalty تنوع را تشویق می‌کند.

### Condition Count Penalty

```python
if active < MIN_CONDITIONS:
    penalty = (MIN_CONDITIONS - active) × 10.0
elif active > MAX_CONDITIONS:
    penalty = (active - MAX_CONDITIONS) × 10.0
```

rule هایی با خیلی کم یا خیلی زیاد شرط جریمه می‌شوند.

---

## ۷. Regime Support

### هدف

بررسی اینکه آیا یک rule فقط در یک regime خاص بازار کار می‌کند یا در همه regime ها.

```python
if PHASE2_REGIME_SUPPORT_ENABLED:
    # بارگذاری مدل GMM از Phase 1
    bundle = load_regime_model(PHASE2_REGIME_MODEL_PATH)
    regime_ids = assign_regime_labels(sampled_df, bundle)
    
    # محاسبه نسبت trade ها در هر regime
    regime_trade_counts = count_trades_per_regime(metrics, regime_ids)
    
    # بررسی concentration
    if max(regime_trade_counts) / total_trades > PHASE2_REGIME_CONCENTRATION_MIN:
        # این rule یک "specialist" است - فقط در یک regime کار می‌کند
        is_specialist = True
```

**Regime Specialist:** یک rule که ۹۰٪+ trade هایش در یک regime اتفاق می‌افتد. این rule ها ممکن است در test خوب عمل نکنند اگر regime بازار تغییر کند.

---

## ۸. Population Initialization

```python
def _init_population(pop_size, feature_infos, rng, dont_care_prob=0.5, seeded_chromosomes=None):
    # ۱. seed کردن از archive قبلی
    seed_count = min(pop_size, int(pop_size × PHASE2_ARCHIVE_SEED_FRACTION), len(seed_rows))
    
    # ۲. بقیه population به صورت random
    for k, fi in enumerate(feature_infos):
        if rng.random() < dont_care_prob:
            population[i, k] = dont_care  # غیرفعال
        else:
            population[i, k] = random_class_index
```

**dont_care_prob = 0.5:** به طور میانگین نیمی از gene ها غیرفعال هستند. این باعث می‌شود rule های اولیه ۳-۴ شرط فعال داشته باشند (که با MIN/MAX_CONDITIONS هماهنگ است).

---

## ۹. Crossover و Mutation

### Crossover (Uniform)

```python
def _crossover(parent_a, parent_b, rng):
    mask = rng.random(K) < 0.5
    child_a = where(mask, parent_a, parent_b)
    child_b = where(mask, parent_b, parent_a)
```

هر gene با احتمال ۵۰٪ از parent_a یا parent_b می‌آید.

### Mutation

```python
def _mutate(chromosome, feature_infos, dont_cares, rng, mutation_rate=0.1):
    for k in range(K):
        if rng.random() < mutation_rate:
            if chromosome[k] == dont_care:
                chromosome[k] = random_class_index  # فعال کردن
            else:
                if rng.random() < 0.3:
                    chromosome[k] = dont_care  # غیرفعال کردن
                else:
                    chromosome[k] = different_class_index  # تغییر کلاس
```

**mutation_rate = 0.1:** هر gene با احتمال ۱۰٪ mutate می‌شود.

---

## ۱۰. Archive System

### دو نوع archive

**Per-run pool** (`outputs/phase2_long_pool.json`):
- در هر اجرا بازنویسی می‌شود
- بهترین rule های Pareto front آخرین اجرا

**Cross-run archive** (`phase2_rule_archive/phase2_long_archive.json`):
- بین اجراها حفظ می‌شود
- بهترین rule های تمام اجراهای قبلی
- در اجرای بعدی، `PHASE2_ARCHIVE_SEED_FRACTION` از population از این archive seed می‌شود

### Warm-start

```python
# در شروع هر اجرا:
seed_fraction = PHASE2_ARCHIVE_SEED_FRACTION  # پیش‌فرض: 0.35
# 35% از population از archive قبلی seed می‌شود
# 65% به صورت random initialize می‌شود
```

**مزیت:** اجراهای بعدی از نقطه بهتری شروع می‌کنند و سریع‌تر به راه‌حل‌های خوب می‌رسند.

---

## ۱۱. Hyperparameters Phase 2

### پارامترهای اصلی

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE2_POPULATION_SIZE` | `200` | جستجوی گسترده‌تر، کندتر | سریع‌تر اما ممکن است به local optimum برسد |
| `PHASE2_GENERATIONS` | `200` | بهینه‌سازی بیشتر، کندتر | سریع‌تر اما ممکن است converge نشود |
| `PHASE2_ARCHIVE_MAX_SIZE` | `500` | pool بزرگ‌تر برای Phase 3 | pool کوچک‌تر، Phase 3 سریع‌تر |
| `PHASE2_ARCHIVE_SEED_FRACTION` | `0.35` | warm-start بیشتر از archive | exploration بیشتر |

### پارامترهای Risk (ثابت در Phase 2)

| پارامتر | پیش‌فرض | توضیح |
|---------|---------|-------|
| `PHASE2_TP` | `3.0` | Take Profit ثابت برای همه rule ها در Phase 2 |
| `PHASE2_SL` | `1.5` | Stop Loss ثابت |
| `PHASE2_CAPITAL_PCT` | `48.0` | درصد سرمایه برای هر trade |

**چرا ثابت؟** در Phase 2 می‌خواهیم فقط "alpha" (توانایی پیش‌بینی جهت) را بهینه کنیم، نه risk parameters. Phase 4 این پارامترها را بهینه می‌کند.

### پارامترهای Support

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `MIN_TRADE_SUPPORT` | `300` | rule های با trade کمتر بیشتر جریمه می‌شوند | rule های با trade کم هم قبول می‌شوند |
| `SUPPORT_PENALTY_MAX` | `50.0` | جریمه بیشتر برای rule های کم‌trade | جریمه کمتر |
| `MIN_TRADE_POOL_FLOOR` | `75` | حداقل مطلق trade برای ورود به pool |  |

### پارامترهای Diversity

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE2_DIVERSITY_HAMMING_THRESHOLD` | `2` | rule های مشابه‌تر هم جریمه می‌شوند | فقط rule های خیلی مشابه جریمه می‌شوند |
| `PHASE2_DIVERSITY_PENALTY` | `5.0` | تنوع بیشتر تشویق می‌شود | تنوع کمتر اهمیت دارد |

### پارامترهای Sortino Transform

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `SORTINO_CAP` | `5.0` | حداکثر مقدار Sortino در fitness | - |
| `SORTINO_SCALE` | `3.0` | نقطه inflection تابع tanh | مقادیر بزرگ‌تر Sortino بیشتر فشرده می‌شوند |

### پارامترهای Regime

| پارامتر | پیش‌فرض | تأثیر |
|---------|---------|-------|
| `PHASE2_REGIME_SUPPORT_ENABLED` | `True` | فعال/غیرفعال کردن بررسی regime |
| `PHASE2_REGIME_CONCENTRATION_MIN` | `0.90` | آستانه تشخیص specialist |
| `PHASE2_REGIME_MIN_WIN_RATE` | `0.40` | حداقل win rate برای regime specialist |
| `PHASE2_JOINT_TRAIN_VAL` | `True` | ارزیابی روی train و val هر دو |

---

## ۱۲. نکات عملی

### اگر pool خالی است:
- `MIN_TRADE_SUPPORT` را کاهش دهید (مثلاً 150)
- `MIN_TRADE_POOL_FLOOR` را کاهش دهید (مثلاً 30)
- `PHASE2_GENERATIONS` را افزایش دهید

### اگر pool rule های مشابه زیادی دارد:
- `PHASE2_DIVERSITY_HAMMING_THRESHOLD` را افزایش دهید (مثلاً 3)
- `PHASE2_DIVERSITY_PENALTY` را افزایش دهید (مثلاً 10.0)

### اگر Phase 2 خیلی کند است:
- `PHASE2_POPULATION_SIZE` را کاهش دهید (مثلاً 100)
- `PHASE2_GENERATIONS` را کاهش دهید (مثلاً 100)
- `PHASE1_SAMPLING_TOTAL` را کاهش دهید

### اگر rule ها در validation خوب عمل نمی‌کنند:
- `PHASE2_JOINT_TRAIN_VAL = True` را تأیید کنید
- `MIN_TRADE_SUPPORT` را افزایش دهید تا rule های با آمار کافی انتخاب شوند
