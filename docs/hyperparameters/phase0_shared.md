# Phase 0 — Shared Infrastructure: Data Loading, Splitting, Backtest Engine

> **مخاطب:** Data Scientist  
> **هدف:** درک کامل زیرساخت مشترک پروژه — از لود داده تا موتور بک‌تست

---

## ۱. معماری کلی Pipeline

```
data/train.csv
      │
      ▼
 Data_Loader          ← 7 مرحله پاک‌سازی
      │
      ▼
 Data_Splitter         ← تقسیم per-symbol chronological 75/25
      │
      ├──► train_75.parquet      (ورودی Phase 1, 2, 3, 4)
      └──► validation_25.parquet (ورودی Phase 3, 4)

data/test.csv
      │
      ▼
 Data_Loader (Phase 5 only)
      │
      ▼
 OOS_Evaluator
```

---

## ۲. Data_Loader — `gpu_fuzzy_trader/data/loader.py`

### چه کاری می‌کند؟

یک CSV می‌خواند و ۷ مرحله پاک‌سازی روی آن اعمال می‌کند. **Stateless** است — هیچ چیزی را cache نمی‌کند.

### مراحل پردازش (به ترتیب اجرا)

#### مرحله ۱: خواندن CSV
```python
df = pd.read_csv(path, sep=",")
```
فایل با جداکننده کاما خوانده می‌شود.

#### مرحله ۲: Parse کردن datetime
```python
df["datetime"] = pd.to_datetime(df["datetime"])
```
ستون datetime به نوع datetime64 تبدیل می‌شود تا مرتب‌سازی زمانی صحیح باشد.

#### مرحله ۳: مرتب‌سازی بر اساس (symbol, datetime)
```python
df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
```
**چرا مهم است؟** تمام منطق بعدی (split، bar_index، release_index) به ترتیب زمانی درون هر symbol وابسته است. اگر این مرتب‌سازی نباشد، label leakage رخ می‌دهد.

#### مرحله ۴: حذف آخرین ۲۸۸ ردیف هر symbol
```python
tail_count = df.groupby("symbol").cumcount(ascending=False)
df = df[tail_count >= TAIL_DROP_ROWS].reset_index(drop=True)
```
**چرا ۲۸۸؟** label های `label_close_288`, `label_min_288`, `label_max_288` نیاز به ۲۸۸ کندل آینده دارند. آخرین ۲۸۸ ردیف هر symbol این label ها را ندارند (NaN هستند). حذف آن‌ها از data leakage جلوگیری می‌کند.

**محاسبه:** 288 کندل × 5 دقیقه = 1440 دقیقه = 24 ساعت. یعنی horizon پیش‌بینی یک روز کامل است.

#### مرحله ۵: حذف ردیف‌هایی که label آن‌ها NaN است
```python
label_cols_present = [c for c in LABEL_COLUMNS if c in df.columns]
df = df.dropna(subset=label_cols_present).reset_index(drop=True)
```
اگر بعد از مرحله ۴ هنوز ردیف‌هایی با label NaN وجود داشت، حذف می‌شوند.

#### مرحله ۶: پر کردن NaN در feature columns با صفر
```python
df[existing_feature_cols] = df[existing_feature_cols].fillna(0)
```
**چرا صفر؟** feature ها discretized هستند (اعداد صحیح). صفر در اکثر mode ها معنای "Inactive" یا "Very Low" دارد — یک مقدار neutral که سیگنال اشتباه نمی‌دهد.

#### مرحله ۷: محاسبه `_symbol_bar_index`
```python
df["_symbol_bar_index"] = df.groupby("symbol").cumcount()
```
برای هر symbol، شماره ترتیبی کندل از صفر شروع می‌شود. این ستون در `precompute_release_indices` استفاده می‌شود تا بدانیم هر trade دقیقاً چه زمانی باید بسته شود.

### خروجی نهایی
DataFrame با:
- تمام ستون‌های اصلی
- ستون `_symbol_bar_index` اضافه شده
- نوع داده‌های downcast شده (float32 به جای float64) برای صرفه‌جویی در RAM

---

## ۳. Data_Splitter — `gpu_fuzzy_trader/data/splitter.py`

### الگوریتم تقسیم

```
برای هر symbol به صورت مستقل:
    N = تعداد ردیف‌های آن symbol
    split_point = floor(N × 0.75)
    train = ردیف‌های 0 تا split_point-1
    validation = ردیف‌های split_point تا N-1

train_df = concat تمام train های همه symbol ها
val_df = concat تمام validation های همه symbol ها
```

### چرا per-symbol؟

اگر split روی کل dataset انجام شود، یک symbol با داده‌های بیشتر ممکن است بیشتر در validation باشد. Per-symbol split تضمین می‌کند هر symbol دقیقاً ۷۵٪ train و ۲۵٪ validation دارد.

### مثال عددی

فرض کنید symbol A دارای ۱۰۰۰ ردیف است:
- `split_point = floor(1000 × 0.75) = 750`
- train: ردیف‌های ۰ تا ۷۴۹ (۷۵۰ ردیف)
- validation: ردیف‌های ۷۵۰ تا ۹۹۹ (۲۵۰ ردیف)

### Persistence

نتیجه به فرمت Parquet ذخیره می‌شود:
- `data/train_75.parquet`
- `data/validation_25.parquet`

**Cache logic:** اگر این فایل‌ها جدیدتر از `train.csv` باشند، در اجرای بعدی مستقیماً لود می‌شوند (بدون re-split). این باعث صرفه‌جویی قابل توجه در زمان می‌شود.

---

## ۴. Backtest Engine — `gpu_fuzzy_trader/backtest/cpu_engine.py`

### اهمیت این ماژول

**CPUBacktestEngine دقیقاً همان منطق `evaluator_v3.ipynb` را پیاده‌سازی می‌کند.** این alignment حیاتی است — اگر engine بهینه‌سازی با engine ارزیابی نهایی فرق داشته باشد، استراتژی‌هایی که در Phase 2 و 3 خوب به نظر می‌رسند در Phase 5 شکست می‌خورند.

### منطق Rule Matching

```python
# برای هر کندل، اولین rule که همه شرایطش برقرار است انتخاب می‌شود
for rule_idx, rule_entry in enumerate(rule_set, start=1):
    rule_signals = _build_rule_signal_mask(df, conditions)
    new_match_mask = rule_signals & (~assigned_mask)  # فقط کندل‌های هنوز assign نشده
    assigned_mask[matched_indices] = True
```

**Priority-based:** اگر کندلی با rule 1 match کند، rule 2 و 3 برای آن کندل بررسی نمی‌شوند. این از duplicate entry جلوگیری می‌کند.

### منطق Trade Outcome

#### Long Direction:
```
TP hit: label_max_288 >= entry × (1 + tp/100)
SL hit: label_min_288 <= entry × (1 - sl/100)
هر دو hit: اگر label_max_before_min == 1 → TP اول رسیده → سود
           اگر label_max_before_min == 0 → SL اول رسیده → ضرر
هیچکدام: خروج با label_close_288
```

#### Short Direction:
```
TP hit: label_min_288 <= entry × (1 - tp/100)
SL hit: label_max_288 >= entry × (1 + sl/100)
هر دو hit: اگر label_max_before_min == 1 → SL اول رسیده → ضرر
           اگر label_max_before_min == 0 → TP اول رسیده → سود
هیچکدام: خروج با -close_ret (برای short، بازگشت معکوس است)
```

### مدیریت سرمایه

```python
position_notional = min(
    equity × (capital_pct / 100) × leverage,
    max(0, equity × MAX_TOTAL_EXPOSURE_PCT/100 × leverage - open_total_exposure)
)
```

**نکته مهم:** هر trade تا زمان `entry_bar + MAX_HOLD_CANDLES` exposure را رزرو می‌کند. این از استفاده از سود آینده برای sizing trade های جدید جلوگیری می‌کند (no look-ahead bias در capital management).

### محاسبه Sortino Ratio

```python
def _sortino_ratio_from_returns(trade_returns, target_return=0.0):
    excess_returns = returns - target_return
    mean_excess_return = mean(excess_returns)
    downside_returns = minimum(excess_returns, 0.0)
    downside_deviation = sqrt(mean(square(downside_returns)))
    return min(mean_excess_return / downside_deviation, SORTINO_CAP)
```

**Sortino vs Sharpe:** Sortino فقط نوسانات منفی را جریمه می‌کند. برای trading مناسب‌تر است چون سود زیاد نباید به عنوان ریسک محسوب شود.

---

## ۵. Hyperparameters مشترک (Phase 0)

### مسیرها

| پارامتر | مقدار پیش‌فرض | توضیح |
|---------|--------------|-------|
| `TRAIN_CSV_PATH` | `"data/train.csv"` | داده آموزش |
| `TEST_CSV_PATH` | `"data/test.csv"` | داده تست — **فقط در Phase 5** |
| `TRAIN_75_PATH` | `"data/train_75.parquet"` | cache split آموزش |
| `VALIDATION_25_PATH` | `"data/validation_25.parquet"` | cache split validation |
| `OUTPUTS_DIR` | `"outputs"` | پوشه خروجی‌ها |

### Schema

| پارامتر | مقدار | توضیح |
|---------|-------|-------|
| `LABEL_COLUMNS` | 5 ستون | هرگز وارد feature matrix نمی‌شوند |
| `META_COLUMNS` | `["datetime", "symbol"]` | metadata — از feature selection حذف می‌شوند |
| `TAIL_DROP_ROWS` | `288` | آخرین N ردیف هر symbol حذف می‌شود |

### پارامترهای Backtest

| پارامتر | مقدار پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|--------------|-------------|------------|
| `INITIAL_CAPITAL` | `1000.0` | نتایج مطلق بزرگ‌تر، نسبی یکسان | نتایج مطلق کوچک‌تر |
| `LEVERAGE` | `1.0` | سود و ضرر بزرگ‌تر، ریسک بیشتر | محافظه‌کارانه‌تر |
| `FEE_PCT` | `0.20` | استراتژی‌های high-turnover بیشتر جریمه می‌شوند | trade های بیشتر سودآور می‌شوند |
| `MAX_HOLD_CANDLES` | `288` | horizon بلندتر، label های متفاوت نیاز است | horizon کوتاه‌تر |
| `MAX_TOTAL_EXPOSURE_PCT` | `100.0` | می‌توان بیشتر از سرمایه expose کرد | محدودیت بیشتر روی position sizing |
| `MIN_POSITION_NOTIONAL` | `1.0` | trade های کوچک‌تر skip می‌شوند | trade های بیشتر اجرا می‌شوند |

### نکات مهم برای Data Scientist

**FEE_PCT = 0.20:** این round-trip fee است (ورود + خروج). برای استراتژی‌هایی که trade های زیاد دارند، این fee به شدت سودآوری را کاهش می‌دهد. اگر می‌خواهید استراتژی‌های high-frequency را تست کنید، این مقدار را واقع‌بینانه تنظیم کنید.

**MAX_HOLD_CANDLES = 288:** این با label horizon هماهنگ است. اگر این مقدار را تغییر دهید، باید label های جدید هم تولید کنید.

**SORTINO_CAP = 5.0:** حداکثر مقدار Sortino که در fitness function استفاده می‌شود. این از pinning شدن بهترین راه‌حل‌ها در مقدار ثابت جلوگیری می‌کند.

---

## ۶. Logging

| پارامتر | مقدار | توضیح |
|---------|-------|-------|
| `LOG_GENERATION_INTERVAL` | `0` | 0 = auto-throttle؛ N > 0 = هر N نسل log بزن |

با `0`، سیستم به صورت هوشمند log می‌کند تا console پر از پیام نشود.
