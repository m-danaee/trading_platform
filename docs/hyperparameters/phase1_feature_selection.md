# Phase 1 — Feature Selection

> **ماژول:** `gpu_fuzzy_trader/features/selector.py` → `Feature_Selector`  
> **ورودی:** `train_75.parquet`  
> **خروجی:** `outputs/selected_features_long.json`, `outputs/selected_features_short.json`

---

## ۱. هدف Phase 1

انتخاب feature های مرتبط، پایدار، و غیرتکراری برای هر direction (long/short) به صورت مستقل. خروجی این phase مستقیماً search space Phase 2 را تعریف می‌کند — هر feature انتخاب‌شده یک gene در chromosome می‌شود.

---

## ۲. الگوریتم کامل (۱۰ مرحله)

### مرحله ۱: شناسایی feature columns

```python
exclude = set(LABEL_COLUMNS) | set(META_COLUMNS) | set(INTERNAL_COLUMNS)
feature_cols = [c for c in train_df.columns if c not in exclude and not c.startswith("_")]
```

ستون‌های label، meta، و internal از feature list حذف می‌شوند.

---

### مرحله ۲: تشخیص mode هر feature

```python
detector = Feature_Detector()
feature_modes = detector.detect_all_modes(train_df, feature_cols)
```

هر feature به یکی از ۶ mode تقسیم می‌شود:

| Mode | شرط تشخیص | تعداد کلاس | مثال |
|------|-----------|-----------|------|
| `binary` | مقادیر ⊆ {0,1}، حداکثر ۲ مقدار منحصربه‌فرد | 2 | سیگنال‌های on/off |
| `ternary` | مقادیر ⊆ {-1,0,1}، حداکثر ۳ مقدار | 3 | جهت حرکت |
| `positive` | min ≥ 0، zero_ratio ≤ 0.3 | 5 | RSI، ATR |
| `sparse_positive` | min ≥ 0، zero_ratio > 0.3 | 5 | اندیکاتورهای نادر |
| `signed` | min < 0، zero_ratio ≤ 0.3 | 10 | MACD، momentum |
| `sparse_signed` | min < 0، zero_ratio > 0.3 | 5 | gap indicators |

**نکته مهم:** `zero_ratio` روی کل series محاسبه می‌شود (شامل صفرها)، نه فقط مقادیر non-NaN. این با `evaluator_v3.ipynb` هماهنگ است.

---

### مرحله ۳: حذف feature های با dispersion پایین

```python
top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
if top_freq > PHASE1_DISPERSION_THRESHOLD:  # پیش‌فرض: 0.95
    # حذف کن
```

اگر بیش از ۹۵٪ مقادیر یک feature یکسان باشند، آن feature اطلاعات مفیدی ندارد و حذف می‌شود.

**تأثیر `PHASE1_DISPERSION_THRESHOLD`:**
- **افزایش (مثلاً 0.99):** feature های near-constant بیشتری نگه داشته می‌شوند → noise بیشتر در Phase 2
- **کاهش (مثلاً 0.80):** feature های با تنوع کمتر هم حذف می‌شوند → feature list کوچک‌تر اما تمیزتر

---

### مرحله ۴: ساخت target direction-specific

این مرحله **قلب Phase 1** است. به جای یک target باینری ساده، یک target سه‌کلاسه ساخته می‌شود:

#### Long Target:
```python
hit_tp = max_288 >= open_next × (1 + TP/100)
hit_sl = min_288 <= open_next × (1 - SL/100)

clear_win = hit_tp & ~hit_sl          # فقط TP زده شد
clear_loss = hit_sl & ~hit_tp         # فقط SL زده شد
tp_first = both_hit & (max_before_min == 1)   # هر دو زده شد، TP اول
sl_first = both_hit & (max_before_min == 0)   # هر دو زده شد، SL اول

win = clear_win | tp_first   → کلاس 2
loss = clear_loss | sl_first → کلاس 0
neutral                      → کلاس 1
```

#### Short Target (mirror):
```python
hit_tp = min_288 <= open_next × (1 - TP/100)
hit_sl = max_288 >= open_next × (1 + SL/100)
tp_first = both_hit & (max_before_min == 0)  # برای short، min باید اول بیاید
```

**چرا این مهم است؟** با `PHASE1_ASYMMETRIC_TARGET = True`، long و short target های واقعاً متفاوتی دارند. این باعث می‌شود feature list های long و short از هم متمایز باشند — که برای یک استراتژی دو‌طرفه ضروری است.

**تأثیر `PHASE1_ASYMMETRIC_TARGET`:**
- **True (پیش‌فرض):** long و short feature list های متفاوت → استراتژی‌های تخصصی‌تر
- **False:** هر دو direction از یک target باینری ساده استفاده می‌کنند → feature list های مشابه‌تر

---

### مرحله ۵ و ۶: امتیازدهی per-symbol با Mutual Information

```python
for sym in symbols:
    sym_df = train_df[train_df["symbol"] == sym]
    scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
    per_symbol_scores[col].append(scores[i])
```

**چرا per-symbol؟** یک feature ممکن است برای symbol A بسیار مرتبط باشد اما برای symbol B بی‌فایده. امتیازدهی per-symbol این تفاوت را capture می‌کند.

**Mutual Information چیست؟** MI میزان اطلاعاتی را اندازه می‌گیرد که یک feature درباره target دارد. برخلاف correlation، MI روابط غیرخطی را هم capture می‌کند.

**discrete_mask:** برای feature های binary/ternary از MI گسسته استفاده می‌شود. برای بقیه از MI پیوسته (k-NN based).

---

### مرحله ۷: محاسبه امتیاز نهایی

```python
relevance = mean(per_symbol_scores)
stability = 1 - (std(per_symbol_scores) / mean(per_symbol_scores))
final_score = relevance × stability
```

**Stability Score:** اگر یک feature در همه symbol ها به یک اندازه مرتبط باشد، stability نزدیک به ۱ است. اگر در بعضی symbol ها خوب و در بعضی بد باشد، stability پایین می‌آید.

**چرا relevance × stability؟** یک feature که در یک symbol عالی است اما در بقیه بی‌فایده، برای یک استراتژی multi-symbol مناسب نیست. این فرمول feature های consistently مرتبط را ترجیح می‌دهد.

---

### مرحله ۷b: Stationarity Filter

این فیلتر بررسی می‌کند که آیا اهمیت یک feature در طول زمان (یا در regime های مختلف) ثابت می‌ماند.

#### دو روش stratification:

**Chronological (پیش‌فرض):**
```
داده train را به N fold زمانی تقسیم کن
برای هر fold، MI را محاسبه کن
بررسی کن که CV و rank drift در حد مجاز باشند
```

**Regime-based:**
```
از مدل GMM Phase 1 برای تقسیم به regime ها استفاده کن
برای هر regime، MI را محاسبه کن
```

#### دو معیار فیلتر:

**CV (Coefficient of Variation):**
```python
cv = std(fold_scores) / mean(fold_scores)
# باید <= PHASE1_STATIONARITY_CV_MAX باشد
```

**Rank Drift:**
```python
# رتبه feature در هر fold محاسبه می‌شود
# max_rank - min_rank باید <= PHASE1_STATIONARITY_RANK_DRIFT_MAX باشد
```

---

### مرحله ۸: حذف feature های redundant

```python
# برای feature های با mode یکسان، correlation پیرسون محاسبه می‌شود
# اگر corr > 0.95، feature با امتیاز پایین‌تر حذف می‌شود
```

**چرا within-mode؟** مقایسه correlation بین یک feature binary و یک feature signed معنایی ندارد.

---

### مرحله ۹: انتخاب Top K

```python
scored.sort(key=lambda x: x["score"], reverse=True)
candidate_k = PHASE1_TOP_K_FEATURES * 2  # ابتدا 2x انتخاب می‌شود
selected = scored[:candidate_k]
```

**چرا 2x؟** برای اینکه در مرحله overlap reduction (مرحله ۱۰) فضای کافی برای جایگزینی وجود داشته باشد.

---

### مرحله ۱۰: کاهش overlap بین long و short

```python
max_shared = int(top_k * PHASE1_MAX_FEATURE_OVERLAP)  # پیش‌فرض: 50% از 20 = 10

# feature های مشترک با کمترین تفاوت امتیاز بین long و short حذف می‌شوند
# از direction ای که امتیاز کمتری دارد حذف می‌شود
```

---

## ۳. Regime Clustering — `gpu_fuzzy_trader/features/regime_cluster.py`

### هدف

تقسیم داده train به regime های بازار (مثلاً: trending، ranging، volatile) برای stationarity filter.

### الگوریتم

```python
# ۱. Z-score کردن feature های regime به صورت per-symbol
X_scaled = z_score_per_symbol(df[PHASE1_REGIME_FEATURES])

# ۲. Fit کردن GMM یا KMeans
model = GaussianMixture(n_components=n_clusters, reg_covar=PHASE1_REGIME_GMM_REG_COVAR)
model.fit(X_scaled)

# ۳. Assign کردن label به هر ردیف
labels = model.predict(X_scaled)
```

### Feature های Regime

```python
PHASE1_REGIME_FEATURES = [
    "realized_vol_20",      # نوسان تاریخی ۲۰ کندل
    "parkinson_vol_20",     # نوسان Parkinson (high-low based)
    "atr_pct_14",           # Average True Range
    "vol_regime_pct_120",   # نوسان نسبی ۱۲۰ کندل
    "efficiency_ratio_20",  # نسبت حرکت خالص به کل مسیر
    "ret_autocorr_1_30",    # خودهمبستگی بازده
    "amihud_illiquidity_20",# معیار نقدشوندگی Amihud
    "vol_ratio_20_100",     # نسبت نوسان کوتاه‌مدت به بلندمدت
]
```

این feature ها ویژگی‌های ساختاری بازار را capture می‌کنند، نه سیگنال‌های معاملاتی.

---

## ۴. Hyperparameters Phase 1

### پارامترهای اصلی

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE1_TOP_K_FEATURES` | `20` | feature list بزرگ‌تر → chromosome بلندتر → search space بزرگ‌تر در Phase 2 | feature list کوچک‌تر → Phase 2 سریع‌تر اما ممکن است feature مهمی از دست برود |
| `PHASE1_DISPERSION_THRESHOLD` | `0.95` | feature های near-constant بیشتری نگه داشته می‌شوند | feature های با تنوع کمتر هم حذف می‌شوند |
| `PHASE1_MAX_FEATURE_OVERLAP` | `0.50` | long و short feature list های مشابه‌تر | long و short کاملاً متفاوت → تخصصی‌تر |
| `PHASE1_ASYMMETRIC_TARGET` | `True` | target های متفاوت برای long/short | target یکسان → feature list های مشابه |

### پارامترهای Stationarity

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE1_STATIONARITY_FOLDS` | `3` | بررسی دقیق‌تر stationarity، کندتر | بررسی کمتر، سریع‌تر |
| `PHASE1_STATIONARITY_CV_MAX` | `1.0` | feature های ناپایدارتر هم قبول می‌شوند | فقط feature های بسیار پایدار قبول می‌شوند |
| `PHASE1_STATIONARITY_RANK_DRIFT_MAX` | `10` | تغییر رتبه بیشتر مجاز است | فقط feature هایی که رتبه‌شان ثابت است قبول می‌شوند |
| `PHASE1_STATIONARITY_STRATIFY` | `"chronological"` | `"regime"` = تقسیم بر اساس regime بازار | - |

### پارامترهای Regime Clustering

| پارامتر | پیش‌فرض | تأثیر افزایش | تأثیر کاهش |
|---------|---------|-------------|------------|
| `PHASE1_REGIME_N_CLUSTERS` | `3` | تقسیم‌بندی دقیق‌تر بازار | کمتر از ۲ = بی‌معنی |
| `PHASE1_REGIME_MIN_SAMPLES` | `100` | fold های کوچک‌تر رد می‌شوند | fold های کوچک‌تر هم استفاده می‌شوند |
| `PHASE1_REGIME_CLUSTERER` | `"gmm"` | `"kmeans"` = سریع‌تر اما کمتر انعطاف‌پذیر | - |
| `PHASE1_REGIME_GMM_REG_COVAR` | `1e-6` | regularization بیشتر → GMM پایدارتر | ممکن است singular matrix error بدهد |

### پارامتر مهم: PHASE1_SAMPLING_TOTAL

| پارامتر | پیش‌فرض | تأثیر |
|---------|---------|-------|
| `PHASE1_SAMPLING_TOTAL` | `600_000` | **اصلی‌ترین knob برای GPU memory در Phase 2** |

این پارامتر تعداد کل ردیف‌هایی است که در Phase 2 برای backtest استفاده می‌شود. به صورت مساوی بین symbol ها تقسیم می‌شود:
- ۱۰ symbol → هر symbol حداکثر ۶۰,۰۰۰ ردیف
- اگر یک symbol کمتر از ۶۰,۰۰۰ ردیف داشته باشد، همه ردیف‌هایش استفاده می‌شود

**تأثیر روی GPU memory:** افزایش این مقدار، JAX array ها را به صورت خطی بزرگ‌تر می‌کند. روی GPU های با حافظه محدود، مقدار ≤ 150,000 توصیه می‌شود.

---

## ۵. خروجی Phase 1

### فرمت JSON

```json
{
  "direction": "long",
  "features": [
    {"name": "amihud_illiquidity_20", "mode": "sparse_positive", "score": 0.0423},
    {"name": "vol_ratio_20_100", "mode": "positive", "score": 0.0381},
    ...
  ]
}
```

### استفاده در Phase 2

هر feature در این لیست یک gene در chromosome می‌شود:
- `mode` تعیین می‌کند چند کلاس دارد (و در نتیجه search space هر gene چقدر است)
- `name` برای decode کردن chromosome به condition string استفاده می‌شود

---

## ۶. نکات عملی برای بهبود

### اگر long و short feature list های خیلی مشابه دارید:
- `PHASE1_MAX_FEATURE_OVERLAP` را کاهش دهید (مثلاً 0.30)
- `PHASE1_ASYMMETRIC_TARGET = True` را تأیید کنید

### اگر Phase 2 خیلی کند است:
- `PHASE1_TOP_K_FEATURES` را کاهش دهید (مثلاً 15)
- `PHASE1_SAMPLING_TOTAL` را کاهش دهید

### اگر feature های انتخاب‌شده در test خوب عمل نمی‌کنند:
- `PHASE1_STATIONARITY_CV_MAX` را کاهش دهید (مثلاً 0.5) تا فقط feature های پایدارتر انتخاب شوند
- `PHASE1_STATIONARITY_RANK_DRIFT_MAX` را کاهش دهید (مثلاً 5)
- `PHASE1_STATIONARITY_STRATIFY = "regime"` را امتحان کنید
