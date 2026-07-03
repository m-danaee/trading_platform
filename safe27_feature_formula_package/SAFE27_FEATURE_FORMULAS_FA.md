# فرمول‌های ۲۷ ویژگی تأییدشده پروژه تریدینگ

این سند فرمول‌های ۲۷ ویژگی‌ای را ثبت می‌کند که روی هر دو دیتاست **Train** و **Test** از نظر مقدار عددی در تلورانس `1e-6` و از نظر کلاس فازی، تطابق کامل داشته‌اند.

چهار ویژگی زیر عمداً در این بسته نیستند، چون با وجود تطابق عددی بسیار بالا، روی تعداد کمی از نقاط مرزی کلاس فازی اختلاف داشتند:

- `body_signed_to_tr`
- `body_to_tr`
- `lower_wick_to_tr`
- `upper_wick_to_tr`

## فایل‌هایی که باید برای بازتولید دقیق ارسال شوند

فرمول خام به‌تنهایی کافی نیست، چون خروجی نهایی هر ویژگی با **Lag و نرمال‌سازی جداگانه برای هر Symbol** ساخته می‌شود. برای بازتولید دقیق، این سه فایل را با هم ارسال کنید:

1. `SAFE27_FEATURE_FORMULAS_FA.md` — توضیح انسانی فرمول‌ها.
2. `safe27_specs.json` — Lag، نوع ویژگی، پیش‌تبدیل و همه پارامترهای دقیق نرمال‌سازی برای Symbolهای ۱ تا ۱۰.
3. `safe27_builder.py` — پیاده‌سازی اجرایی فرمول‌ها.

## داده خام لازم

ورودی باید کندل پنج‌دقیقه‌ای پیوسته و مرتب‌شده برای هر Symbol، با ستون‌های زیر باشد:

```text
datetime, symbol, open, high, low, close, volume
```

تمام Rollingها، EMAها و Lagها باید **به‌صورت مستقل داخل هر Symbol** محاسبه شوند.

## قراردادهای مشترک

- `O_t,H_t,L_t,C_t,V_t`: Open، High، Low، Close و Volume در کندل `t`.
- `EMA(X,n)`: `pandas.Series.ewm(span=n, adjust=False, min_periods=n).mean()`.
- `WilderEMA(X,n)`: `ewm(alpha=1/n, adjust=False, min_periods=n).mean()`.
- تقسیم بر صفر به `NaN` تبدیل می‌شود؛ مقدار مصنوعی یا `bfill` استفاده نمی‌شود.
- همه Featureها به‌جز `mom_stoch_rsi_14_14_3` با **Lag=1** استفاده می‌شوند؛ یعنی مقدار نهایی ردیف `t` از Candidate ردیف `t-1` می‌آید.
- `mom_stoch_rsi_14_14_3` دارای **Lag=2** است.

## تبدیل نهایی و نرمال‌سازی

پس از ساخت Candidate، ابتدا Lag و سپس `pre_transform` اعمال می‌شود. پارامترهای دقیق هر Symbol در `safe27_specs.json` قرار دارند.

### روش `fit_interior`

برای Symbol شماره `s`:

```text
z_t = (x_t - low_s) / width_s
```

اگر خانواده ویژگی `positive` باشد:

```text
feature_t = clip(z_t, 0, 1)
```

اگر خانواده ویژگی `signed` باشد:

```text
feature_t = clip(2*z_t - 1, -1, 1)
```

### روش Quantile

```text
z_t = (x_t - low_s) / (high_s - low_s)
```

سپس بر اساس خانواده، همان تبدیل Positive یا Signed بالا اعمال می‌شود.

### روش خطی `linear_fit_interior_clip`

```text
feature_t = clip(a_s * x_t + b_s, 0, 1)
```

---
## 1. `atr_pct_14`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `atr_pct_14_ema`
- **فرمول خام:**

ابتدا True Range محاسبه می‌شود:
`TR_t = max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)`
سپس `ATR14_t = EMA(TR, span=14, adjust=False, min_periods=14)` و مقدار خام:
`raw_t = ATR14_t / C_t`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `atr_pct_14` داخل فایل `safe27_specs.json`.

## 2. `band_bb_percB_20_2`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `percent_b_20_samp_signed`
- **فرمول خام:**

`SMA20_t = mean(C,20)` و `STD20_t = std(C,20, ddof=1)`
`Upper_t = SMA20_t + 2×STD20_t`
`Lower_t = SMA20_t - 2×STD20_t`
`raw_t = 2×(C_t-Lower_t)/(Upper_t-Lower_t)-1`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `band_bb_percB_20_2` داخل فایل `safe27_specs.json`.

## 3. `channel_pos_20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `channel_pos_20_hl`
- **فرمول خام:**

`HH20_t = rolling_max(H,20)` و `LL20_t = rolling_min(L,20)`
`raw_t = (C_t-LL20_t)/(HH20_t-LL20_t)`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `channel_pos_20` داخل فایل `safe27_specs.json`.

## 4. `close_location_value`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `close_location_signed_hl`
- **فرمول خام:**

`raw_t = 2×(C_t-L_t)/(H_t-L_t)-1`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `close_location_value` داخل فایل `safe27_specs.json`.

## 5. `dmi_balance_14`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `dmi_balance_wilder_14`
- **فرمول خام:**

`up_t = H_t-H_{t-1}` و `down_t = L_{t-1}-L_t`
`+DM_t = up_t` اگر `up_t>down_t` و `up_t>0`، وگرنه صفر.
`-DM_t = down_t` اگر `down_t>up_t` و `down_t>0`، وگرنه صفر.
با هموارسازی Wilder (`EWM alpha=1/14`):
`+DI_t = 100×Wilder(+DM,14)/Wilder(TR,14)`
`-DI_t = 100×Wilder(-DM,14)/Wilder(TR,14)`
`raw_t = (+DI_t-(-DI_t))/(+DI_t+(-DI_t))`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `dmi_balance_14` داخل فایل `safe27_specs.json`.

## 6. `dollar_vol_rel_20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `dollar_vol_rel_20_ema`
- **فرمول خام:**

`DollarVolume_t = C_t×V_t`
`ratio_t = DollarVolume_t / EMA(DollarVolume,20)_t`
مقدار پیش‌تبدیل‌شده: `x_t = -|ratio_t|`

- **پیش‌تبدیل:** `x=-|raw|`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `dollar_vol_rel_20` داخل فایل `safe27_specs.json`.

## 7. `downside_semivol_20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `downside_semivol_20_logret`
- **فرمول خام:**

`r_t = ln(C_t)-ln(C_{t-1})`
`raw_t = sqrt(mean(min(r_t,0)^2, window=20))`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `downside_semivol_20` داخل فایل `safe27_specs.json`.

## 8. `efficiency_ratio_20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `efficiency_ratio_20`
- **فرمول خام:**

`raw_t = |C_t-C_{t-20}| / Σ_{i=t-19}^{t}|C_i-C_{i-1}|`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `efficiency_ratio_20` داخل فایل `safe27_specs.json`.

## 9. `ema_gap_atr_20`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `ema_gap_atr20_ema`
- **فرمول خام:**

`EMA20_t = EMA(C,20)` و `ATR14_t = EMA(TR,14)`
`raw_t = (C_t-EMA20_t)/ATR14_t`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `ema_gap_atr_20` داخل فایل `safe27_specs.json`.

## 10. `ema_slope_atr_20_5`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `ema_slope_atr20_5_ema`
- **فرمول خام:**

`EMA20_t = EMA(C,20)` و `ATR14_t = EMA(TR,14)`
`raw_t = (EMA20_t-EMA20_{t-5})/ATR14_t`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `ema_slope_atr_20_5` داخل فایل `safe27_specs.json`.

## 11. `macd_hist_atr`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `macd_hist_atr_ema`
- **فرمول خام:**

`MACD_t = EMA(C,12)_t-EMA(C,26)_t`
`Signal_t = EMA(MACD,9)_t`
`Hist_t = MACD_t-Signal_t`
`raw_t = Hist_t/ATR14_t`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `macd_hist_atr` داخل فایل `safe27_specs.json`.

## 12. `parkinson_vol_20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `parkinson_vol_20`
- **فرمول خام:**

`raw_t = sqrt(mean(ln(H_t/L_t)^2, window=20)/(4×ln(2)))`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `parkinson_vol_20` داخل فایل `safe27_specs.json`.

## 13. `percent_b_20`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `percent_b_20_samp_signed`
- **فرمول خام:**

همان فرمول خام `band_bb_percB_20_2`:
`raw_t = 2×(C_t-Lower_t)/(Upper_t-Lower_t)-1`
اما پارامترهای نرمال‌سازی آن به‌صورت مستقل در فایل Spec نگهداری می‌شوند.

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `percent_b_20` داخل فایل `safe27_specs.json`.

## 14. `range_compression_20_100`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `range_compression_20_100`
- **فرمول خام:**

`Range20_t = rolling_max(H,20)-rolling_min(L,20)`
`Range100_t = rolling_max(H,100)-rolling_min(L,100)`
`raw_t = Range20_t/Range100_t`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `range_compression_20_100` داخل فایل `safe27_specs.json`.

## 15. `realized_vol_20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `realized_vol_20_logret`
- **فرمول خام:**

`r_t = ln(C_t)-ln(C_{t-1})`
`raw_t = std(r, window=20, ddof=0)`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `realized_vol_20` داخل فایل `safe27_specs.json`.

## 16. `ret_autocorr_1_30`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `logret_autocorr_1_30`
- **فرمول خام:**

`r_t = ln(C_t)-ln(C_{t-1})`
`raw_t = rolling_corr(r_t, r_{t-1}, window=30)`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `ret_autocorr_1_30` داخل فایل `safe27_specs.json`.

## 17. `return_skew_30`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `return_skew_30_logret`
- **فرمول خام:**

`r_t = ln(C_t)-ln(C_{t-1})`
`raw_t = rolling_skew(r, window=30)` با تعریف `pandas.Series.rolling(30).skew()`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `return_skew_30` داخل فایل `safe27_specs.json`.

## 18. `roc_10`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `roc_10_pct`
- **فرمول خام:**

`raw_t = C_t/C_{t-10}-1`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `roc_10` داخل فایل `safe27_specs.json`.

## 19. `rsi_centered_14`

- **خانواده:** `signed`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `rsi_wilder_centered`
- **فرمول خام:**

RSI به روش Wilder:
`Δ_t=C_t-C_{t-1}`، `Gain=max(Δ,0)`، `Loss=max(-Δ,0)`
`AvgGain=WilderEMA(Gain, α=1/14)` و `AvgLoss=WilderEMA(Loss, α=1/14)`
`RSI14=100-100/(1+AvgGain/AvgLoss)`
`raw_t=(RSI14_t-50)/50`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `rsi_centered_14` داخل فایل `safe27_specs.json`.

## 20. `tr_to_atr_14`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `tr_to_atr14_ema`
- **فرمول خام:**

`ratio_t = TR_t/ATR14_t`
مقدار پیش‌تبدیل‌شده: `x_t=-|ratio_t|`

- **پیش‌تبدیل:** `x=-|raw|`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `tr_to_atr_14` داخل فایل `safe27_specs.json`.

## 21. `up_close_ratio_5`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `up_close_ratio_5`
- **فرمول خام:**

`I_t = 1` اگر `C_t>C_{t-1}`، وگرنه `0`
`raw_t = mean(I, window=5)`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `q0.2_0.8` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `up_close_ratio_5` داخل فایل `safe27_specs.json`.

## 22. `upside_semivol_20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `upside_semivol_20_logret`
- **فرمول خام:**

`r_t = ln(C_t)-ln(C_{t-1})`
`raw_t = sqrt(mean(max(r_t,0)^2, window=20))`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `upside_semivol_20` داخل فایل `safe27_specs.json`.

## 23. `vol_over_ema20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `vol_over_ema20_extra`
- **فرمول خام:**

`ratio_t = V_t/EMA(V,20)_t`
پیش‌تبدیل: `x_t = -(1-ratio_t)=ratio_t-1`

- **پیش‌تبدیل:** `x=-(1-raw)=raw-1`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `vol_over_ema20` داخل فایل `safe27_specs.json`.

## 24. `vol_over_median20`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `vol_over_median20_extra`
- **فرمول خام:**

`ratio_t = V_t/median(V, window=20)_t`
پیش‌تبدیل: `x_t = -(1-ratio_t)=ratio_t-1`

- **پیش‌تبدیل:** `x=-(1-raw)=raw-1`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `vol_over_median20` داخل فایل `safe27_specs.json`.

## 25. `mom_stoch_rsi_14_14_3`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `2`
- **Candidate در کد:** `stoch_rsi_wilder_14_sma3`
- **فرمول خام:**

ابتدا `RSI14` به روش Wilder ساخته می‌شود.
`StochRSI_t = (RSI_t-min(RSI,14)_t)/(max(RSI,14)_t-min(RSI,14)_t)`
`K_t = mean(StochRSI, window=3)`
پیش‌تبدیل: `x_t = 1-K_t`

- **پیش‌تبدیل:** `x=1-raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `mom_stoch_rsi_14_14_3` داخل فایل `safe27_specs.json`.

## 26. `vol_ratio_20_100`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `vol_ratio_ema_20_100`
- **فرمول خام:**

`ratio_t = EMA(V,20)_t/EMA(V,100)_t`
پیش‌تبدیل: `x_t = 1-ratio_t`

- **پیش‌تبدیل:** `x=1-raw`
- **روش نرمال‌سازی نهایی:** `fit_interior` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `vol_ratio_20_100` داخل فایل `safe27_specs.json`.

## 27. `log_range_over_vol_100`

- **خانواده:** `positive`
- **Lag داخل هر Symbol:** `1`
- **Candidate در کد:** `log_close_rollrange_over_rv_100`
- **فرمول خام:**

`HH100^C_t = rolling_max(C,100)` و `LL100^C_t = rolling_min(C,100)`
`LogRange_t = ln(HH100^C_t/LL100^C_t)`
`RV100_t = std(ln(C_t)-ln(C_{t-1}), window=100, ddof=0)`
`raw_t = LogRange_t/RV100_t`

- **پیش‌تبدیل:** بدون تغییر: `x=raw`
- **روش نرمال‌سازی نهایی:** `linear_fit_interior_clip` با Scope برابر `per_symbol`.
- **پارامترهای دقیق:** در ورودی `log_range_over_vol_100` داخل فایل `safe27_specs.json`.

---

## فهرست نهایی ۲۷ ویژگی

```text
atr_pct_14
band_bb_percB_20_2
channel_pos_20
close_location_value
dmi_balance_14
dollar_vol_rel_20
downside_semivol_20
efficiency_ratio_20
ema_gap_atr_20
ema_slope_atr_20_5
macd_hist_atr
parkinson_vol_20
percent_b_20
range_compression_20_100
realized_vol_20
ret_autocorr_1_30
return_skew_30
roc_10
rsi_centered_14
tr_to_atr_14
up_close_ratio_5
upside_semivol_20
vol_over_ema20
vol_over_median20
mom_stoch_rsi_14_14_3
vol_ratio_20_100
log_range_over_vol_100
```

## نکته مهم برای داده‌های جدید

پارامترهای `low/width` یا `a/b` موجود در `safe27_specs.json` از دیتاست مرجع استخراج و Freeze شده‌اند. برای تولید داده‌های بعدی، نباید آن‌ها را دوباره روی کل داده آینده Fit کرد؛ همان پارامترها باید اعمال شوند تا مقیاس و مرزهای فازی با دیتاست پروژه ثابت بماند.
