# Phase 5 — Out-of-Sample Evaluation

> **ماژول:** `gpu_fuzzy_trader/phases/phase5_oos.py` → `OOS_Evaluator`  
> **ورودی:** `data/test.csv`, `outputs/long.json`, `outputs/short.json`  
> **خروجی:** `outputs/reports/test_*.json`, `outputs/reports/test_*.csv`, `outputs/reports/test_*.png`

---

## ۱. هدف Phase 5

ارزیابی نهایی استراتژی روی داده‌های کاملاً ندیده (test set). این تنها phase است که باید به عنوان "حقیقت" در نظر گرفته شود.

**مهم:** `data/test.csv` هرگز در Phase 1-4 استفاده نمی‌شود. هر استفاده از test data در مراحل قبلی = data leakage.

---

## ۲. جریان کامل

```python
def run(self):
    # ۱. بارگذاری استراتژی‌ها
    strategies = self.load_strategies()  # long.json و short.json
    
    # ۲. آماده‌سازی داده‌ها
    datasets = self._load_datasets_by_split()
    # datasets = {"train": ..., "validation": ..., "test": ...}
    
    # ۳. ارزیابی هر استراتژی روی هر split
    for direction, strategy in strategies.items():
        for split, split_df in datasets.items():
            metrics, per_symbol_rows, trade_log = self._evaluate_strategy(split_df, strategy, direction)
    
    # ۴. ذخیره گزارش‌ها
    self._save_report(test_metrics, direction)
    self._save_per_symbol_csv(all_per_symbol)
```

**چرا روی train و validation هم ارزیابی می‌شود؟** برای مقایسه عملکرد در سه split و تشخیص overfitting. اگر train >> validation >> test، مشکل overfitting داریم.

---

## ۳. آماده‌سازی Test Data

```python
@staticmethod
def prepare_test_data(test_csv_path):
    loader = Data_Loader()
    return loader.load_dataset(test_csv_path)
```

**دقیقاً همان pipeline آموزش:**
1. خواندن CSV
2. Parse datetime
3. مرتب‌سازی (symbol, datetime)
4. حذف آخرین ۲۸۸ ردیف per symbol
5. حذف ردیف‌های با label NaN
6. پر کردن NaN feature ها با صفر
7. محاسبه `_symbol_bar_index`

---

## ۴. Metrics گزارش‌شده

### Metrics اصلی

| Metric | فرمول | توضیح |
|--------|-------|-------|
| `total_return_pct` | `(final_equity / INITIAL_CAPITAL - 1) × 100` | بازده کل |
| `max_drawdown_pct` | `max((peak - equity) / peak × 100)` | بیشترین افت از peak |
| `win_rate` | `wins / executed_trades × 100` | درصد trade های سودآور |
| `profit_factor` | `gross_profit / gross_loss` | نسبت سود به ضرر |
| `sortino_ratio` | `mean_excess_return / downside_deviation` | ریسک-بازده تعدیل‌شده |
| `executed_trades` | تعداد | تعداد trade های اجراشده |
| `account_ruined` | bool | آیا equity به صفر رسید؟ |

### Per-Symbol Metrics

برای هر symbol:
- `trade_count`: تعداد trade ها
- `win_rate`: درصد trade های سودآور
- `net_pnl`: سود/زیان خالص

---

## ۵. Zero-Trade Handling

```python
if metrics["executed_trades"] == 0:
    metrics["account_ruined"] = False
    metrics["total_return_pct"] = 0.0
```

اگر هیچ trade ای اجرا نشد:
- بازده = 0٪ (نه منفی)
- account_ruined = False (حساب خراب نشده، فقط trade نشده)

---

## ۶. گزارش‌های تولیدشده

### JSON Reports

**`test_long_report.json` / `test_short_report.json`:**
```json
{
  "direction": "long",
  "total_return_pct": 12.5,
  "max_drawdown_pct": 8.3,
  "win_rate": 0.52,
  "profit_factor": 1.8,
  "executed_trades": 450,
  "account_status": "survived",
  "final_equity": 1125.0,
  "per_symbol_metrics": {...}
}
```

### CSV Reports

**`test_per_symbol_performance.csv`:**
```
direction, symbol, trade_count, win_rate, net_pnl
long, BTCUSDT, 45, 0.53, 12.3
long, ETHUSDT, 38, 0.50, 8.7
...
```

### Visual Reports

- **Equity Curve:** نمودار رشد سرمایه در طول زمان
- **Per-Rule Breakdown:** عملکرد هر rule به صورت جداگانه
- **Distribution:** توزیع بازده trade ها
- **Spearman Correlation:** همبستگی feature ها با نتایج
- **Feature Stratified Performance:** عملکرد در سطوح مختلف هر feature

---

## ۷. تفسیر نتایج

### علائم خوب
- `total_return_pct` مثبت در test
- `max_drawdown_pct` < 15٪
- `win_rate` > 45٪
- `profit_factor` > 1.5
- عملکرد مشابه در train، validation، و test (بدون overfitting)

### علائم نگران‌کننده
- `total_return_pct` منفی در test
- `max_drawdown_pct` > 20٪
- `executed_trades` خیلی کم (< 100)
- تفاوت زیاد بین train و test performance

### تشخیص Overfitting

```
اگر: train_return >> validation_return >> test_return
→ overfitting در Phase 2 یا 3

اگر: train_return ≈ validation_return >> test_return
→ distribution shift بین train/val و test
```

---

## ۸. نکات عملی

### اگر test return منفی است:
- بررسی کنید آیا validation return هم منفی بود
- اگر validation خوب بود اما test بد: distribution shift احتمالی
- `PHASE3_VAL_SORTINO_RATIO_GATE` را افزایش دهید
- `PHASE3_VAL_GATE_PENALTY` را افزایش دهید

### اگر trade های کمی اجرا شد:
- بررسی کنید آیا feature های انتخاب‌شده در test data وجود دارند
- `MIN_TRADE_SUPPORT` را در Phase 2 کاهش دهید
- `PHASE3_MIN_SYMBOL_COVERAGE` را کاهش دهید

### اگر drawdown خیلی بالا است:
- `PHASE4_MAX_WORST_DRAWDOWN_PCT` را کاهش دهید
- `PHASE4_SL_MAX` را کاهش دهید
- `PHASE4_CAPITAL_PCT_MAX` را کاهش دهید
