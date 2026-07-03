بسته فرمول‌های ۲۷ ویژگی تأییدشده

برای ارسال به همکار، هر سه فایل را با هم بفرستید:
- SAFE27_FEATURE_FORMULAS_FA.md
- safe27_specs.json
- safe27_builder.py

اجرای نمونه:
python safe27_builder.py --raw raw_ohlcv_5m_all_encoded.csv --spec safe27_specs.json --out safe27_features.csv

ورودی لازم:
datetime,symbol,open,high,low,close,volume
