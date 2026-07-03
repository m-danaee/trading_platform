#!/usr/bin/env python3
"""
Download 5-min klines from Binance and create train_1.csv / test_1.csv
for the 10 symbols: ADA, AVAX, BNB, BTC, DOGE, DOT, ETH, SOL, TRX, XRP (USDT pairs)

Date ranges:
  train_1.csv : 2025-01-01 00:00 -> 2025-09-30 23:55
  test_1.csv  : 2025-10-01 00:00 -> 2025-12-31 23:55  (test ends 2026-01-01 exclusive)
"""
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
import csv

SYMBOLS = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'SOL', 'TRX', 'XRP']
SYMBOL_TO_ID = {s: i + 1 for i, s in enumerate(SYMBOLS)}
INTERVAL = '5m'
CACHE_DIR = Path('.cache/prices/binance_5m')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Binance limit per call = 1000 candles; 5m * 1000 = 5000 min ≈ 3.47 days
BATCH_SIZE = 1000


def dt_to_ms(dt):
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fetch_klines(symbol, start_ms, end_ms):
    """Fetch all 5m klines for symbol in [start_ms, end_ms)."""
    url = 'https://api.binance.com/api/v3/klines'
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = urllib.parse.urlencode({
            'symbol': f'{symbol}USDT',
            'interval': INTERVAL,
            'startTime': cur,
            'endTime': end_ms,
            'limit': BATCH_SIZE,
        })
        req_url = f'{url}?{params}'
        data = []
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req_url, timeout=30) as r:
                    data = json.loads(r.read().decode())
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if not data:
            break
        all_rows.extend(data)
        # move start to last candle open time + 1
        cur = data[-1][0] + 5 * 60 * 1000
        if len(data) < BATCH_SIZE:
            break
        time.sleep(0.1)  # be polite
    return all_rows


def download_symbol(symbol):
    cache_file = CACHE_DIR / f'{symbol}.json'
    # Train+test range: 2025-01-01 to 2026-01-01 (exclusive)
    start_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    start_ms = dt_to_ms(start_dt)
    end_ms = dt_to_ms(end_dt)

    if cache_file.exists():
        print(f'  [{symbol}] loading from cache')
        with open(cache_file) as f:
            return json.load(f)

    print(f'  [{symbol}] downloading from Binance...')
    rows = fetch_klines(symbol, start_ms, end_ms)
    with open(cache_file, 'w') as f:
        json.dump(rows, f)
    print(f'    got {len(rows)} candles')
    return rows


def main():
    train_rows = []
    test_rows = []

    # Date ranges
    train_end_ms = dt_to_ms(datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc))
    test_start_ms = train_end_ms
    test_end_ms = dt_to_ms(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))  # exclusive, so last candle is 2025-12-31 23:55

    for sym in SYMBOLS:
        rows = download_symbol(sym)
        sym_id = SYMBOL_TO_ID[sym]
        for k in rows:
            open_t = int(k[0])
            if open_t >= test_end_ms:
                continue  # beyond test range
            o, h, l, c, v = k[1], k[2], k[3], k[4], k[5]
            dt_str = ms_to_dt(open_t).strftime('%Y-%m-%d %H:%M')
            if open_t < train_end_ms:
                train_rows.append([dt_str, sym_id, o, h, l, c, v])
            else:
                test_rows.append([dt_str, sym_id, o, h, l, c, v])

    # Sort by datetime, then symbol
    train_rows.sort(key=lambda r: (r[0], r[1]))
    test_rows.sort(key=lambda r: (r[0], r[1]))

    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)

    header = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    with open(out_dir / 'train_1.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(train_rows)
    with open(out_dir / 'test_1.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(test_rows)

    print(f'\n✓ data/train_1.csv : {len(train_rows)} rows')
    print(f'✓ data/test_1.csv  : {len(test_rows)} rows')
    if train_rows:
        print(f'  train range: {train_rows[0][0]} .. {train_rows[-1][0]}')
    if test_rows:
        print(f'  test  range: {test_rows[0][0]} .. {test_rows[-1][0]}')


if __name__ == '__main__':
    main()
