#!/usr/bin/env python3
"""
Download 5-min klines for 2024-01-01 to 2024-12-31 (to extend train_2.csv)
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
CACHE_DIR = Path('.cache/prices/binance_5m_2024')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
        cur = data[-1][0] + 5 * 60 * 1000
        if len(data) < BATCH_SIZE:
            break
        time.sleep(0.1)
    return all_rows


def download_symbol(symbol):
    cache_file = CACHE_DIR / f'{symbol}.json'
    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)  # up to end of 2024
    start_ms = dt_to_ms(start_dt)
    end_ms = dt_to_ms(end_dt)

    if cache_file.exists():
        print(f'  [{symbol}] loading from cache')
        with open(cache_file) as f:
            return json.load(f)

    print(f'  [{symbol}] downloading from Binance (2024 data)...')
    rows = fetch_klines(symbol, start_ms, end_ms)
    with open(cache_file, 'w') as f:
        json.dump(rows, f)
    print(f'    got {len(rows)} candles')
    return rows


def main():
    train_2024_rows = []

    # 2024-12-31 23:59:59 -> last candle at 23:55
    end_2024_ms = dt_to_ms(datetime(2025, 1, 1, tzinfo=timezone.utc))

    for sym in SYMBOLS:
        rows = download_symbol(sym)
        sym_id = SYMBOL_TO_ID[sym]
        for k in rows:
            open_t = int(k[0])
            if open_t >= end_2024_ms:
                continue
            o, h, l, c, v = k[1], k[2], k[3], k[4], k[5]
            dt_str = ms_to_dt(open_t).strftime('%Y-%m-%d %H:%M')
            train_2024_rows.append([dt_str, sym_id, o, h, l, c, v])

    train_2024_rows.sort(key=lambda r: (r[0], r[1]))

    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)

    header = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    train_2024_path = out_dir / 'train_1_extended_2024.csv'
    
    with open(train_2024_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(train_2024_rows)

    print(f'\n✓ data/train_1_extended_2024.csv : {len(train_2024_rows)} rows')
    if train_2024_rows:
        print(f'  train range: {train_2024_rows[0][0]} .. {train_2024_rows[-1][0]}')


if __name__ == '__main__':
    main()
