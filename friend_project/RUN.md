# trader_bigdata_governor

```bash
cd /home/ubuntu/bigdata_trader_2
mkdir -p logs
nohup env PYTHONPATH=/home/ubuntu/bigdata_trader_2 \
  /home/ubuntu/trading_platform-main/.venv/bin/python \
  -m gpu_fuzzy_trader.auto_search \
  --hours 7 \
  --output-root outputs/trader_bigdata_governor_7h \
  --start-direction long \
  > logs/trader_bigdata_governor_7h.log 2>&1 &
```

```bash
zip -r trader_bigdata_governor_7h_full.zip outputs/trader_bigdata_governor_7h logs/trader_bigdata_governor_7h.log -x "*.png"
```
