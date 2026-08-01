# Knowledge Index

Graph JSON: `.opencode/knowledge/graph.json`

## jq recipes

Top in-degree nodes:
```
jq '.nodes | sort_by(-.in_degree) | .[0:10] | map({id, in_degree})' .opencode/knowledge/graph.json
```
Top importers (group by .to):
```
jq '.edges | group_by(.to) | map({id: .[0].to, in: length}) | sort_by(-.in) | .[0:10]' .opencode/knowledge/graph.json
```
Who imports gpu_engine.py:
```
jq '[.edges[] | select(.to == "gpu_fuzzy_trader/backtest/gpu_engine.py") | .from]' .opencode/knowledge/graph.json
```
Who imports cpu_engine.py:
```
jq '[.edges[] | select(.to == "gpu_fuzzy_trader/backtest/cpu_engine.py") | .from]' .opencode/knowledge/graph.json
```
Who imports config.py:
```
jq '[.edges[] | select(.to == "gpu_fuzzy_trader/config.py") | .from]' .opencode/knowledge/graph.json
```
