"""Read-only HTML dashboard for existing pipeline artifacts.

The dashboard deliberately uses only the Python standard library. It reads a
run directory's JSON/CSV/PNG outputs, renders a self-contained HTML page, and
can optionally serve that directory over a local HTTP server. It never loads
market data or executes a pipeline phase.

Examples
--------
    .venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/run_a
    .venv/bin/python -m gpu_fuzzy_trader.dashboard --output outputs/run_a --serve
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any


_DIRECTIONS = ("long", "short")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _load_json(path: Path) -> Any | None:
    """Load a JSON artifact, returning ``None`` for missing/broken files."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _history_rows(value: Any) -> list[dict[str, Any]]:
    """Normalize supported Phase 2 history shapes to a list of records."""
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("history", "generations", "records"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _split_metrics(
    rb_report: Any,
    oos_report: Any,
    generalization: Any,
) -> dict[str, dict[str, Any]]:
    """Collect train/validation/test metrics from the available reports."""
    if isinstance(generalization, dict):
        split_metrics = generalization.get("split_metrics")
        if isinstance(split_metrics, dict):
            return {
                str(split): dict(metrics)
                for split, metrics in split_metrics.items()
                if isinstance(metrics, dict)
            }

    result: dict[str, dict[str, Any]] = {}
    if isinstance(rb_report, dict):
        for split, key in (("train", "train_metrics"), ("validation", "valid_metrics")):
            metrics = rb_report.get(key)
            if isinstance(metrics, dict):
                result[split] = dict(metrics)
    if isinstance(oos_report, dict):
        result["test"] = {
            key: value
            for key, value in oos_report.items()
            if key not in {"direction", "per_symbol_metrics"}
        }
    return result


def _direction_data(root: Path, direction: str) -> dict[str, Any]:
    """Read all dashboard artifacts for one trading direction."""
    reports = root / "reports"
    rb_report = _load_json(reports / f"rb_governor_{direction}_report.json")
    oos_report = _load_json(reports / f"test_{direction}_report.json")
    generalization = _load_json(
        reports / f"generalization_diagnostics_{direction}.json",
    )
    strategy = _load_json(root / f"{direction}.json")
    history_raw = _load_json(root / f"phase2_{direction}_history.json")

    assets: list[str] = []
    if reports.is_dir():
        for path in sorted(reports.iterdir()):
            if path.is_file() and direction in path.name.lower():
                if path.suffix.lower() in _IMAGE_SUFFIXES:
                    assets.append(path.relative_to(root).as_posix())

    strategy_dict = strategy if isinstance(strategy, dict) else None
    rb_dict = rb_report if isinstance(rb_report, dict) else None
    if strategy_dict is None:
        status = "missing"
    elif bool(strategy_dict.get("deployment_accepted")):
        status = "accepted"
    elif rb_dict is not None and bool(rb_dict.get("fail_closed")):
        status = "fail-closed"
    elif strategy_dict.get("reason"):
        status = "rejected"
    else:
        status = "available"

    return {
        "status": status,
        "strategy": strategy_dict,
        "rb": rb_dict,
        "oos": oos_report if isinstance(oos_report, dict) else None,
        "generalization": (
            generalization if isinstance(generalization, dict) else None
        ),
        "split_metrics": _split_metrics(rb_report, oos_report, generalization),
        "phase2_history": _history_rows(history_raw),
        "assets": assets,
    }


def build_dashboard_data(output_dir: str | Path) -> dict[str, Any]:
    """Build a JSON-serialisable snapshot from an output directory."""
    root = Path(output_dir).expanduser()
    config_path = root / "reports" / "config_audit.json"
    config_audit = _load_json(config_path)
    directions = {
        direction: _direction_data(root, direction)
        for direction in _DIRECTIONS
    }
    accepted = sum(
        1 for data in directions.values() if data["status"] == "accepted"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root),
        "config_audit_available": isinstance(config_audit, dict),
        "config_audit": config_audit if isinstance(config_audit, dict) else None,
        "accepted_directions": accepted,
        "directions": directions,
    }


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPU Fuzzy Trader Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#0c111b; --panel:#141c2b; --panel2:#192337; --text:#e7edf7; --muted:#91a0b8; --line:#2a3850; --good:#38d39f; --warn:#f2bd57; --bad:#ff7777; --accent:#79a8ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:linear-gradient(145deg,#0b1019,#101a2b); color:var(--text); font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
    main { max-width:1400px; margin:0 auto; padding:28px; }
    header { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; margin-bottom:24px; }
    h1,h2,h3,p { margin-top:0; } h1 { margin-bottom:5px; font-size:30px; letter-spacing:-.03em; } h2 { font-size:18px; margin-bottom:16px; } h3 { font-size:16px; margin-bottom:8px; }
    .eyebrow { color:var(--accent); font-size:11px; font-weight:700; letter-spacing:.16em; margin-bottom:5px; }
    .muted { color:var(--muted); } .small { font-size:12px; }
    button { border:1px solid var(--line); background:var(--panel2); color:var(--text); border-radius:9px; padding:9px 13px; cursor:pointer; } button:hover { border-color:var(--accent); }
    .grid { display:grid; gap:14px; grid-template-columns:repeat(4,minmax(0,1fr)); margin-bottom:22px; }
    .card,.panel { background:rgba(20,28,43,.92); border:1px solid var(--line); border-radius:13px; box-shadow:0 12px 30px rgba(0,0,0,.14); }
    .card { padding:16px; } .card-label { color:var(--muted); font-size:12px; } .card-value { font-size:23px; font-weight:700; margin-top:4px; }
    .panel { padding:20px; margin-bottom:22px; } .direction-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .direction { background:var(--panel2); border:1px solid var(--line); border-radius:11px; padding:17px; min-width:0; }
    .direction-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }
    .direction-head h3 { margin:0; text-transform:capitalize; }
    .pill { border-radius:999px; padding:3px 9px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
    .pill-accepted { color:#081d17; background:var(--good); } .pill-fail-closed,.pill-rejected { color:#281717; background:var(--bad); } .pill-available { color:#2b210b; background:var(--warn); } .pill-missing { color:var(--muted); background:#33425b; }
    .mini-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-bottom:15px; } .mini { border:1px solid var(--line); border-radius:8px; padding:9px; } .mini .card-value { font-size:17px; }
    table { width:100%; border-collapse:collapse; font-size:12px; } th,td { border-bottom:1px solid var(--line); padding:8px 6px; text-align:right; white-space:nowrap; } th:first-child,td:first-child { text-align:left; } th { color:var(--muted); font-weight:600; }
    .table-wrap { overflow:auto; } .empty { color:var(--muted); border:1px dashed var(--line); border-radius:8px; padding:12px; font-size:12px; }
    .history { margin-top:16px; } canvas { display:block; width:100%; height:150px; background:#111a29; border-radius:8px; border:1px solid var(--line); }
    .assets { display:flex; flex-wrap:wrap; gap:8px; margin-top:15px; } .assets a { color:var(--accent); text-decoration:none; border:1px solid var(--line); border-radius:7px; padding:5px 8px; font-size:11px; } .assets a:hover { border-color:var(--accent); }
    .config { white-space:pre-wrap; max-height:280px; overflow:auto; background:#0f1725; border-radius:8px; padding:12px; color:#b9c8df; font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
    @media (max-width:900px) { .grid,.direction-grid { grid-template-columns:1fr 1fr; } .mini-grid { grid-template-columns:1fr 1fr; } }
    @media (max-width:600px) { main { padding:16px; } header { display:block; } header button { margin-top:10px; } .grid,.direction-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <div><div class="eyebrow">GPU FUZZY TRADER</div><h1>Run dashboard</h1><p class="muted" id="subtitle"></p></div>
    <button onclick="location.reload()">Refresh artifacts</button>
  </header>
  <section class="grid" id="summary"></section>
  <section class="panel"><h2>Direction health</h2><div class="direction-grid" id="directions"></div></section>
  <section class="panel"><h2>Configuration audit</h2><div id="config"></div></section>
</main>
<script>
const DATA = __DASHBOARD_DATA__;
const esc = value => String(value ?? "").replace(/[&<>\"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;", "'":"&#39;"}[ch]));
const number = (value, digits=2) => { const n=Number(value); return Number.isFinite(n) ? n.toFixed(digits) : "—"; };
const pct = value => { const n=Number(value); return Number.isFinite(n) ? `${n.toFixed(2)}%` : "—"; };
const int = value => { const n=Number(value); return Number.isFinite(n) ? Math.round(n).toLocaleString() : "—"; };
function statusLabel(status) { return ({"accepted":"accepted","fail-closed":"fail-closed","rejected":"rejected","available":"available","missing":"missing"}[status] || status); }
function statusClass(status) { return `pill pill-${status || "missing"}`; }
function splitTable(splits) {
  const names=["train","validation","test"].filter(name => splits && splits[name]);
  if (!names.length) return '<div class="empty">No split metrics found.</div>';
  return `<div class="table-wrap"><table><thead><tr><th>Split</th><th>Return</th><th>PF</th><th>Win rate</th><th>Drawdown</th><th>Trades</th></tr></thead><tbody>${names.map(name => { const m=splits[name]||{}; return `<tr><td>${esc(name)}</td><td>${pct(m.total_return_pct)}</td><td>${number(m.profit_factor)}</td><td>${pct(m.win_rate)}</td><td>${pct(m.max_drawdown_pct)}</td><td>${int(m.executed_trades)}</td></tr>`; }).join("")}</tbody></table></div>`;
}
function historyValue(row) { for (const key of ["max_return_pct","max_robust_return_pct","mean_robust_return_pct","mean_raw_train_return_pct","mean_return_pct"]) { if (Number.isFinite(Number(row[key]))) return Number(row[key]); } return null; }
function drawHistory(id, rows) {
  const canvas=document.getElementById(id); if (!canvas) return;
  const ctx=canvas.getContext("2d"), dpr=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect(); canvas.width=rect.width*dpr; canvas.height=150*dpr; ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,rect.width,150); const points=(rows||[]).map((row,i)=>({x:Number(row.generation ?? i), y:historyValue(row)})).filter(p=>p.y!==null);
  if (points.length<2) { ctx.fillStyle="#91a0b8"; ctx.font="12px system-ui"; ctx.fillText("No chartable Phase 2 history found.",12,24); return; }
  const ys=points.map(p=>p.y), min=Math.min(...ys), max=Math.max(...ys), span=Math.max(1e-9,max-min), left=35, right=12, top=12, bottom=24, w=rect.width-left-right, h=150-top-bottom;
  ctx.strokeStyle="#2a3850"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(left,top); ctx.lineTo(left,top+h); ctx.lineTo(left+w,top+h); ctx.stroke();
  ctx.strokeStyle="#79a8ff"; ctx.lineWidth=2; ctx.beginPath(); points.forEach((p,i)=>{ const x=left+(i/Math.max(1,points.length-1))*w, y=top+(1-(p.y-min)/span)*h; i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
  ctx.fillStyle="#91a0b8"; ctx.font="11px system-ui"; ctx.fillText(`${min.toFixed(2)}%`,3,top+h); ctx.fillText(`${max.toFixed(2)}%`,3,top+8); ctx.fillText(`gen ${points[0].x}`,left,149); ctx.fillText(`gen ${points[points.length-1].x}`,Math.max(left,rect.width-65),149);
}
function directionCard(direction, data) {
  const rb=data.rb||{}, oos=data.oos||{}, strategy=data.strategy||{};
  const reason=rb.reason || strategy.reason || "";
  const assets=(data.assets||[]).map(asset=>`<a href="${encodeURI(asset)}">${esc(asset.split('/').pop())}</a>`).join("");
  return `<article class="direction"><div class="direction-head"><h3>${esc(direction)}</h3><span class="${statusClass(data.status)}">${esc(statusLabel(data.status))}</span></div><div class="mini-grid"><div class="mini"><div class="card-label">OOS return</div><div class="card-value">${pct(oos.total_return_pct)}</div></div><div class="mini"><div class="card-label">OOS trades</div><div class="card-value">${int(oos.executed_trades)}</div></div><div class="mini"><div class="card-label">RB score</div><div class="card-value">${number(rb.rb_score)}</div></div><div class="mini"><div class="card-label">Rules</div><div class="card-value">${int(strategy.rules_set ? strategy.rules_set.length : rb.selected_rules)}</div></div></div>${reason ? `<p class="muted small">${esc(reason)}</p>` : ""}<h3>Split metrics</h3>${splitTable(data.split_metrics)}<div class="history"><h3>Phase 2 trend</h3><canvas id="history-${esc(direction)}"></canvas></div>${assets ? `<div class="assets">${assets}</div>` : ""}</article>`;
}
function render() {
  document.getElementById("subtitle").textContent=`${DATA.output_dir} · generated ${DATA.generated_at}`;
  document.getElementById("summary").innerHTML=`<div class="card"><div class="card-label">Accepted directions</div><div class="card-value">${int(DATA.accepted_directions)} / 2</div></div><div class="card"><div class="card-label">Config audit</div><div class="card-value">${DATA.config_audit_available ? "available" : "missing"}</div></div><div class="card"><div class="card-label">Long status</div><div class="card-value">${esc(statusLabel(DATA.directions.long.status))}</div></div><div class="card"><div class="card-label">Short status</div><div class="card-value">${esc(statusLabel(DATA.directions.short.status))}</div></div>`;
  document.getElementById("directions").innerHTML=Object.entries(DATA.directions).map(([direction,data])=>directionCard(direction,data)).join("");
  Object.entries(DATA.directions).forEach(([direction,data])=>drawHistory(`history-${direction}`,data.phase2_history));
  document.getElementById("config").innerHTML=DATA.config_audit ? `<div class="config">${esc(JSON.stringify(DATA.config_audit,null,2))}</div>` : '<div class="empty">No config_audit.json found in this output directory.</div>';
}
render();
</script>
</body>
</html>
"""


def render_dashboard(data: dict[str, Any]) -> str:
    """Render a dashboard snapshot as a self-contained HTML document."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Prevent an artifact string from prematurely closing the JSON script tag.
    payload = (
        payload.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return _HTML_TEMPLATE.replace("__DASHBOARD_DATA__", payload)


def write_dashboard(
    output_dir: str | Path,
    dashboard_path: str | Path | None = None,
) -> Path:
    """Write ``dashboard.html`` for *output_dir* and return its path."""
    root = Path(output_dir).expanduser()
    target = Path(dashboard_path).expanduser() if dashboard_path else root / "dashboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_dashboard(build_dashboard_data(root)),
        encoding="utf-8",
    )
    return target


def serve_dashboard(output_dir: str | Path, host: str, port: int) -> None:
    """Serve an output directory, including its generated dashboard."""
    root = Path(output_dir).expanduser().resolve()
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((host, int(port)), handler)
    print(f"Dashboard available at http://{host}:{port}/dashboard.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the read-only pipeline dashboard")
    parser.add_argument(
        "--output", default="outputs", help="Pipeline output directory to inspect",
    )
    parser.add_argument(
        "--file", default=None, help="Dashboard HTML path (default: <output>/dashboard.html)",
    )
    parser.add_argument("--serve", action="store_true", help="Serve the output directory after rendering")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    args = parser.parse_args(argv)
    target = write_dashboard(args.output, args.file)
    print(f"Dashboard written to {target}")
    if args.serve:
        serve_dashboard(args.output, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

