#!/usr/bin/env node
/* Builds a Nexus knowledge graph for the trading_platform repo.
   Outputs .opencode/knowledge/graph.json + graph.md + index.md.
   Deterministic, safe to re-run, only writes under .opencode/knowledge. */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(process.argv[2] || '.');
const OUT = path.resolve(process.argv[3] || path.join(ROOT, '.opencode', 'knowledge'));
const PKG = 'gpu_fuzzy_trader';

function walk(dir, acc = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.name === '.git' || e.name === '.venv' || e.name === '.opencode' || e.name === '__pycache__' || e.name === 'node_modules') continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, acc);
    else if (e.name.endsWith('.py') || e.name.endsWith('.md') || e.name.endsWith('.ipynb')) acc.push(p);
  }
  return acc;
}

const files = walk(ROOT).sort();
const rel = (p) => path.relative(ROOT, p).replace(/\\/g, '/');

// map relpath -> {name, lang, lines, imports:Set, importedBy:Set}
const nodes = {};
for (const f of files) {
  const r = rel(f);
  const src = fs.readFileSync(f, 'utf8');
  nodes[r] = { name: r, lang: f.endsWith('.py') ? 'python' : f.endsWith('.ipynb') ? 'jupyter' : 'markdown', lines: src.split('\n').length, imports: new Set(), importedBy: new Set() };
}

// import regex: from <pkg>... import X  OR  import <pkg>
const importRe = /^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))/gm;

function resolveImport(node, spec) {
  if (!spec) return null;
  const base = spec.split('.')[0];
  if (base === PKG) {
    // map submodule to file
    const relTarget = spec.replaceAll('.', '/');
    // candidate file paths
    const candidates = [relTarget + '.py', spec === PKG ? 'gpu_fuzzy_trader/__init__.py' : null];
    for (const c of candidates) if (c && nodes[c]) return c;
    // try package __init__: gpu_fuzzy_trader.backtest -> backtest/__init__.py
    const initPath = relTarget + '/__init__.py';
    if (nodes[initPath]) return initPath;
    // fallback: find longest matching package dir file
    const baseDir = relTarget.replaceAll('.', '/');
    if (nodes[baseDir + '/__init__.py']) return baseDir + '/__init__.py';
    return null;
  }
  return null; // external lib
}

for (const [r, node] of Object.entries(nodes)) {
  if (!r.endsWith('.py')) continue;
  const src = fs.readFileSync(path.join(ROOT, r), 'utf8');
  let m;
  const re = new RegExp(importRe.source, 'gm');
  while ((m = re.exec(src)) !== null) {
    const spec = m[1] || m[2];
    if (!spec) continue;
    if (spec.startsWith('__') || spec.startsWith('.')) continue; // relative, skip
    const target = resolveImport(r, spec);
    if (target && target !== r) {
      node.imports.add(target);
      nodes[target].importedBy.add(r);
    }
  }
}

// edges
const edges = [];
for (const [r, node] of Object.entries(nodes)) {
  for (const t of node.imports) {
    edges.push({ from: r, to: t, tag: 'EXTRACTED', kind: 'import' });
  }
}

const nodeList = Object.entries(nodes).map(([r, n]) => ({
  id: r, name: n.name, lang: n.lang, lines: n.lines,
  in_degree: n.importedBy.size, out_degree: n.imports.size,
}));

const langStats = {};
for (const n of nodeList) {
  langStats[n.lang] = langStats[n.lang] || { files: 0, lines: 0 };
  langStats[n.lang].files++;
  langStats[n.lang].lines += n.lines;
}

const graph = {
  meta: {
    generator: 'nexus-knowledge-graph',
    version: '1.0',
    root: ROOT,
    commit: (() => { try { return require('child_process').execSync('git -C ' + ROOT + ' rev-parse HEAD', {encoding:'utf8'}).trim(); } catch { return 'n/a'; } })(),
    built_at: new Date().toISOString(),
  },
  nodes: nodeList,
  edges,
  lang_stats: langStats,
  counts: { nodes: nodeList.length, edges: edges.length },
};

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'graph.json'), JSON.stringify(graph, null, 2));

// ---- markdown summary ----
const byDegree = [...nodeList].sort((a, b) => b.in_degree - a.in_degree);
const hubs = byDegree.slice(0, 12);
const focus = [
  'gpu_fuzzy_trader/backtest/cpu_engine.py',
  'gpu_fuzzy_trader/backtest/gpu_engine.py',
  'gpu_fuzzy_trader/config.py',
  'RUN.md',
  'tests/benchmark/test_phase2_gpu_throughput.py',
  'tests/benchmark/test_phase2_numba_warmup.py',
];
const focusInfo = {};
for (const f of focus) {
  if (nodes[f]) {
    focusInfo[f] = {
      in_degree: nodes[f].importedBy.size,
      out_degree: nodes[f].imports.size,
      imported_by: [...nodes[f].importedBy].sort(),
      imports: [...nodes[f].imports].sort(),
    };
  }
}

let md = `# Nexus Knowledge Graph — trading_platform\n\n`;
md += `- **Built**: ${graph.meta.built_at}\n`;
md += `- **Commit**: \`${graph.meta.commit}\`\n`;
md += `- **Nodes**: ${graph.counts.nodes} | **Edges**: ${graph.counts.edges}\n\n`;
md += `## Language stats\n`;
for (const [k, v] of Object.entries(langStats)) md += `- ${k}: ${v.files} files / ${v.lines} lines\n`;
md += `\n## Hub nodes (top by in-degree)\n\n`;
md += `| file | in | out | lines |\n|---|---|---|---|\n`;
for (const h of hubs) md += `| ${h.id} | ${h.in_degree} | ${h.out_degree} | ${h.lines} |\n`;
md += `\n## Focus files\n\n`;
for (const f of focus) {
  if (!focusInfo[f]) continue;
  const fi = focusInfo[f];
  md += `### ${f}\n`;
  md += `- in_degree=${fi.in_degree}, out_degree=${fi.out_degree}\n`;
  if (fi.imported_by.length) md += `- **imported_by**: ${fi.imported_by.join(', ')}\n`;
  if (fi.imports.length) md += `- **imports**: ${fi.imports.join(', ')}\n`;
  md += `\n`;
}
fs.writeFileSync(path.join(OUT, 'graph.md'), md);

// ---- index.md with jq recipes ----
let idx = `# Knowledge Index\n\nGraph JSON: \`.opencode/knowledge/graph.json\`\n\n## jq recipes\n\n`;
idx += `Top in-degree nodes:\n\`\`\`\njq '.nodes | sort_by(-.in_degree) | .[0:10] | map({id, in_degree})' .opencode/knowledge/graph.json\n\`\`\`\n`;
idx += `Top importers (group by .to):\n\`\`\`\njq '.edges | group_by(.to) | map({id: .[0].to, in: length}) | sort_by(-.in) | .[0:10]' .opencode/knowledge/graph.json\n\`\`\`\n`;
idx += `Who imports gpu_engine.py:\n\`\`\`\njq '[.edges[] | select(.to == "gpu_fuzzy_trader/backtest/gpu_engine.py") | .from]' .opencode/knowledge/graph.json\n\`\`\`\n`;
idx += `Who imports cpu_engine.py:\n\`\`\`\njq '[.edges[] | select(.to == "gpu_fuzzy_trader/backtest/cpu_engine.py") | .from]' .opencode/knowledge/graph.json\n\`\`\`\n`;
idx += `Who imports config.py:\n\`\`\`\njq '[.edges[] | select(.to == "gpu_fuzzy_trader/config.py") | .from]' .opencode/knowledge/graph.json\n\`\`\`\n`;
fs.writeFileSync(path.join(OUT, 'index.md'), idx);

console.log('Wrote graph.json, graph.md, index.md');
console.log('nodes=' + graph.counts.nodes, 'edges=' + graph.counts.edges);
