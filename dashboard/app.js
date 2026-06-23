"use strict";

// Oracle Dashboard — buildless frontend. Fetches the read-only JSON API and
// renders KPIs + AntV G2 v5 widgets. Every widget degrades gracefully when its
// source returns verdict INSUFFICIENT_DATA / ERROR. No mutation, GET-only.

const REFRESH_MS = 60000;
const C = { accent: "#58a6ff", green: "#3fb950", red: "#f85149", amber: "#d29922" };
// Dark theme that matches the panel background (#0d1117 / #161b22).
const G2_THEME = { type: "classicDark", view: { viewFill: "transparent", plotFill: "transparent" } };

let refreshTimer = null;

// ---- G2 chart registry ----------------------------------------------------
// One Chart instance per container, reused across refreshes. We clear() and
// re-options() rather than recreate so the canvas isn't thrashed. When a tile
// degrades to a badge (placeholder), the cached chart is destroyed so the next
// data-bearing render rebuilds cleanly into the restored container.
const _charts = {};
function g2render(id, options, height) {
  const el = document.getElementById(id);
  if (!el) return;
  let c = _charts[id];
  if (!c) {
    el.innerHTML = "";
    c = new G2.Chart({ container: el, autoFit: true, height: height || 260, theme: G2_THEME });
    _charts[id] = c;
  }
  c.clear();
  c.options(options);
  c.render();
}
function g2destroy(id) {
  const c = _charts[id];
  if (c) { try { c.destroy(); } catch (e) { /* fail-open */ } delete _charts[id]; }
}

// ---- helpers --------------------------------------------------------------
async function api(path) {
  try {
    const res = await fetch("/api/" + path, { headers: { "Accept": "application/json" } });
    return await res.json();
  } catch (e) {
    return { verdict: "ERROR", error: String(e) };
  }
}

function isUsable(d) { return d && d.verdict !== "INSUFFICIENT_DATA" && d.verdict !== "ERROR"; }

function badge(d) {
  if (!d) return '<span class="badge error">no data</span>';
  if (d.verdict === "ERROR") return `<span class="badge error">error</span>`;
  if (d.verdict === "INSUFFICIENT_DATA") return '<span class="badge insufficient">insufficient data</span>';
  return "";
}

function placeholder(elId, d) {
  g2destroy(elId);
  const el = document.getElementById(elId);
  if (el) el.innerHTML = `<div style="padding:30px 0;text-align:center">${badge(d)}</div>`;
}

function fmtMoney(v) {
  if (v == null || isNaN(v)) return "—";
  const s = v < 0 ? "-" : "";
  return `${s}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}
function fmtPct(v, dp = 1) { return (v == null || isNaN(v)) ? "—" : (v * 100).toFixed(dp) + "%"; }
function fmtNum(v, dp = 3) { return (v == null || isNaN(v)) ? "—" : Number(v).toFixed(dp); }

function setSigned(elId, value, text) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.textContent = text;
  el.classList.remove("pos", "neg");
  if (value > 0) el.classList.add("pos");
  else if (value < 0) el.classList.add("neg");
}

// ---- KPI row --------------------------------------------------------------
// Sourced from the single-leg stores (realized_pnl_log / trading_history /
// active_trades) — the data this deployment actually produces.
async function loadKpis() {
  const k = await api("single-leg/kpis");
  const usable = isUsable(k);
  setSigned("kpi-pnl", usable ? k.realized_total : null,
    usable ? fmtMoney(k.realized_total) : "—");
  setSigned("kpi-today", usable ? k.today_realized : null,
    usable ? fmtMoney(k.today_realized) : "—");
  document.getElementById("kpi-winrate").textContent =
    (usable && k.win_rate != null) ? fmtPct(k.win_rate) : "—";
  document.getElementById("kpi-open").textContent =
    (usable && k.open_positions != null) ? k.open_positions : "—";
  document.getElementById("kpi-closed").textContent =
    (usable && k.closed_trades != null) ? k.closed_trades : "—";
  _grClosed = k;
  renderGreenRed();
}

// ---- RL Episodes ----------------------------------------------------------
async function loadEpisodes() {
  const d = await api("single-leg/episodes");
  const statsEl = document.getElementById("episodes-stats");
  const counts = (d && d.chosen_action_counts) || {};
  const labels = Object.keys(counts);
  if (!isUsable(d) || !labels.length) {
    placeholder("episodes-bars", d);
    if (statsEl) statsEl.innerHTML = isUsable(d) ? "" : badge(d);
    return;
  }
  g2render("episodes-bars", {
    type: "interval",
    autoFit: true,
    data: labels.map(k => ({ action: k, count: Number(counts[k]) })),
    encode: { x: "action", y: "count" },
    axis: { y: { title: "episodes" }, x: { title: null } },
    style: { fill: C.accent },
  }, 260);
  const s = d.stats || {};
  statsEl.innerHTML =
    `total <b>${s.total ?? 0}</b> · completed <b>${s.completed ?? 0}</b> · ` +
    `win rate <b>${fmtPct(s.win_rate)}</b> · mean net <b>${fmtNum(s.mean_net_pnl_pct, 2)}%</b>`;
}

// ---- Regime ---------------------------------------------------------------
async function loadRegime() {
  const d = await api("regime");
  const reasonsEl = document.getElementById("regime-reasons");
  const capEl = document.getElementById("regime-caption");
  if (!isUsable(d)) { placeholder("regime-gauge", d); reasonsEl.innerHTML = ""; if (capEl) capEl.innerHTML = ""; return; }
  const conf = Number(d.confidence || 0);
  const label = String(d.label || "—").replace(/_/g, " ");
  if (capEl) capEl.innerHTML =
    `<div class="g-label" style="color:${C.accent}">${escapeHtml(label)}</div>` +
    `<div class="g-sub">market regime</div>`;
  g2render("regime-gauge", {
    type: "gauge",
    autoFit: true,
    data: { value: { target: +(conf * 100).toFixed(0), total: 100 } },
    legend: false,
    scale: { color: { range: [C.accent, "#1c2330"] } },
    style: {
      textContent: (target) => `${target}%`,
      textFontSize: 30,
      textFontWeight: 700,
      textFill: "#e6edf3",
      arcShape: "round",
      pointerStroke: C.accent,
      pinStroke: C.accent,
    },
  }, 240);
  const reasons = Array.isArray(d.reasons) ? d.reasons : [];
  reasonsEl.innerHTML = reasons.length
    ? "<ul>" + reasons.map(r => `<li>${escapeHtml(String(r))}</li>`).join("") + "</ul>"
    : "";
}

// ---- Market sentiment (Fear & Greed) --------------------------------------
// 0-100 gauge with CNN-style fear->greed color bands, classification label,
// blended source, and the per-component breakdown. Read-only; degrades to a
// badge on INSUFFICIENT_DATA / ERROR.
const FG_BANDS = [
  { lo: 0, hi: 25, color: "#f85149" },   // Extreme Fear
  { lo: 25, hi: 45, color: "#d29922" },  // Fear
  { lo: 45, hi: 55, color: "#8b949e" },  // Neutral
  { lo: 55, hi: 75, color: "#56b870" },  // Greed
  { lo: 75, hi: 100, color: "#3fb950" }, // Extreme Greed
];
function fgColor(score) {
  for (const b of FG_BANDS) if (score >= b.lo && score <= b.hi) return b.color;
  return C.accent;
}
function renderSentimentKpi(d) {
  const val = document.getElementById("kpi-sentiment");
  const sub = document.getElementById("kpi-sentiment-sub");
  if (!val) return;
  if (!isUsable(d) || d.score == null) {
    val.textContent = "—"; val.style.color = "";
    if (sub) sub.textContent = "";
    return;
  }
  val.textContent = Math.round(d.score);
  val.style.color = fgColor(d.score);
  if (sub) sub.textContent = d.classification || "";
}
async function loadSentiment() {
  const d = await api("sentiment");
  renderSentimentKpi(d);
  const compEl = document.getElementById("sentiment-components");
  const capEl = document.getElementById("sentiment-caption");
  if (!isUsable(d) || d.score == null) {
    placeholder("sentiment-gauge", d);
    if (compEl) compEl.innerHTML = "";
    if (capEl) capEl.innerHTML = "";
    return;
  }
  const score = Number(d.score);
  const cls = d.classification || "—";
  const col = fgColor(score);
  if (capEl) capEl.innerHTML =
    `<div class="g-label" style="color:${col}">${escapeHtml(String(cls))}</div>` +
    `<div class="g-sub">fear &amp; greed</div>`;
  g2render("sentiment-gauge", {
    type: "gauge",
    autoFit: true,
    data: {
      value: {
        target: +score.toFixed(0),
        total: 100,
        thresholds: FG_BANDS.map(b => b.hi),  // [25,45,55,75,100]
      },
    },
    legend: false,
    scale: { color: { range: FG_BANDS.map(b => b.color) } },
    style: {
      textContent: (target) => `${target}`,
      textFontSize: 32,
      textFontWeight: 700,
      textFill: col,
      arcShape: "round",
      pointerStroke: "#e6edf3",
      pinStroke: "#e6edf3",
    },
  }, 240);

  const src = d.source ? `<span class="muted">source: ${escapeHtml(String(d.source))}` +
    (d.cnn_score != null ? ` · CNN ${Math.round(d.cnn_score)}` : "") +
    (d.custom_score != null ? ` · custom ${Math.round(d.custom_score)}` : "") +
    (d.from_cache ? " · cached" : "") + "</span>" : "";
  const comps = Array.isArray(d.components) ? d.components.filter(c => c && c.available && c.score != null) : [];
  const list = comps.length
    ? "<ul>" + comps.map(c =>
        `<li>${escapeHtml(String(c.name).replace(/_/g, " "))}: ` +
        `<b style="color:${fgColor(Number(c.score))}">${Math.round(c.score)}</b></li>`).join("") + "</ul>"
    : "";
  if (compEl) compEl.innerHTML = src + list;
}

// ---- Probability calibration (reliability curve from /calibration/pop) ----
async function loadProbability() {
  const [prob, pop] = await Promise.all([api("probability"), api("calibration/pop")]);
  const statsEl = document.getElementById("prob-stats");
  if (isUsable(pop) && Array.isArray(pop.buckets) && pop.buckets.length) {
    const obs = pop.buckets.map(b => ({
      x: Number(b.predicted ?? b.mid ?? b.p ?? b.bucket),
      y: Number(b.realized ?? b.actual ?? b.win_rate),
    }));
    g2render("prob-curve", {
      type: "view",
      autoFit: true,
      scale: { x: { domain: [0, 1] }, y: { domain: [0, 1] } },
      axis: { x: { title: "predicted" }, y: { title: "realized" } },
      children: [
        { type: "line", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], encode: { x: "x", y: "y" }, style: { stroke: "#8b949e", lineDash: [4, 4] }, tooltip: false },
        { type: "line", data: obs, encode: { x: "x", y: "y" }, style: { stroke: C.accent, lineWidth: 2 } },
        { type: "point", data: obs, encode: { x: "x", y: "y" }, style: { fill: C.accent } },
      ],
    }, 260);
  } else {
    placeholder("prob-curve", pop);
  }
  if (isUsable(prob)) {
    statsEl.innerHTML =
      `Brier <b>${fmtNum(prob.brier)}</b> · baseline <b>${fmtNum(prob.baseline_brier)}</b> · ` +
      `skill <b>${fmtNum(prob.skill)}</b> · n=${prob.sample_size}`;
  } else { statsEl.innerHTML = badge(prob); }
}

// ---- Agents (hit-rate + lift) ---------------------------------------------
async function loadAgents() {
  const d = await api("agents");
  if (!isUsable(d) || !Array.isArray(d.agents) || !d.agents.length) { placeholder("agents-bars", d); return; }
  const rows = d.agents.map(a => ({ agent: String(a.agent), hit_rate: Number(a.hit_rate) }));
  const children = [
    { type: "interval", encode: { x: "agent", y: "hit_rate" }, style: { fill: C.accent } },
  ];
  if (d.base_win_rate != null) {
    children.push({
      type: "lineY", data: [Number(d.base_win_rate)],
      style: { stroke: C.amber, lineDash: [4, 4] },
    });
  }
  g2render("agents-bars", {
    type: "view",
    autoFit: true,
    data: rows,
    coordinate: { transform: [{ type: "transpose" }] },
    scale: { y: { domain: [0, 1] } },
    axis: { y: { title: "hit rate" }, x: { title: null } },
    children,
  }, Math.max(260, rows.length * 34));
}

// ---- Feature importance ---------------------------------------------------
async function loadFeatures() {
  const d = await api("feature-importance");
  if (!isUsable(d) || !Array.isArray(d.features) || !d.features.length) { placeholder("features-bars", d); return; }
  const rows = d.features.map(f => ({ agent: String(f.agent), importance: Number(f.importance) }));
  g2render("features-bars", {
    type: "interval",
    autoFit: true,
    data: rows,
    encode: { x: "agent", y: "importance" },
    coordinate: { transform: [{ type: "transpose" }] },
    axis: { y: { title: "mean contribution" }, x: { title: null } },
    style: { fill: C.green },
  }, Math.max(260, rows.length * 34));
}

// ---- Weights --------------------------------------------------------------
async function loadWeights() {
  const d = await api("weights");
  const driftEl = document.getElementById("weights-drift");
  const cur = (d && d.current) || {};
  const keys = Object.keys(cur);
  if (!isUsable(d) || !keys.length) { placeholder("weights-bars", d); driftEl.innerHTML = ""; return; }
  const rows = keys.map(k => ({ agent: k, weight: Number(cur[k]) }));
  g2render("weights-bars", {
    type: "interval",
    autoFit: true,
    data: rows,
    encode: { x: "agent", y: "weight" },
    axis: { y: { title: "weight" }, x: { title: null } },
    style: { fill: C.accent },
  }, 260);
  driftEl.innerHTML = `snapshots <b>${d.snapshots ?? 0}</b> · drift <b>${fmtNum(d.drift, 3)}</b>`;
}

// ---- EV attribution -------------------------------------------------------
async function loadEv() {
  const d = await api("ev-attribution");
  let buckets = d && (d.ev_buckets || d.buckets);
  // The API returns ev_buckets as a dict {label: stats}; normalize to an array.
  if (buckets && !Array.isArray(buckets) && typeof buckets === "object") {
    buckets = Object.entries(buckets).map(([label, b]) => ({ label, ...b }));
  }
  if (!d || d.verdict === "ERROR" || !Array.isArray(buckets) || !buckets.length) { placeholder("ev-bars", d); return; }
  // Dual axis: win rate as bars (left), profit factor as a line (right).
  const rows = buckets.map(b => ({
    label: String(b.label ?? b.bucket ?? b.range ?? ""),
    win_rate: Number(b.win_rate ?? b.winrate),
    profit_factor: Number(b.profit_factor ?? b.pf),
  }));
  g2render("ev-bars", {
    type: "view",
    autoFit: true,
    data: rows,
    children: [
      {
        type: "interval", encode: { x: "label", y: "win_rate" },
        scale: { y: { domain: [0, 1] } },
        axis: { y: { title: "win rate" } },
        style: { fill: C.accent },
      },
      {
        type: "line", encode: { x: "label", y: "profit_factor" },
        scale: { y: { independent: true } },
        axis: { y: { position: "right", title: "PF" } },
        style: { stroke: C.green, lineWidth: 2 },
      },
      {
        type: "point", encode: { x: "label", y: "profit_factor" },
        scale: { y: { independent: true } },
        style: { fill: C.green },
      },
    ],
  }, 280);
}

// ---- Tables ---------------------------------------------------------------
const sortState = {}; // elId -> { idx, dir }  (dir: 1 asc, -1 desc)

function cmpVals(a, b) {
  const an = (a === "" || a === undefined) ? null : a;
  const bn = (b === "" || b === undefined) ? null : b;
  if (an == null && bn == null) return 0;
  if (an == null) return 1;   // blanks sort last
  if (bn == null) return -1;
  if (typeof an === "number" && typeof bn === "number") return an - bn;
  return String(an).localeCompare(String(bn));
}

// columns: [{ label, num, get:r=>html, sort?:r=>rawValue }]
// opts.sortable -> clickable headers toggle asc/desc; sort key is `sort` (or `get`).
function renderTable(elId, columns, rows, d, opts = {}) {
  const el = document.getElementById(elId);
  if (!rows || !rows.length) { el.innerHTML = `<div style="padding:20px 0">${badge(d)}</div>`; return; }
  const sortable = !!opts.sortable;
  let display = rows;
  const st = sortState[elId];
  if (sortable && st && columns[st.idx]) {
    const col = columns[st.idx];
    const acc = col.sort || col.get;
    display = rows.slice().sort((a, b) => cmpVals(acc(a), acc(b)) * st.dir);
  }
  const head = "<tr>" + columns.map((c, i) => {
    const active = sortable && st && st.idx === i;
    const arrow = active ? (st.dir > 0 ? " ▲" : " ▼") : "";
    const cls = sortable ? ' class="sortable"' : "";
    const attr = sortable ? ` data-col="${i}"` : "";
    return `<th${cls}${attr}>${escapeHtml(c.label)}${arrow}</th>`;
  }).join("") + "</tr>";
  const body = display.map(r => "<tr>" + columns.map(c => {
    const v = c.get(r);
    const cls = c.num ? "num" : "";
    return `<td class="${cls}">${v}</td>`;
  }).join("") + "</tr>").join("");
  el.innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  if (sortable) {
    el.querySelectorAll("th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const idx = Number(th.getAttribute("data-col"));
        const prev = sortState[elId];
        sortState[elId] = (prev && prev.idx === idx)
          ? { idx, dir: -prev.dir } : { idx, dir: 1 };
        renderTable(elId, columns, rows, d, opts);
      });
    });
  }
}

async function loadRegimePerf() {
  const d = await api("regime-performance");
  renderTable("regperf-table", [
    { label: "Regime", get: r => escapeHtml(String(r.regime ?? r.label ?? "—")) },
    { label: "Trades", num: true, get: r => r.trades ?? r.n ?? "—" },
    { label: "Win rate", num: true, get: r => fmtPct(r.win_rate) },
    { label: "Avg P/L", num: true, get: r => signedCell(r.average_pnl ?? r.avg_pnl) },
  ], isUsable(d) ? d.regimes : null, d);
}

async function loadHypotheses() {
  const d = await api("hypotheses");
  const rows = (d && Array.isArray(d.hypotheses)) ? d.hypotheses : null;
  renderTable("hyp-table", [
    { label: "Hypothesis", get: r => escapeHtml(String(r.hypothesis_name ?? "—")) },
    { label: "Conclusion", get: r => conclusionBadge(r.conclusion) },
    { label: "Confidence", num: true, get: r => fmtNum(r.confidence, 2) },
    { label: "WR A/B", num: true, get: r => `${fmtPct(r.win_rate_a, 0)} / ${fmtPct(r.win_rate_b, 0)}` },
    { label: "Effect", num: true, get: r => fmtNum(r.effect_size, 2) },
  ], rows && rows.length ? rows : null, d);
}

async function loadPositions() {
  const d = await api("single-leg/positions");
  const rows = (d && Array.isArray(d.positions)) ? d.positions : null;
  _grOpen = d;
  renderGreenRed();
  const num = v => (v == null || isNaN(v)) ? null : Number(v);
  renderTable("pos-table", [
    { label: "Symbol", get: r => escapeHtml(String(r.symbol ?? "—")), sort: r => r.symbol },
    { label: "Underlying", get: r => escapeHtml(String(r.underlying ?? "—")), sort: r => r.underlying },
    { label: "Qty", num: true, get: r => r.quantity ?? "—", sort: r => num(r.quantity) },
    { label: "Entry", num: true, get: r => fmtNum(r.entry_price, 2), sort: r => num(r.entry_price) },
    { label: "Current", num: true, get: r => fmtNum(r.current_price, 2), sort: r => num(r.current_price) },
    { label: "P/L $", num: true, get: r => signedCell(r.unrealized_pl), sort: r => num(r.unrealized_pl) },
    { label: "P/L %", num: true, get: r => signedPctCell(r.unrealized_plpc), sort: r => num(r.unrealized_plpc) },
    { label: "Opened", get: r => escapeHtml(String(r.entry_time ?? "—")), sort: r => r.entry_time },
    { label: "EV", num: true, get: r => fmtNum(r.expected_value, 2), sort: r => num(r.expected_value) },
    { label: "PoP", num: true, get: r => r.probability_of_profit != null ? fmtPct(r.probability_of_profit, 0) : "—", sort: r => num(r.probability_of_profit) },
  ], rows && rows.length ? rows : null, d, { sortable: true });
}

// ---- Explain --------------------------------------------------------------
async function explainTicker(t) {
  const out = document.getElementById("explain-out");
  const clean = String(t || "").trim().toUpperCase();
  if (!/^[A-Z.]{1,8}$/.test(clean)) { out.innerHTML = '<span class="badge error">invalid ticker</span>'; return; }
  out.innerHTML = '<span class="muted">…</span>';
  const d = await api("explain/" + encodeURIComponent(clean));
  if (!isUsable(d)) { out.innerHTML = badge(d); return; }
  const p = d.probability || {};
  const pill = (k, v) => `<div class="prob-pill"><div class="v">${fmtPct(v, 0)}</div><div class="k">${k}</div></div>`;
  const votes = Array.isArray(d.votes) ? d.votes : [];
  const voteRows = votes.map(v =>
    `<tr><td>${escapeHtml(String(v.agent ?? v.name ?? "—"))}</td>` +
    `<td class="num">${fmtNum(v.bullish_score ?? v.bull, 2)}</td>` +
    `<td class="num">${fmtNum(v.bearish_score ?? v.bear, 2)}</td>` +
    `<td class="num">${fmtNum(v.confidence, 2)}</td></tr>`).join("");
  // `explanation` is an object {summary_str, top_reasons, ...}, not a string.
  const ex = (d.explanation && typeof d.explanation === "object") ? d.explanation : {};
  const summary = ex.summary_str || d.summary_str || "";
  const reasons = Array.isArray(ex.top_reasons) ? ex.top_reasons : [];
  out.innerHTML =
    `<div class="prob-row">${pill("P(call)", p.call ?? p.p_call)}${pill("P(put)", p.put ?? p.p_put)}${pill("P(no-trade)", p.no_trade ?? p.p_no_trade)}</div>` +
    (voteRows ? `<div class="tablewrap"><table><thead><tr><th>Agent</th><th>Bull</th><th>Bear</th><th>Conf</th></tr></thead><tbody>${voteRows}</tbody></table></div>` : "") +
    (summary ? `<p class="muted">${escapeHtml(String(summary))}</p>` : "") +
    (reasons.length ? `<ul class="muted">${reasons.map(r => `<li>${escapeHtml(String(r))}</li>`).join("")}</ul>` : "");
}

// ---- Green vs Red tile ----------------------------------------------------
// Dollar sums of profit (green) vs loss (red), split across two KPI tiles:
// OPEN by live unrealized P/L (from /single-leg/positions) and CLOSED by
// realized P/L (from /single-leg/kpis). Each tile shows the dollar pair on the
// value row and each side's share of the combined magnitude on the sub row.
// The two sources refresh independently, so each loader caches its slice and
// re-renders both tiles.
let _grOpen = null;    // last /single-leg/positions payload
let _grClosed = null;  // last /single-leg/kpis payload

function grPair(green, red) {
  return `<span class="pos">${fmtMoney(green)}</span>` +
    ` <span class="muted">/</span> ` +
    `<span class="neg">${fmtMoney(red)}</span>`;
}

// Green/red share of the combined |$| magnitude; null when there's nothing.
function grShare(green, red) {
  const g = Math.abs(green || 0), r = Math.abs(red || 0);
  const tot = g + r;
  if (tot <= 0) return null;
  return { g: g / tot, r: r / tot };
}

function grPct(green, red) {
  const sh = grShare(green, red);
  if (!sh) return `<span class="muted">—</span>`;
  return `<span class="pos">${(sh.g * 100).toFixed(0)}%</span>` +
    ` <span class="muted">/</span> ` +
    `<span class="neg">${(sh.r * 100).toFixed(0)}%</span>`;
}

function renderGrTile(valId, subId, green, red, ready, fallback) {
  const val = document.getElementById(valId);
  if (!val) return;
  val.classList.remove("pos", "neg");
  const sub = document.getElementById(subId);
  if (ready && green != null && red != null) {
    val.innerHTML = grPair(green, red);
    if (sub) sub.innerHTML = grPct(green, red);
  } else {
    val.innerHTML = fallback;
    if (sub) sub.innerHTML = "";
  }
}

function renderGreenRed() {
  // OPEN: live unrealized P/L. No broker marks -> say so explicitly.
  const o = _grOpen;
  renderGrTile(
    "kpi-greenred-open", "kpi-greenred-open-sub",
    o ? o.green_sum : null, o ? o.red_sum : null,
    isUsable(o) && o.marks_available,
    (o && o.marks_available === false) ? `<span class="muted">no live marks</span>` : "—");

  // CLOSED: realized P/L.
  const c = _grClosed;
  renderGrTile(
    "kpi-greenred-closed", "kpi-greenred-closed-sub",
    c ? c.closed_green_sum : null, c ? c.closed_red_sum : null,
    isUsable(c), "—");
}

// ---- misc render helpers --------------------------------------------------
function signedCell(v) {
  if (v == null || isNaN(v)) return "—";
  const cls = v > 0 ? "pos" : (v < 0 ? "neg" : "");
  return `<span class="${cls}">${fmtMoney(v)}</span>`;
}
function signedPctCell(v) {
  if (v == null || isNaN(v)) return "—";
  const cls = v > 0 ? "pos" : (v < 0 ? "neg" : "");
  return `<span class="${cls}">${fmtPct(v, 1)}</span>`;
}
function conclusionBadge(c) {
  const s = String(c || "").toUpperCase();
  let cls = "insufficient";
  if (s.includes("CONFIRM") || s.includes("SUPPORT")) cls = "ok";
  else if (s.includes("REJECT") || s.includes("REFUTE")) cls = "error";
  return `<span class="badge ${cls}">${escapeHtml(s || "—")}</span>`;
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- orchestration --------------------------------------------------------
async function refreshAll() {
  const status = document.getElementById("status");
  status.textContent = "refreshing…"; status.className = "status";
  try {
    await Promise.all([
      loadKpis(), loadRegime(), loadSentiment(), loadProbability(), loadAgents(),
      loadFeatures(), loadWeights(), loadEv(), loadRegimePerf(),
      loadHypotheses(), loadEpisodes(), loadPositions(),
    ]);
    status.textContent = "live"; status.className = "status ok";
  } catch (e) {
    status.textContent = "error"; status.className = "status err";
  }
  document.getElementById("updated").textContent =
    "updated " + new Date().toLocaleTimeString();
}

function setupAutoRefresh() {
  const box = document.getElementById("autorefresh");
  function arm() {
    if (refreshTimer) clearInterval(refreshTimer);
    if (box.checked) refreshTimer = setInterval(refreshAll, REFRESH_MS);
  }
  box.addEventListener("change", arm);
  arm();
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refresh").addEventListener("click", refreshAll);
  document.getElementById("explain-form").addEventListener("submit", e => {
    e.preventDefault();
    explainTicker(document.getElementById("explain-input").value);
  });
  setupAutoRefresh();
  refreshAll();
});
