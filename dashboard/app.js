"use strict";

// Oracle Dashboard — buildless frontend. Fetches the read-only JSON API and
// renders KPIs + Plotly widgets. Every widget degrades gracefully when its
// source returns verdict INSUFFICIENT_DATA / ERROR. No mutation, GET-only.

const REFRESH_MS = 60000;
const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#8b949e", size: 12 },
  margin: { l: 50, r: 16, t: 16, b: 40 },
  xaxis: { gridcolor: "#2a3441", zerolinecolor: "#2a3441" },
  yaxis: { gridcolor: "#2a3441", zerolinecolor: "#2a3441" },
};
const PLOT_CFG = { displayModeBar: false, responsive: true };
const C = { accent: "#58a6ff", green: "#3fb950", red: "#f85149", amber: "#d29922" };

let refreshTimer = null;

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
  Plotly.react("episodes-bars", [
    { x: labels, y: labels.map(k => counts[k]), type: "bar", marker: { color: C.accent } },
  ], { ...PLOT_LAYOUT, height: 220, yaxis: { ...PLOT_LAYOUT.yaxis, title: "episodes" } }, PLOT_CFG);
  const s = d.stats || {};
  statsEl.innerHTML =
    `total <b>${s.total ?? 0}</b> · completed <b>${s.completed ?? 0}</b> · ` +
    `win rate <b>${fmtPct(s.win_rate)}</b> · mean net <b>${fmtNum(s.mean_net_pnl_pct, 2)}%</b>`;
}

// ---- Regime ---------------------------------------------------------------
async function loadRegime() {
  const d = await api("regime");
  const reasonsEl = document.getElementById("regime-reasons");
  if (!isUsable(d)) { placeholder("regime-gauge", d); reasonsEl.innerHTML = ""; return; }
  const conf = Number(d.confidence || 0);
  Plotly.react("regime-gauge", [{
    type: "indicator", mode: "gauge+number",
    value: +(conf * 100).toFixed(0),
    number: { suffix: "%", font: { color: "#e6edf3" } },
    title: { text: d.label || "—", font: { color: "#e6edf3", size: 16 } },
    gauge: {
      axis: { range: [0, 100], tickcolor: "#8b949e" },
      bar: { color: C.accent },
      bgcolor: "#1c2330", borderwidth: 0,
    },
  }], { ...PLOT_LAYOUT, height: 220 }, PLOT_CFG);
  const reasons = Array.isArray(d.reasons) ? d.reasons : [];
  reasonsEl.innerHTML = reasons.length
    ? "<ul>" + reasons.map(r => `<li>${escapeHtml(String(r))}</li>`).join("") + "</ul>"
    : "";
}

// ---- Probability calibration (reliability curve from /calibration/pop) ----
async function loadProbability() {
  const [prob, pop] = await Promise.all([api("probability"), api("calibration/pop")]);
  const statsEl = document.getElementById("prob-stats");
  if (isUsable(pop) && Array.isArray(pop.buckets) && pop.buckets.length) {
    const xs = pop.buckets.map(b => b.predicted ?? b.mid ?? b.p ?? b.bucket);
    const ys = pop.buckets.map(b => b.realized ?? b.actual ?? b.win_rate);
    Plotly.react("prob-curve", [
      { x: [0, 1], y: [0, 1], mode: "lines", line: { dash: "dot", color: "#8b949e" }, name: "ideal" },
      { x: xs, y: ys, mode: "lines+markers", line: { color: C.accent }, name: "observed" },
    ], { ...PLOT_LAYOUT, height: 240, xaxis: { ...PLOT_LAYOUT.xaxis, title: "predicted", range: [0, 1] }, yaxis: { ...PLOT_LAYOUT.yaxis, title: "realized", range: [0, 1] }, showlegend: false }, PLOT_CFG);
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
  const names = d.agents.map(a => a.agent);
  Plotly.react("agents-bars", [
    { x: d.agents.map(a => a.hit_rate), y: names, type: "bar", orientation: "h", marker: { color: C.accent }, name: "hit rate" },
  ], {
    ...PLOT_LAYOUT, height: Math.max(220, names.length * 34),
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "hit rate", range: [0, 1] },
    shapes: d.base_win_rate != null ? [{
      type: "line", x0: d.base_win_rate, x1: d.base_win_rate, y0: -0.5, y1: names.length - 0.5,
      line: { color: C.amber, dash: "dash", width: 1 },
    }] : [],
  }, PLOT_CFG);
}

// ---- Feature importance ---------------------------------------------------
async function loadFeatures() {
  const d = await api("feature-importance");
  if (!isUsable(d) || !Array.isArray(d.features) || !d.features.length) { placeholder("features-bars", d); return; }
  Plotly.react("features-bars", [
    { x: d.features.map(f => f.importance), y: d.features.map(f => f.agent), type: "bar", orientation: "h", marker: { color: C.green } },
  ], { ...PLOT_LAYOUT, height: Math.max(220, d.features.length * 34), xaxis: { ...PLOT_LAYOUT.xaxis, title: "mean contribution" } }, PLOT_CFG);
}

// ---- Weights --------------------------------------------------------------
async function loadWeights() {
  const d = await api("weights");
  const driftEl = document.getElementById("weights-drift");
  const cur = (d && d.current) || {};
  const keys = Object.keys(cur);
  if (!isUsable(d) || !keys.length) { placeholder("weights-bars", d); driftEl.innerHTML = ""; return; }
  Plotly.react("weights-bars", [
    { x: keys, y: keys.map(k => cur[k]), type: "bar", marker: { color: C.accent } },
  ], { ...PLOT_LAYOUT, height: 240, yaxis: { ...PLOT_LAYOUT.yaxis, title: "weight" } }, PLOT_CFG);
  driftEl.innerHTML = `snapshots <b>${d.snapshots ?? 0}</b> · drift <b>${fmtNum(d.drift, 3)}</b>`;
}

// ---- EV attribution -------------------------------------------------------
async function loadEv() {
  const d = await api("ev-attribution");
  const buckets = d && (d.ev_buckets || d.buckets);
  if (!d || d.verdict === "ERROR" || !Array.isArray(buckets) || !buckets.length) { placeholder("ev-bars", d); return; }
  const labels = buckets.map(b => b.label ?? b.bucket ?? b.range ?? "");
  const wr = buckets.map(b => b.win_rate ?? b.winrate);
  const pf = buckets.map(b => b.profit_factor ?? b.pf);
  Plotly.react("ev-bars", [
    { x: labels, y: wr, type: "bar", name: "win rate", marker: { color: C.accent } },
    { x: labels, y: pf, type: "bar", name: "profit factor", marker: { color: C.green }, yaxis: "y2" },
  ], {
    ...PLOT_LAYOUT, height: 260, barmode: "group",
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "win rate" },
    yaxis2: { overlaying: "y", side: "right", gridcolor: "transparent", title: "PF" },
    legend: { orientation: "h", y: 1.15 },
  }, PLOT_CFG);
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
    { label: "Avg P/L", num: true, get: r => signedCell(r.avg_pnl) },
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
  out.innerHTML =
    `<div class="prob-row">${pill("P(call)", p.call ?? p.p_call)}${pill("P(put)", p.put ?? p.p_put)}${pill("P(no-trade)", p.no_trade ?? p.p_no_trade)}</div>` +
    (voteRows ? `<div class="tablewrap"><table><thead><tr><th>Agent</th><th>Bull</th><th>Bear</th><th>Conf</th></tr></thead><tbody>${voteRows}</tbody></table></div>` : "") +
    (d.explanation ? `<p class="muted">${escapeHtml(String(d.explanation))}</p>` : "");
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
      loadKpis(), loadRegime(), loadProbability(), loadAgents(),
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
