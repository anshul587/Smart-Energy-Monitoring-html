firebase.initializeApp(firebaseConfig);

const maxPower = 3000;
const colors = ["#2578ff", "#7b4cf6", "#f28d2f", "#13b887", "#e45f92", "#16a7d9", "#a269d8", "#d88323", "#2c9e72"];
const $ = (id) => document.getElementById(id);

const dashboard = $("dashboardContent");
const meterTemplate = $("meterTemplate");
const unitRate = $("unitRate");
let metersData = {};

// Debug flag - set to true in console to enable verbose logging: localStorage.debug = 'true'
const DEBUG = localStorage.getItem("debug") === "true";
let powerHistoryMode = false;
let historyRequestId = 0;
let selectedPzem = "all"; // "all" or 1-9

/* =========================================================================
   PZEM FRESHNESS (LIVE vs OFF)
   -------------------------------------------------------------------------
   The firmware (SmartEnergyMonitor.ino, uploadLive()) PATCHes meters/pzem_N
   only when it has a currently-valid PZEM reading, once every
   LIVE_UPLOAD_INTERVAL_MS (config.h = 10000 ms). If a PZEM stops responding,
   the firmware simply stops writing for that meter — it does NOT delete or
   blank the old object. So meters/pzem_N (with its last voltage/current/
   power/energy and old timestamp/lastSeen) stays in Firebase forever,
   completely unchanged. That's the exact root cause of stale data reading
   as "live": existence of meters/pzem_N was previously being used as the
   online signal, but existence never expires — only "timestamp"/"lastSeen"
   (written by the firmware in Unix seconds, see readingJson() in the .ino)
   tells you when the meter actually last reported.

   Freshness timeout: 3x the firmware's real 10 s live-upload interval = 30 s.
   That's long enough to absorb one or two missed upload cycles (a brief
   Wi-Fi/auth hiccup) without the card flapping LIVE/OFF/LIVE, but short
   enough that a meter which has actually stopped reporting reads as OFF
   within half a minute — not an arbitrary long window.
   ========================================================================= */
const LIVE_UPLOAD_INTERVAL_MS = 10000;              // must match config.h LIVE_UPLOAD_INTERVAL_MS
const FRESHNESS_TIMEOUT_MS = LIVE_UPLOAD_INTERVAL_MS * 3; // 30 s
const HISTORY_SLOT_MS = 5 * 60 * 1000;              // must match config.h HISTORY_SLOT_SECONDS (300 s)

/* How old (ms) a meter's last reported reading is, or null if it has never
   reported a timestamp/lastSeen at all. */
function meterAgeMs(meter) {
  const lastSeenRaw = meter.lastSeen ?? meter.timestamp;
  if (lastSeenRaw === undefined || lastSeenRaw === null || lastSeenRaw === "") return null;
  const lastSeenMs = timestampMilliseconds(lastSeenRaw);
  return Number.isFinite(lastSeenMs) ? Date.now() - lastSeenMs : null;
}

/* THE single authoritative LIVE/OFF calculation for a meter. Every place in
   this file that needs to know whether a PZEM's CURRENT reading can be
   trusted — cards, summary totals, the popup, the live charts — calls this
   same function. This is about meter connectivity/freshness only; it has
   nothing to do with what equipment a PZEM is wired to or how much power it
   is drawing. */
function isMeterFresh(meter) {
  const age = meterAgeMs(meter);
  return age !== null && age <= FRESHNESS_TIMEOUT_MS;
}

/* =========================================================================
   TWO-STATUS CARD LOGIC (Communication / AC Supply)
   -------------------------------------------------------------------------
   Communication reuses isMeterFresh() above — already the single
   authoritative, timeout-based LIVE/OFF signal used everywhere else in this
   file (cards, summary totals, charts). It is not re-derived from a raw
   "communicationOnline" flag: a boolean written once by the firmware would
   itself go stale the moment a PZEM actually drops off, which is exactly
   the bug the timestamp/lastSeen freshness check was built to avoid.

   AC supply is read from meter.acSupplyOn — the field name given in the
   brief. It's read independently of Communication: CONNECTED + AC OFF is a
   normal, valid combination and must never be shown as OFFLINE. Only when
   Communication itself is OFFLINE does AC fall back to UNKNOWN, since a
   stale reading's last AC value can no longer be trusted as current.

   (A separate "Load" status/meter.loadOn was previously also shown here but
   has been removed from the UI — AC Supply alone is sufficient for the
   dashboard's ON/OFF indication. This does not touch meter.loadOn in
   Firebase; the field, if the firmware writes it, is simply no longer read
   or displayed here.)

   If a PZEM's Firebase object simply doesn't have acSupplyOn yet (firmware
   not sending it), tristate() returns null and the card shows UNKNOWN
   rather than guessing true/false. */
function tristate(value) {
  if (value === true || value === 1 || value === "1" || value === "true") return true;
  if (value === false || value === 0 || value === "0" || value === "false") return false;
  return null; // undefined/null/"" or anything else unrecognized
}

/* Returns { communication: bool, ac: true|false|null } */
function getThreeStatus(meter) {
  const communication = isMeterFresh(meter);
  if (!communication) return { communication, ac: null };
  return { communication, ac: tristate(meter.acSupplyOn) };
}

// Cache of the last computed live-summary values, written by renderDashboard()
// and read by updateFrequency() (which always runs immediately afterward) to
// print one combined [LIVE SUMMARY] diagnostic line per update.
let lastLiveSummary = { freshMeters: 0, power: 0, voltage: 0, frequency: 0 };

function number(value, decimals = 0) {
  return Number(value || 0).toFixed(decimals);
}

function inr(value) {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(value);
}

/* Converts a Stage 9 forecast power series (W, one value per 5-min slot) into
   forecasted energy (kWh): W -> kW -> kWh. Drops non-finite values, clamps
   negative power to 0. Never returns NaN/negative. */
function forecastEnergyFromPower(powers) {
  if (!Array.isArray(powers)) return 0;
  const SLOT_HOURS = 5 / 60;  // history slot = 5 minutes
  let kwh = 0;
  for (const p of powers) {
    const v = Number(p);
    if (!Number.isFinite(v)) continue;
    kwh += (Math.max(0, v) / 1000) * SLOT_HOURS;
  }
  return Number.isFinite(kwh) && kwh > 0 ? kwh : 0;
}

function getMeter(number) {
  return metersData[`pzem_${number}`] || metersData[`pzem${number}`] || {};
}

function createDatasets() {
  return Array.from({ length: 9 }, (_, index) => ({
    label: `PZEM ${index + 1}`,
    data: [],
    borderColor: colors[index],
    backgroundColor: colors[index],
    borderWidth: 2,
    tension: 0.35,
    pointRadius: 0,
    pointHoverRadius: 4,
    spanGaps: false /* a stale/OFF meter pushes null — show a real break, not a bridged line */
  }));
}

/* Add common frequency summary card and power range/PZEM selectors */
function createMonitoringUi() {
  const summaryCardLayout = document.createElement("style");

summaryCardLayout.textContent = `
  .summary-grid {
    grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
  }

  @media (max-width: 1180px) {
    .summary-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    }
  }

  @media (max-width: 760px) {
    .summary-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
  }

  @media (max-width: 500px) {
    .summary-grid {
      grid-template-columns: 1fr !important;
    }
  }
`;

document.head.appendChild(summaryCardLayout);
  const card = document.createElement("article");
  card.className = "summary-card";
  card.innerHTML = `
    <span class="summary-label">Common frequency</span>
    <strong id="commonFrequency">0.00 <small>Hz</small></strong>
    <span class="summary-caption" id="frequencyCaption">Waiting for live meter data</span>
  `;
  document.querySelector(".summary-grid").appendChild(card);

  const powerHeading = document.querySelector(".chart-panel .chart-heading");
  const oldNote = powerHeading.querySelector(".chart-note");

  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;align-items:center;gap:8px;flex-wrap:wrap;";
  controls.innerHTML = `
    <label style="font-size:11px;font-weight:700;color:var(--muted)">Power data</label>
    <select id="powerRange" style="padding:7px 9px;border:1px solid var(--line);border-radius:9px;background:var(--surface);color:var(--ink);font-weight:700">
      <option value="1h">1 Hour</option>
      <option value="6h">6 Hours</option>
      <option value="12h">12 Hours</option>
      <option value="today">Today</option>
      <option value="yesterday">Yesterday</option>
      <option value="7d" selected>7 Days</option>
      <option value="30d">30 Days</option>
    </select>
    <label style="font-size:11px;font-weight:700;color:var(--muted)">PZEM</label>
    <select id="pzemSelect" style="padding:7px 9px;border:1px solid var(--line);border-radius:9px;background:var(--surface);color:var(--ink);font-weight:700">
      <option value="all">All PZEMs</option>
      <option value="1">PZEM 1</option>
      <option value="2">PZEM 2</option>
      <option value="3">PZEM 3</option>
      <option value="4">PZEM 4</option>
      <option value="5">PZEM 5</option>
      <option value="6">PZEM 6</option>
      <option value="7">PZEM 7</option>
      <option value="8">PZEM 8</option>
      <option value="9">PZEM 9</option>
    </select>
  `;

  if (oldNote) controls.appendChild(oldNote);
  powerHeading.appendChild(controls);
}

createMonitoringUi();

/* Rounds a positive value up to a "nice" round number (1/1.2/1.5/2/2.5/3/4/
   5/6/8/10 x a power of ten) — the standard approach for auto-scaling chart
   axes so ticks land on clean numbers instead of arbitrary fractions. */
function niceNumber(value) {
  if (!Number.isFinite(value) || value <= 0) return 0;
  const exponent = Math.floor(Math.log10(value));
  const magnitude = Math.pow(10, exponent);
  const fraction = value / magnitude;
  const niceFractions = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
  const niceFraction = niceFractions.find((candidate) => candidate >= fraction) ?? 10;
  return niceFraction * magnitude;
}

/* Fixed Y-axis buckets so small loads (2 W, 5 W, 8 W, 10 W...) stay
   readable instead of being squashed near zero on a large fixed axis. The
   bucket ceiling is chosen from the highest value currently on the chart
   across ALL PZEM lines — this is still ONE shared axis for every dataset,
   never a per-line axis. */
function powerAxisMaxForHighest(highest) {
  if (highest <= 10) return 20;
  if (highest <= 50) return 100;
  if (highest <= 100) return 200;
  if (highest <= 250) return 500;
  if (highest <= 500) return 1000;
  if (highest <= 1000) return 2000;
  if (highest <= 2000) return 3000;
  return 5000; // covers >2000-4000 W and 4000 W+ (spec: both bucket to 0-5 kW)
}

/* Computes the power (W) axis max + tick step from the highest value
   currently on the chart, using the fixed buckets above. */
function computePowerAxisRange(highestPowerWatts) {
  const safeHighest = Number.isFinite(highestPowerWatts) && highestPowerWatts > 0 ? highestPowerWatts : 0;
  const max = powerAxisMaxForHighest(safeHighest);
  const step = niceNumber(max / 5) || max; // aim for ~5 divisions, rounded to a nice step
  return { max, step };
}

/* Formats a watts value for axis ticks/tooltips: plain W under 1000, kW
   (up to 1 decimal place) at or above 1000 — matches how the meter cards
   and modal already display power, just with the unit made explicit here. */
function formatPowerLabel(watts) {
  if (!Number.isFinite(watts)) return "0 W";
  if (watts >= 1000) {
    const kw = watts / 1000;
    const rounded = Math.round(kw * 10) / 10;
    return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)} kW`;
  }
  return `${Math.round(watts)} W`;
}

/* ======================= DYNAMIC Y-AXIS TICKS =======================
   Fixed label precision breaks under zoom: chartjs-plugin-zoom narrows the
   scale's min/max and Chart.js then emits finer ticks, but labels rounded to
   a fixed precision collapse onto identical strings ("100 W" x4, "50.0 Hz"
   x4). These helpers derive decimal precision from the scale's CURRENT
   visible window (this.min/this.max at tick-draw time), so labels stay
   unique and gain decimals as the window narrows. Data is never modified. */
function visibleDecimals(min, max, targetTicks = 6, cap = 4) {
  const range = Number(max) - Number(min);
  if (!Number.isFinite(range) || range <= 0) return 0;
  const approxStep = range / targetTicks;
  if (!(approxStep > 0)) return 0;
  return Math.min(cap, Math.max(0, Math.ceil(-Math.log10(approxStep))));
}

/* Decimals needed so every RENDERED tick gets its own label. Derived from
   the real gap between neighbouring ticks (not just window/range), because
   Chart.js can emit more ticks than the nominal budget and includes
   unaligned window-edge ticks — both caused duplicate labels before. */
function tickDecimalsForScale(scale, index, ticks) {
  const min = Number(scale && scale.min);
  const max = Number(scale && scale.max);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return 0;
  let step = null;
  if (Array.isArray(ticks) && ticks.length > 1) {
    const i = Math.min(Math.max(index, 1), ticks.length - 1);
    const a = Number(ticks[i] && ticks[i].value);
    const b = Number(ticks[i - 1] && ticks[i - 1].value);
    if (Number.isFinite(a) && Number.isFinite(b)) step = Math.abs(a - b);
  }
  if (!step || !Number.isFinite(step) || step <= 0) {
    return visibleDecimals(min, max);
  }
  return Math.min(4, Math.max(0, Math.ceil(-Math.log10(step))));
}

function powerAxisTickLabel(value, index, ticks, scale) {
  const v = Number(value);
  if (!Number.isFinite(v)) return "";
  const decimals = tickDecimalsForScale(scale, index, ticks);
  if (decimals === 0 && Math.abs(Number(scale && scale.max)) >= 1000) {
    return formatPowerLabel(v); // wide/unzoomed: legacy "2 kW"/"500 W" style
  }
  if (Math.abs(Number(scale && scale.max)) >= 1000) {
    return `${(v / 1000).toFixed(Math.min(decimals + 1, 3))} kW`;
  }
  return `${v.toFixed(decimals)} W`;
}

function frequencyAxisTickLabel(value, index, ticks, scale) {
  const v = Number(value);
  if (!Number.isFinite(v)) return "";
  return `${v.toFixed(tickDecimalsForScale(scale, index, ticks))} Hz`;
}

/* Once the user zooms/pans, the visible window belongs to them: periodic
   data updates must stop snapping the axis back to the full-data bucket,
   and fixed tick steps must get out of the way so Chart.js can lay ticks
   out for the zoomed window instead. chartjs-plugin-zoom's completion
   callbacks proved unreliable to bind to, so zoom state is detected
   directly: the plugin expresses zoom by rewriting scale.options.min/max,
   and axisStepGuard/updatePowerYAxis compare those against the stock
   (unzoomed) window on every update. resetChartZoom() hands control back. */
const axisStepGuard = {
  id: "axisStepGuard",
  beforeUpdate(chart) {
    const y = chart.options.scales && chart.options.scales.y;
    if (!y || !y.stepGuard) return; // opt-in: only axes with a fixed stock step
    const g = y.stepGuard;
    const t = y.ticks;
    const zoomed =
      Number(y.min) !== Number(g.min) ||
      Number(y.max) !== Number(g.max);
    if (zoomed) {
      if (t.stepSize != null) {
        g.stock = t.stepSize;
        delete t.stepSize; // a fixed step is wrong for an arbitrary zoomed window
      }
    } else if (t.stepSize == null && g.stock != null) {
      t.stepSize = g.stock;
    }
  }
};

function resetChartZoom(chart) {
  if (!chart || typeof chart.resetZoom !== "function") return;
  chart.resetZoom();
  chart.$axisJustReset = true; // lets updatePowerYAxis() reclaim the axis
  chart.update("none");
}

/* ================= SUBTLE 3D GRAPH DEPTH (visual only) =================
   One shared inline plugin applied to every existing line chart. It draws
   purely decorative depth INSIDE the canvas chartArea — a faint top-down
   light wash (back wall), a one-point-perspective floor grid in the lower
   band, and a soft drop shadow under each dataset line/points (extrusion).
   No data, scales, tooltips, zoom/pan or layout behavior is touched; the
   real Chart.js line remains the only data ink. Theme colors come from the
   --g3d-* CSS variables so light/dark both work with no JS theme handling. */
function hexToRgba(hex, alpha) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(String(hex || "").trim());
  if (!m) return `rgba(128, 128, 128, ${alpha})`;
  return `rgba(${parseInt(m[1], 16)}, ${parseInt(m[2], 16)}, ${parseInt(m[3], 16)}, ${alpha})`;
}

function g3dVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

const graphDepth = {
  id: "graphDepth",
  beforeDatasetsDraw(chart) {
    const area = chart.chartArea;
    const ctx = chart.ctx;
    if (!area || area.width < 80 || area.height < 60) return;

    /* Back wall: faint light wash from the top (subtle lighting) */
    const wallHi = g3dVar("--g3d-wall");
    if (wallHi) {
      const wall = ctx.createLinearGradient(0, area.top, 0, area.bottom);
      wall.addColorStop(0, wallHi);
      wall.addColorStop(0.35, "rgba(0,0,0,0)");
      ctx.fillStyle = wall;
      ctx.fillRect(area.left, area.top, area.width, area.height);
    }

    /* Perspective floor: receding slats + converging rails in the bottom
       band, clipped to the plot area so ticks/labels stay untouched */
    const floorInk = g3dVar("--g3d-floor");
    if (!floorInk) return;
    const bandH = Math.min(area.height * 0.24, 64);
    const horizon = area.bottom - bandH;
    const cx = area.left + area.width / 2;
    ctx.save();
    ctx.beginPath();
    ctx.rect(area.left, horizon, area.width, bandH);
    ctx.clip();
    ctx.strokeStyle = floorInk;
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      const t = Math.pow(i / 5, 1.7); // slats bunch up toward the horizon
      ctx.globalAlpha = 0.55 + 0.45 * (i / 5);
      ctx.beginPath();
      ctx.moveTo(area.left, area.bottom - bandH * t);
      ctx.lineTo(area.right, area.bottom - bandH * t);
      ctx.stroke();
    }
    ctx.globalAlpha = 0.5;
    [-0.42, -0.14, 0.14, 0.42].forEach((off) => {
      ctx.beginPath();
      ctx.moveTo(cx + off * area.width, area.bottom);
      ctx.lineTo(cx + off * 0.16 * area.width, horizon);
      ctx.stroke();
    });
    ctx.restore();
  },
  /* Per-dataset soft drop shadow = depth under the plotted line + lifted
     look for visible points. save/restore always paired, even for hidden
     or empty datasets, so the canvas state can never leak. */
  beforeDatasetDraw(chart, args) {
    const ctx = chart.ctx;
    ctx.save();
    const dataset = chart.data.datasets[args.index];
    const meta = chart.getDatasetMeta(args.index);
    if (dataset && meta && !meta.hidden && dataset.data.length > 1) {
      ctx.shadowColor = hexToRgba(dataset.borderColor, 0.22);
      ctx.shadowBlur = 5;
      ctx.shadowOffsetX = 1.5;
      ctx.shadowOffsetY = 3;
    }
  },
  afterDatasetDraw(chart, args) {
    chart.ctx.restore();
  }
};

/* ================= 3D POWER BARS (visual only) =================
   Heights the card's mini bar viz from the SAME live watts number the
   card already displays — no new data sources, timers, or listeners.
   Height is a pure function of the real reading against the existing
   maxPower scale; OFFLINE/invalid readings get the grey .is-offline
   stub state and never animate. Emergency/warning coloring is left
   entirely to the existing card-level classes in CSS. */
function updatePowerViz(cardEl, watts, isLive) {
  const viz = cardEl && cardEl.querySelector(".power-viz");
  if (!viz) return;
  if (!isLive || !Number.isFinite(watts) || watts <= 0) {
    viz.classList.add("is-offline");
    return;
  }
  viz.classList.remove("is-offline");
  const pct = Math.max(10, Math.min(100, (watts / maxPower) * 100));
  for (const bar of viz.children) bar.style.height = pct.toFixed(1) + "%";
}



/* THE single shared path from a raw Firebase power value to a normalized
   WATTS number. Every place that reads power — live cards, the live power
   chart, the popup/history chart, peak/average calculations — goes through
   this one function, so there is exactly one W/kW conversion system, not
   two conflicting ones (Part 12).
   Accepts either a reading object ({ power: 1.2, ... }, the current
   firmware format) or a bare number (older/legacy rows that were never
   wrapped in an object) — mirrors the `data.power ?? data` fallback that
   was already in use, just centralized in one place. Returns 0 only when
   the value genuinely can't be parsed as a number (NaN) — a real stored
   value, however large, is passed through unchanged and never silently
   clamped or hidden (Part 11); callers that display peak/average add their
   own diagnostic logging for suspiciously large values instead. */
function normalizePowerWatts(source) {
  const raw = (source && typeof source === "object") ? source.power : source;
  const watts = Number(raw);
  return Number.isFinite(watts) ? watts : 0;
}

/* Get theme-aware chart colors from CSS variables */
function getChartColors() {
  const style = getComputedStyle(document.body);
  return {
    grid: style.getPropertyValue('--chart-grid').trim(),
    tick: style.getPropertyValue('--chart-tick').trim(),
    title: style.getPropertyValue('--chart-title').trim(),
    legend: style.getPropertyValue('--chart-legend').trim(),
    tooltipBg: style.getPropertyValue('--chart-tooltip-bg').trim(),
    tooltipText: style.getPropertyValue('--chart-tooltip-text').trim()
  };
}

/* Update all charts with current theme colors */
function updateChartThemeColors() {
  const colors = getChartColors();
  
  [powerChart, frequencyChart, overviewChart, modalHistoryChart, forecastChart].forEach(chart => {
    if (!chart) return;
    
    chart.options.scales.x.grid.color = colors.grid;
    chart.options.scales.x.ticks.color = colors.tick;
    chart.options.scales.x.title.color = colors.title;
    chart.options.scales.y.grid.color = colors.grid;
    chart.options.scales.y.ticks.color = colors.tick;
    chart.options.scales.y.title.color = colors.title;
    chart.options.plugins.legend.labels.color = colors.legend;
    chart.options.plugins.tooltip.backgroundColor = colors.tooltipBg;
    chart.options.plugins.tooltip.titleColor = colors.tooltipText;
    chart.options.plugins.tooltip.bodyColor = colors.tooltipText;
    chart.update('none');
  });
}

const chartColors = getChartColors();

const powerChart = new Chart($("powerChart"), {
  type: "line",
  plugins: [axisStepGuard, graphDepth],
  data: { labels: [], datasets: createDatasets() },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { 
        labels: { 
          boxWidth: 9, 
          usePointStyle: true, 
          pointStyle: "circle", 
          padding: 14, 
          color: chartColors.legend,
          font: { size: 11, weight: 500, family: "'DM Sans', sans-serif" }
        } 
      },
      tooltip: {
        backgroundColor: chartColors.tooltipBg,
        titleColor: chartColors.tooltipText,
        bodyColor: chartColors.tooltipText,
        titleFont: { size: 12, weight: 600, family: "'Space Grotesk', sans-serif" },
        bodyFont: { size: 11, family: "'DM Sans', sans-serif" },
        padding: 12,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: (context) => `${context.dataset.label}: ${formatPowerLabel(Number(context.parsed.y || 0))}`
        }
      },
      zoom: {
        pan: { enabled: true, mode: 'xy' },
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' },
        limits: { x: { min: 'original', max: 'original', minRange: 10 }, y: { min: 'original', max: 'original' } }
      }
    },
    scales: {
      x: { 
        grid: { display: false, color: chartColors.grid }, 
        ticks: { 
          maxTicksLimit: 8, 
          color: chartColors.tick,
          font: { size: 10, family: "'DM Sans', sans-serif" },
          maxRotation: 0,
          autoSkip: true,
          autoSkipPadding: 20
        } 
      },
      y: {
        beginAtZero: true,
        min: 0,
        max: 300,
        stepGuard: { min: 0, max: 300 }, // stock window; axisStepGuard watches for user zoom
        title: { display: true, text: "Power (W)", color: chartColors.title, font: { size: 11, weight: 600, family: "'DM Sans', sans-serif" } },
        ticks: { 
          stepSize: 60, 
          color: chartColors.tick,
          font: { size: 10, family: "'DM Sans', sans-serif" },
          callback(value, index, ticks) { return powerAxisTickLabel(value, index, ticks, this); },
          padding: 8
        },
        grid: { color: chartColors.grid, drawBorder: false }
      }
    }
  }
});

const frequencyChart = new Chart($("frequencyChart"), {
  type: "line",
  plugins: [axisStepGuard, graphDepth],
  data: {
    labels: [],
    datasets: [{
      label: "Common frequency",
      data: [],
      borderColor: "#1e6bd6",
      backgroundColor: "rgba(30, 107, 214, .12)",
      borderWidth: 3,
      tension: 0.35,
      pointRadius: 2,
      pointHoverRadius: 5,
      fill: true,
      spanGaps: true
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { 
        labels: { 
          boxWidth: 9, 
          usePointStyle: true, 
          pointStyle: "circle", 
          color: chartColors.legend,
          font: { size: 11, weight: 500, family: "'DM Sans', sans-serif" }
        } 
      },
      tooltip: {
        backgroundColor: chartColors.tooltipBg,
        titleColor: chartColors.tooltipText,
        bodyColor: chartColors.tooltipText,
        titleFont: { size: 12, weight: 600, family: "'Space Grotesk', sans-serif" },
        bodyFont: { size: 11, family: "'DM Sans', sans-serif" },
        padding: 12,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: (context) => `Common frequency: ${Number(context.parsed.y || 0).toFixed(2)} Hz`
        }
      },
      zoom: {
        pan: { enabled: true, mode: 'xy' },
        zoom: { wheel: { enabled: true, modifierKey: 'ctrl' }, pinch: { enabled: true }, mode: 'xy' },
        limits: { x: { min: 'original', max: 'original', minRange: 10 }, y: { min: 'original', max: 'original' } }
      }
    },
    scales: {
      x: { 
        grid: { display: false, color: chartColors.grid }, 
        ticks: { 
          maxTicksLimit: 8, 
          color: chartColors.tick,
          font: { size: 10, family: "'DM Sans', sans-serif" },
          maxRotation: 0,
          autoSkip: true,
          autoSkipPadding: 20
        } 
      },
      y: {
        min: 48,
        max: 52,
        stepGuard: { min: 48, max: 52 }, // stock window; axisStepGuard watches for user zoom
        title: { display: true, text: "Frequency (Hz)", color: chartColors.title, font: { size: 11, weight: 600, family: "'DM Sans', sans-serif" } },
        ticks: { 
          stepSize: 0.5, 
          color: chartColors.tick,
          font: { size: 10, family: "'DM Sans', sans-serif" },
          callback(value, index, ticks) { return frequencyAxisTickLabel(value, index, ticks, this); },
          padding: 8
        },
        grid: { color: chartColors.grid, drawBorder: false }
      }
    }
  }
});

/* Test/debug handle: lets DevTools and automated tests reach the chart
   instances without changing production behavior. Exists only when
   localStorage.debug === "true" (same gate as all [DEBUG] logging above). */
if (DEBUG) window.__charts = { powerChart, frequencyChart, helpers: { resetChartZoom } };

function updateFrequency() {
  // Part 5 fix: filter to FRESH meters only. The previous version accepted
  // any meter with frequency > 0 regardless of age, so a PZEM that stopped
  // reporting hours ago (but whose last frequency reading happened to be a
  // normal ~50 Hz) kept contributing to "Common frequency" forever — the
  // same stale-existence bug as the cards, just for this one widget.
  const freshValues = [];

  for (let i = 1; i <= 9; i++) {
    const meter = getMeter(i);
    if (!isMeterFresh(meter)) continue;
    const value = Number(meter.frequency);
    if (Number.isFinite(value)) freshValues.push(value);
  }

  // WHEN ZERO METERS ARE FRESH: do NOT append new data points, do NOT extend
  // the timeline, do NOT add zero or stale values. Preserve existing chart data.
  if (freshValues.length === 0) {
    // Only update the caption to reflect no fresh meters, but do not add
    // new points to the chart historical data.
    $("commonFrequency").innerHTML = `${number(0, 2)} <small>Hz</small>`;
    $("frequencyCaption").textContent = "No fresh meters — waiting for live data";
    return;  // <-- ADD: stop here, do not push new points
  }

  const common = freshValues.length
    ? freshValues.reduce((sum, value) => sum + value, 0) / freshValues.length
    : 0;

  $("commonFrequency").innerHTML = `${number(common, 2)} <small>Hz</small>`;
  $("frequencyCaption").textContent = freshValues.length
    ? `${freshValues.length} live meters · ${Math.min(...freshValues).toFixed(2)}–${Math.max(...freshValues).toFixed(2)} Hz`
    : "No fresh meters — waiting for live data";

  const label = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });

  // WHEN ONE OR MORE METERS ARE FRESH: append new point only
  frequencyChart.data.labels.push(label);
  frequencyChart.data.datasets[0].data.push(common);

  if (frequencyChart.data.labels.length > 100) {
    frequencyChart.data.labels.shift();
    frequencyChart.data.datasets[0].data.shift();
  }

  frequencyChart.update();
}
function updatePowerYAxis() {
  const values = powerChart.data.datasets
    .flatMap((dataset) => dataset.data)
    .map(Number)
    .filter((value) => Number.isFinite(value) && value >= 0);

  const highestPower = values.length ? Math.max(...values) : 0;
  const { max, step } = computePowerAxisRange(highestPower);

  // While the user has zoomed/panned (plugin moved scale.options.min/max off
  // the stock window), leave the axis alone — do not snap it back to the
  // full-data bucket on every data update. resetChartZoom() sets
  // $axisJustReset so the next update reclaims and re-buckets the axis.
  const yOpts = powerChart.options.scales.y;
  const guard = yOpts.stepGuard;
  let zoomed = guard && guard.max != null && (Number(yOpts.min) !== Number(guard.min) || Number(yOpts.max) !== Number(guard.max));
  if (powerChart.$axisJustReset) {
    zoomed = false;
    powerChart.$axisJustReset = false;
  }
  if (!zoomed) {
    yOpts.min = 0;
    yOpts.max = max;
    yOpts.ticks.stepSize = step;
    if (guard) guard.max = max;
  }

  if (DEBUG) {
    powerChart.data.datasets.forEach((dataset, index) => {
      const nums = dataset.data.map(Number).filter(Number.isFinite);
      if (!nums.length) return;
      console.log(`[CHART DEBUG] PZEM ${index + 1} datasetMin=${Math.min(...nums)} datasetMax=${Math.max(...nums)} axisMax=${max}`);
    });
  }
}
function updateLivePower() {
  if (powerHistoryMode) return;

  // Check if any meter has fresh data
  const hasFreshMeter = Array.from({ length: 9 }, (_, i) => getMeter(i + 1))
    .some(isMeterFresh);

  if (!hasFreshMeter) {
    // No fresh meters: do not extend timeline, do not add points
    return;
  }

  const label = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });

  powerChart.data.labels.push(label);

  powerChart.data.datasets.forEach((dataset, index) => {
    const meter = getMeter(index + 1);
    dataset.data.push(isMeterFresh(meter) ? normalizePowerWatts(meter) : null);
  });

  if (powerChart.data.labels.length > 30) {
    powerChart.data.labels.shift();
    powerChart.data.datasets.forEach((dataset) => dataset.data.shift());
  }

  updatePowerYAxis();
  powerChart.update();
}

/* Apply PZEM selection filter to power chart datasets */
function applyPzemSelection() {
  const selectValue = selectedPzem;
  const showAll = selectValue === "all";
  const selectedIndex = showAll ? null : parseInt(selectValue, 10) - 1;

  powerChart.data.datasets.forEach((dataset, index) => {
    dataset.hidden = !showAll && index !== selectedIndex;
  });

  powerChart.update();
}

/* =========================================================================
   ENERGY BILL CALCULATOR — historical consumption, independent of live
   online/offline status
   -------------------------------------------------------------------------
   The bill must reflect Ending cumulative Energy - Starting cumulative
   Energy from the stored "history/pzem_N" readings (the same archival path
   loadPowerHistory()/renderModalHistory() already read), over the last 30
   days. It must NOT depend on whether a PZEM is currently fresh/online —
   an offline meter with valid stored history still contributes its
   historical consumption. billHistoricalData is populated once by
   loadBillHistoricalData() (and refreshed periodically) and every bill UI
   render/rate change reads from this cache instead of live meter state. */
let billHistoricalData = null; // { perMeter: [{id, hasData, consumption}], totalConsumption }
let billLoadFailed = false;

async function loadBillHistoricalData() {
  const { start, end } = historyRangeToWindow("30d");

  try {
    const snapshots = await Promise.all(
      Array.from({ length: 9 }, (_, index) =>
        firebase.database()
          .ref(`history/pzem_${index + 1}`)
          .orderByKey()
          .startAt(String(Math.floor(start / 1000)))
          .once("value")
      )
    );

    const perMeter = snapshots.map((snapshot, index) => {
      const id = `pzem_${index + 1}`;

      const points = Object.entries(snapshot.val() || {})
        .map(([timestamp, data]) => ({
          t: timestampMilliseconds(timestamp),
          energy: Number(data && typeof data === "object" ? data.energy : NaN)
        }))
        .filter((point) => point.t <= end && Number.isFinite(point.energy))
        .sort((a, b) => a.t - b.t);

      // Need at least a start AND end cumulative-energy reading in the
      // window to compute a consumption delta — one lone reading (or none)
      // is genuinely insufficient, not a real 0.00 kWh.
      if (points.length < 2) return { id, hasData: false, consumption: 0 };

      const startEnergy = points[0].energy;
      const endEnergy = points[points.length - 1].energy;
      let consumption = endEnergy - startEnergy;

      if (consumption < 0) {
        // A cumulative energy counter should never decrease within the
        // window unless the meter was reset — clamp rather than show a
        // negative bill line, but keep it traceable in the console.
        console.warn(`[BILL DEBUG] PZEM ${index + 1} energy counter decreased (start=${startEnergy}, end=${endEnergy}) — likely a meter reset, clamped to 0`);
        consumption = 0;
      }

      return { id, hasData: true, consumption };
    });

    const totalConsumption = perMeter.reduce((sum, meter) => sum + (meter.hasData ? meter.consumption : 0), 0);

    billHistoricalData = { perMeter, totalConsumption };
    billLoadFailed = false;
  } catch (error) {
    console.error("[BILL DEBUG] Failed to load historical energy for bill calculator", error);
    billLoadFailed = true;
  }

  renderBillUI();
}

/* Shows ONE combined total for the whole monitored system (all PZEM 1-9
   summed) — not a per-meter breakdown. A per-meter row is intentionally NOT
   rendered here; this is system-wide billing, not individual PZEM billing. */
function renderBillUI() {
  const rate = Number(unitRate.value || 0);

  if (!billHistoricalData) {
    $("billTotalUnits").innerHTML = billLoadFailed ? "Insufficient historical data" : "Loading… <small>kWh</small>";
    $("billTotalCost").textContent = "—";
    $("billRateText").textContent = "Select your price per unit";
    return;
  }

  const { perMeter, totalConsumption } = billHistoricalData;
  const anyMeterHasData = perMeter.some((meter) => meter.hasData);

  $("billTotalUnits").innerHTML = anyMeterHasData
    ? `${number(totalConsumption, 2)} <small>kWh</small>`
    : "Insufficient historical data";

  $("billTotalCost").textContent = anyMeterHasData && rate
    ? new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(totalConsumption * rate)
    : "—";

  $("billRateText").textContent = rate ? `At ₹${rate.toFixed(2)} per unit` : "Select your price per unit";
  renderBillPrediction();
}

/* =========================================================================
   STAGE 10: AI BILL PREDICTION (dashboard half)
   -------------------------------------------------------------------------
   Combines the EXISTING manual-calculator actual consumption
   (billHistoricalData, already loaded by loadBillHistoricalData) with the
   Stage 9 SYSTEM forecast to estimate the upcoming bill. Reads the SAME
   sources everything else uses — no second history pipeline. The manual
   Bill Calculator above is never replaced; this is an ADDITIVE estimate.
   ========================================================================= */
function renderBillPrediction() {
  const actualEl = $("billActualUnits");
  const fcEl = $("billForecastUnits");
  const totalEl = $("billEstimatedTotalUnits");
  const costEl = $("billPredictedCost");
  const diffEl = $("billPredictedDiff");
  const noteEl = $("billPredNote");
  const capEl = $("billPredCaption");
  const rateEl = $("billPredRate");
  if (!actualEl) return;  // area not present in DOM

  // --- ACTUAL energy (from the existing Bill Calculator's 30-day history load) ---
  let actualKwh = null;
  if (billHistoricalData && billHistoricalData.perMeter &&
      billHistoricalData.perMeter.some((m) => m.hasData)) {
    actualKwh = billHistoricalData.totalConsumption;
  }

  // --- FORECAST (system scope, matches the selected horizon) ---
  const horizonKey = forecastHorizon === "24h" ? "forecast_24h" : "forecast_7d";
  const rec = forecastCache["system"];
  let forecastKwh = null;
  let confidence = "low";
  if (rec && rec[horizonKey]) {
    const h = rec[horizonKey];
    if (h.status === "FORECAST") {
      forecastKwh = forecastEnergyFromPower(h.forecast_power_w);
      confidence = h.confidence || rec.confidence || "low";
    }
  }

  const rate = Number(unitRate.value || 0);

  // --- Insufficient: withhold predicted values, keep manual calculator working ---
  if (actualKwh === null || forecastKwh === null) {
    const reasons = [];
    if (actualKwh === null) reasons.push("historical energy consumption unavailable");
    if (forecastKwh === null) reasons.push("power forecast unavailable");
    noteEl.innerHTML = `<span class="forecast-pill none">Insufficient data</span>`;
    capEl.innerHTML = `${reasons.join(" · ")}. The manual bill calculator above remains fully functional.`;
    actualEl.innerHTML = actualKwh === null ? "—" : `${number(Math.max(0, actualKwh), 2)} <small>kWh</small>`;
    fcEl.innerHTML = "—";
    totalEl.innerHTML = "—";
    costEl.textContent = "—";
    diffEl.textContent = "—";
    rateEl.textContent = rate ? `At ₹${rate.toFixed(2)} per unit` : "Select your price per unit";
    return;
  }

  // --- Sufficient: ACTUAL + FORECAST = ESTIMATED, then PREDICTED bill ---
  const actual = Math.max(0, actualKwh);
  const fc = Math.max(0, forecastKwh);
  const estimatedTotal = actual + fc;
  const estBill = rate ? estimatedTotal * rate : null;
  const currentBill = rate ? actual * rate : null;
  const diff = (estBill !== null && currentBill !== null) ? estBill - currentBill : null;

  actualEl.innerHTML = `${number(actual, 2)} <small>kWh</small>`;
  fcEl.innerHTML = `${number(fc, 2)} <small>kWh</small>`;
  totalEl.innerHTML = `${number(estimatedTotal, 2)} <small>kWh</small>`;
  costEl.textContent = estBill !== null ? inr(estBill) : "—";
  diffEl.textContent = diff !== null ? `+${inr(diff)}` : "—";
  rateEl.textContent = rate ? `At ₹${rate.toFixed(2)} per unit` : "Select your price per unit";

  const pillClass = confidence === "high" ? "high" : confidence === "medium" ? "medium" : "low";
  const horizonLabel = forecastHorizon === "24h" ? "24h" : "7d";
  noteEl.innerHTML = `<span class="forecast-pill ${pillClass}">${confidence} confidence</span>`;
  capEl.innerHTML = `ACTUAL so far + FORECAST (${horizonLabel}) ≈ ESTIMATED total for the billing period. ` +
    `Predicted bill is an <em>estimate</em>, not a guaranteed amount.`;
}

function renderDashboard() {
  const entries = Array.from({ length: 9 }, (_, index) => {
    const number = index + 1;
    return [`pzem_${number}`, getMeter(number)];
  });

  dashboard.replaceChildren();

  entries.forEach(([id, meter], index) => {
    const card = meterTemplate.content.cloneNode(true);
    const isLive = isMeterFresh(meter);
    const power = isLive ? normalizePowerWatts(meter) : 0;

    card.querySelector(".meter-number").textContent = String(index + 1).padStart(2, "0");
    card.querySelector(".meter-name").textContent = `PZEM ${index + 1}`;
    card.querySelector(".meter-id").textContent = id.toUpperCase();
    // OFF/stale meters show numeric zero, never "--" and never the old
    // stale Firebase reading — these zeros are a UI-only placeholder
    // meaning "no current live measurement available" (Part 1/6). Nothing
    // is written back to Firebase and no historical data is touched.
    card.querySelector(".meter-power strong").textContent = isLive ? number(power, 1) : "0";
    card.querySelector(".voltage").textContent = isLive ? `${number(meter.voltage, 1)} V` : "0 V";
    card.querySelector(".current").textContent = isLive ? `${number(meter.current, 2)} A` : "0.00 A";
    card.querySelector(".energy").textContent = isLive ? `${number(meter.energy, 2)} kWh` : "0.00 kWh";
    card.querySelector(".pf").textContent = isLive ? number(meter.pf, 2) : "0.00";
    const freqCell = card.querySelector(".freq");
    if (freqCell) freqCell.textContent = isLive ? `${number(meter.frequency, 1)} Hz` : "0.00 Hz";
    card.querySelector(".power-track span").style.width = `${isLive ? Math.min((power / maxPower) * 100, 100) : 0}%`;
    updatePowerViz(card, power, isLive);

    // Two independent status pills — Communication and AC Supply. Each
    // PZEM's own meter object drives its own pills only; nothing here
    // reads or is influenced by any other PZEM's data.
    const { communication, ac } = getThreeStatus(meter);

    const commStatus = card.querySelector(".comm-status");
    commStatus.classList.add(communication ? "online" : "offline");
    commStatus.querySelector("b").textContent = communication ? "CONNECTED" : "OFFLINE";

    const acStatus = card.querySelector(".ac-status");
    acStatus.classList.add(ac === true ? "online" : ac === false ? "off" : "unknown");
    acStatus.querySelector("b").textContent = ac === true ? "AC ON" : ac === false ? "AC OFF" : "UNKNOWN";

card.querySelector(".meter-card").dataset.meterNumber = String(index + 1); /* enables click-to-open popup */
    
    // Add offline class for visual styling
    const meterCard = card.querySelector(".meter-card");
    if (!isLive) {
      meterCard.classList.add("offline");
    }
    
    // STAGE 4: Add emergency-fault class and fault info if an EMERGENCY alert is active
    const alert = meterAlerts[`pzem_${index + 1}`];
    let faultInfoHTML = "";
    if (alert && alert.severity === "EMERGENCY") {
      meterCard.classList.add("emergency-fault");
      faultInfoHTML = `
        <div class="fault-badge">${alert.type.replace(/_/g, ' ')}</div>
        <div class="fault-details">
          <span class="fault-severity">[${alert.severity}]</span>
          ${alert.measured_value !== undefined ? `<span class="fault-value">${alert.measured_value} ${getUnitForType(alert.type)}</span>` : ""}
          ${alert.timestamp ? `<span class="fault-time">${new Date(alert.timestamp * 1000).toLocaleTimeString()}</span>` : ""}
        </div>
      `;
      meterCard.classList.remove("offline");
    } else if (alert && alert.severity === "WARNING") {
      // WARNING severity: add subtle indicator but no red blink
      faultInfoHTML = `
        <div class="fault-badge warning-badge">${alert.type.replace(/_/g, ' ')}</div>
        <div class="fault-details"><span class="fault-severity">[${alert.severity}]</span></div>
      `;
    }
    
    // STAGE 6: Add AI anomaly/fault info to the fault-info area
    const aiState = meterAIStates[`pzem_${index + 1}`];
    if (aiState && aiState.type === "anomaly") {
      const severity = aiState.severity || "NORMAL";
      const scoreDisplay = aiState.score !== null ? ` (score: ${Number(aiState.score).toFixed(2)})` : "";
      const aiBadge = severity === "EMERGENCY" ? "ai-alert-emergency" : severity === "WARNING" ? "ai-alert-warning" : "ai-alert-anomaly";
      faultInfoHTML += `<div class="fault-badge ai-alert-${aiBadge.replace("ai-alert-", "")}">AI ANOMALY</div>
        <div class="fault-details"><span>${aiState.label}${scoreDisplay} [${severity}]</span> ${new Date(aiState.timestamp * 1000).toLocaleTimeString()}</div>`;
    } else if (aiState && aiState.type === "fault") {
      const severity = aiState.severity || "NORMAL";
      const faultType = aiState.faultType || "unknown";
      const valueDisplay = aiState.measuredValue !== undefined ? ` — ${aiState.measuredValue}` : "";
      const aiBadge = severity === "EMERGENCY" ? "ai-alert-emergency" : severity === "WARNING" ? "ai-alert-warning" : "ai-alert-anomaly";
      faultInfoHTML += `<div class="fault-badge ai-alert-${aiBadge.replace("ai-alert-", "")}">AI FAULT</div>
        <div class="fault-details"><span>${faultType}${valueDisplay} [${severity}]</span> ${new Date(aiState.timestamp * 1000).toLocaleTimeString()}</div>`;
    }
    
    // Apply the combined fault-info HTML
    const faultInfo = card.querySelector(".fault-info");
    if (faultInfo) {
      faultInfo.innerHTML = faultInfoHTML;
    }
    
    // STAGE 6: Add AI status area to the card
    const meterAiState = meterAIStates[`pzem_${index + 1}`];
    const aiStatus = card.querySelector(".ai-status");
    if (aiStatus) {
      if (meterAiState && meterAiState.type && meterAiState.type !== "anomaly" && meterAiState.type !== "fault") {
        // No active AI anomaly/fault — show NORMAL or NO AI DATA
        if (!meterAiState.severity || meterAiState.severity === "NORMAL") {
          aiStatus.className = "ai-status ai-status-normal";
          aiStatus.innerHTML = `<span class="ai-status-pill">NORMAL</span><span class="ai-status-details">No active anomalies or faults</span>`;
        } else {
          aiStatus.className = "ai-status ai-status-normal";
          aiStatus.innerHTML = `<span class="ai-status-pill">NORMAL</span><span class="ai-status-details">AI monitoring active</span>`;
        }
      } else if (meterAiState && meterAiState.type === "anomaly") {
        // Active anomaly
        const severity = meterAiState.severity || "NORMAL";
        const scoreDisplay = meterAiState.score !== null ? ` (score: ${Number(meterAiState.score).toFixed(2)})` : "";
        aiStatus.className = `ai-status ai-status-anomaly`;
        aiStatus.innerHTML = `<span class="ai-status-pill">ANOMALY</span><span class="ai-status-details"> ${meterAiState.label}${scoreDisplay} [${severity}]</span><span class="ai-status-details"> ${new Date(meterAiState.timestamp * 1000).toLocaleTimeString()}</span>`;
      } else if (meterAiState && meterAiState.type === "fault") {
        // Active fault
        const severity = String(meterAiState.severity || "NORMAL");
        const faultType = String(meterAiState.faultType || "unknown");
        const valueDisplay = meterAiState.measuredValue !== undefined ? ` — ${meterAiState.measuredValue}` : "";
        aiStatus.className = `ai-status ai-status-${severity.toLowerCase() === "emergency" ? "emergency" : severity.toLowerCase() === "warning" ? "warning" : "anomaly"}`;
        aiStatus.innerHTML = `<span class="ai-status-pill">${severity}</span><span class="ai-status-details"> ${faultType}${valueDisplay} [${severity}]</span><span class="ai-status-details"> ${new Date((meterAiState.timestamp || 0) * 1000).toLocaleTimeString()}</span>`;
      } else {
        // No AI data yet
        aiStatus.className = "ai-status ai-status-no-data";
        aiStatus.innerHTML = `<span class="ai-status-pill">NO AI DATA</span><span class="ai-status-details">AI monitoring not yet active</span>`;
      }
    }
    
    dashboard.appendChild(card);

    if (DEBUG) {
      const age = meterAgeMs(meter);
      console.log(`[DASHBOARD FRESHNESS] PZEM ${index + 1} age=${age === null ? "n/a" : age + "ms"} communication=${communication ? "CONNECTED" : "OFFLINE"} ac=${ac === true ? "ON" : ac === false ? "OFF" : "UNKNOWN"}`);
    }
  });

  // Every summary widget below is derived ONLY from `online` (fresh)
  // meters. With zero fresh meters, every reduce() below naturally lands on
  // 0 — Part 4's "0 fresh meters -> 0 W / 0 V / 0/9" requirement falls out
  // of the filter itself, not a separate special case.
  const online = entries.filter(([, meter]) => isMeterFresh(meter));
  const totalPower = online.reduce((sum, [, meter]) => sum + normalizePowerWatts(meter), 0);
  const totalEnergy = online.reduce((sum, [, meter]) => sum + Number(meter.energy || 0), 0);
  const averageVoltage = online.length
    ? online.reduce((sum, [, meter]) => sum + Number(meter.voltage || 0), 0) / online.length
    : 0;

  $("totalPower").innerHTML = `${number(totalPower, 1)} <small>W</small>`;
  $("totalEnergy").innerHTML = `${number(totalEnergy, 2)} <small>kWh</small>`;
  $("onlineMeters").innerHTML = `${online.length} <small>/ 9</small>`;
  $("averageVoltage").innerHTML = `${number(averageVoltage, 1)} <small>V</small>`;
  $("meterCount").textContent = "9 meters";

  // STAGE 6: AI overview/summary — compact, only when AI data exists
  const activeAnomalyMeters = Object.values(meterAIStates).filter(
    (s) => s && s.type === "anomaly" && (s.severity === "EMERGENCY" || s.severity === "WARNING")
  ).length;
  const activeFaultMeters = Object.values(meterAIStates).filter(
    (s) => s && s.type === "fault" && (s.severity === "EMERGENCY" || s.severity === "WARNING")
  ).length;
  const criticalAIEvent = Object.values(meterAIStates).find(
    (s) => s && s.type === "anomaly" && s.severity === "EMERGENCY"
  ) || Object.values(meterAIStates).find(
    (s) => s && s.type === "fault" && s.severity === "EMERGENCY"
  );

  const criticalPzemNumber = (() => {
    if (!criticalAIEvent) return null;
    for (const [key, state] of Object.entries(meterAIStates)) {
      if (state === criticalAIEvent) {
        return parseInt(key.replace("pzem_", ""), 10);
      }
    }
    return null;
  })();

  const aiSummary = document.createElement("div");
  aiSummary.id = "aiSummary";
  aiSummary.style.cssText = `
    margin-top: 8px;
    padding: 6px 10px;
    background: var(--surface-soft);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    font-size: 10px;
    color: var(--muted);
    display: none;
  `;
  if (criticalAIEvent) {
    aiSummary.style.display = "block";
    aiSummary.innerHTML = criticalAIEvent.type === "anomaly"
      ? `<b>AI:</b> anomaly on PZEM ${criticalPzemNumber || "?"} [${criticalAIEvent.severity}] ${new Date(criticalAIEvent.timestamp * 1000).toLocaleTimeString()}`
      : `<b>AI:</b> fault on PZEM ${criticalPzemNumber || "?"} [${criticalAIEvent.severity}] ${new Date(criticalAIEvent.timestamp * 1000).toLocaleTimeString()}`;
  } else {
    aiSummary.style.display = "none";
    aiSummary.innerHTML = "";
  }
  document.querySelector(".summary-grid").parentNode.insertBefore(aiSummary, document.querySelector(".summary-grid").nextSibling);

  // Cached for the [LIVE SUMMARY] diagnostic printed by updateFrequency(),
  // which always runs immediately after this function.
  lastLiveSummary.freshMeters = online.length;
  lastLiveSummary.power = totalPower;
  lastLiveSummary.voltage = averageVoltage;
}

function timestampMilliseconds(timestamp) {
  const value = Number(timestamp);
  return String(timestamp).length > 10 ? value : value * 1000;
}

/* Uses the exact same range interpretation as the PZEM popup's Historical
   Usage selector (historyRangeToWindow(), defined below) so both selectors
   behave consistently — same 1h/6h/12h/today/yesterday/7d/30d windows,
   same Firebase "history/pzem_N" source, just applied to all 9 meters at
   once instead of one. */
async function loadPowerHistory(range) {
  const note = document.querySelector(".chart-panel .chart-note");
  const requestId = ++historyRequestId;
  const { start, end } = historyRangeToWindow(range);

  powerHistoryMode = true;
  note.textContent = "Loading history...";

  try {
    const snapshots = await Promise.all(
      Array.from({ length: 9 }, (_, index) =>
        firebase.database()
          .ref(`history/pzem_${index + 1}`)
          .orderByKey()
          .startAt(String(Math.floor(start / 1000)))
          .once("value")
      )
    );

    if (requestId !== historyRequestId) return;

    const timeline = new Map();

    snapshots.forEach((snapshot, meterIndex) => {
      Object.entries(snapshot.val() || {}).forEach(([timestamp, data]) => {
        const time = timestampMilliseconds(timestamp);
        if (time > end) return; // respect the range's end boundary too (e.g. "Yesterday" must exclude today)

        if (!timeline.has(time)) timeline.set(time, Array(9).fill(null));
        timeline.get(time)[meterIndex] = normalizePowerWatts(data);
      });
    });

    const points = [...timeline.entries()].sort(([a], [b]) => a - b);

    if (!points.length) {
      powerHistoryMode = false;
      powerChart.data.labels = [];
      powerChart.data.datasets.forEach((dataset) => { dataset.data = []; });
      resetChartZoom(powerChart);
      powerChart.update();
      applyPzemSelection();
      note.textContent = "No stored history — showing live data";
      return;
    }

    powerChart.data.labels = points.map(([time]) =>
      new Date(time).toLocaleString([], {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit"
      })
    );

    powerChart.data.datasets.forEach((dataset, index) => {
      dataset.data = points.map(([, values]) => values[index]);
    });
    resetChartZoom(powerChart); // new query = fresh axis; old zoom window no longer applies
    updatePowerYAxis();
    powerChart.update();
    applyPzemSelection();

    powerHistoryMode = false;

    const rangeLabel = range === "1h" ? "1 hour"
      : range === "6h" ? "6 hours"
      : range === "12h" ? "12 hours"
      : range === "today" ? "today"
      : range === "yesterday" ? "yesterday"
      : range === "30d" ? "30 days"
      : "7 days";

    note.textContent = `${points.length} readings · ${rangeLabel}`;
  } catch (error) {
    console.error(error);
    powerHistoryMode = false;
    note.textContent = "History unavailable — showing live data";
    applyPzemSelection();
  }
}

function useLiveData(data) {
  metersData = data || {};

  renderDashboard();
  updateFrequency();
  updateLivePower();
  updateSystemStatus();

  $("lastUpdated").textContent = `Last synchronised ${new Date().toLocaleTimeString()}`;

  trackPzemRuntimeState(); /* isolated: powers the new per-meter popup only, does not affect anything above */
}

/* System status based on meter freshness, not Firebase connection */
function updateSystemStatus() {
  const entries = Array.from({ length: 9 }, (_, index) => [`pzem_${index + 1}`, getMeter(index + 1)]);
  const online = entries.filter(([, meter]) => isMeterFresh(meter));
  
  const statusEl = $("connectionStatus");
  if (online.length === 0) {
    statusEl.className = "connection-pill offline";
    statusEl.innerHTML = "<span></span> System offline";
  } else {
    statusEl.className = "connection-pill online";
    statusEl.innerHTML = "<span></span> System online";
  }
}

/* Dashboard is read-only: it authenticates anonymously (no password embedded
   in public JS) purely so RTDB rules requiring "auth != null" allow the read.
   Write access stays restricted to the ESP32's email/password device user. */
function showConnectionError(message) {
  $("connectionStatus").className = "connection-pill error";
  $("connectionStatus").innerHTML = "<span></span> Connection error";
  $("lastUpdated").textContent = message;
}

function attachLiveListener() {
  /* Supports both /meters/pzem_1 and your ESP code /energy/pzem1 */
  firebase.database().ref("meters").on(
    "value",
    (snapshot) => {
      if (Object.keys(snapshot.val() || {}).length) useLiveData(snapshot.val());
    },
    (error) => {
      console.error("[DASHBOARD] Firebase read permission denied", error);
      showConnectionError("Live data blocked by database rules — see console");
    }
  );

  firebase.database().ref("energy").on(
    "value",
    (snapshot) => {
      if (Object.keys(snapshot.val() || {}).length) useLiveData(snapshot.val());
    },
    (error) => {
      console.error("[DASHBOARD] Firebase read permission denied", error);
    }
  );
}

firebase.auth().signInAnonymously().catch((error) => {
  console.error("[DASHBOARD] Firebase Auth error", error.code, error.message);
  showConnectionError("Sign-in failed — see console");
});

/* STAGE 4: Cache for latest fault diagnosis alerts per PZEM,
   populated by the Firebase 'alerts' child listeners added below. */
let meterAlerts = {};

/* STAGE 6: Cache for latest AI anomaly/fault state per PZEM,
   populated by the Firebase 'ai/anomalies' and 'ai/faults' child listeners. */
let meterAIStates = {};

/* STAGE 4: Firebase listeners for fault diagnosis alerts written by the
   ESP32 firmware. Alerts are stored at /alerts/pzem_N/<unix-timestamp>
   with JSON containing type, severity, measured_value, timestamp, etc. */
function attachFaultAlertListener() {
  /* child_added: new alerts arriving */
  firebase.database().ref("alerts").on(
    "child_added",
    (snapshot) => {
      const alert = snapshot.val();
      const pzemNumber = alert.pzem_number;
      if (!pzemNumber) return;
      meterAlerts[`pzem_${pzemNumber}`] = {
        type: alert.type,
        severity: alert.severity || "UNKNOWN",
        measured_value: alert.measured_value,
        timestamp: alert.timestamp,
        reason: alert.reason || "",
        evidence: alert.evidence || {}
      };
      if (DEBUG) console.log(`[ALERT] New fault for PZEM ${pzemNumber}: ${alert.type} ${alert.severity}`);
    },
    (error) => {
      console.error("[DASHBOARD] Firebase alerts child_added error", error);
    }
  );
  /* child_changed: alert updated (e.g. severity change, value update) */
  firebase.database().ref("alerts").on(
    "child_changed",
    (snapshot) => {
      const alert = snapshot.val();
      const pzemNumber = alert.pzem_number;
      if (!pzemNumber) return;
      meterAlerts[`pzem_${pzemNumber}`] = {
        type: alert.type,
        severity: alert.severity || "UNKNOWN",
        measured_value: alert.measured_value,
        timestamp: alert.timestamp,
        reason: alert.reason || "",
        evidence: alert.evidence || {}
      };
    },
    (error) => {
      console.error("[DASHBOARD] Firebase alerts child_changed error", error);
    }
  );
  /* child_removed: alert cleared */
  firebase.database().ref("alerts").on(
    "child_removed",
    (snapshot) => {
      const pzemNumber = snapshot.val()?.pzem_number;
      if (pzemNumber) {
        delete meterAlerts[`pzem_${pzemNumber}`];
      }
    },
    (error) => {
      console.error("[DASHBOARD] Firebase alerts child_removed error", error);
    }
  );
}

/* STAGE 6: Firebase listeners for AI anomaly and fault results written by
   the AI backend Stage 5. Anomalies at /ai/anomalies/pzem_N/<timestamp>,
   faults at /ai/faults/pzem_N/<timestamp>. */
function attachAIImplListener() {
  /* Anomalies: /ai/anomalies/pzem_N/<timestamp> */
  firebase.database().ref("ai/anomalies").on(
    "child_added",
    (snapshot) => {
      const anomaly = snapshot.val();
      const pzemNumber = anomaly.pzem_number;
      if (!pzemNumber) return;
      meterAIStates["pzem_" + pzemNumber] = {
        type: "anomaly",
        label: anomaly.anomaly_label || "Anomaly",
        score: anomaly.anomaly_score !== undefined ? anomaly.anomaly_score : null,
        severity: anomaly.anomaly_severity || "NORMAL",
        timestamp: anomaly.timestamp,
        eventTimestamp: anomaly.eventTimestamp
      };
      if (DEBUG) console.log("[AI] New anomaly for PZEM " + pzemNumber + ": " + anomaly.anomaly_label + " severity=" + anomaly.anomaly_severity);
    },
    (error) => {
      console.error("[DASHBOARD] Firebase anomalies child_added error", error);
    }
  );
  /* faults: /ai/faults/pzem_N/<timestamp> */
  firebase.database().ref("ai/faults").on(
    "child_changed",
    (snapshot) => {
      const fault = snapshot.val();
      const pzemNumber = fault.pzem_number;
      if (!pzemNumber) return;
      meterAIStates["pzem_" + pzemNumber] = {
        type: "fault",
        faultType: fault.fault_type || "unknown",
        severity: fault.severity || "NORMAL",
        measuredValue: fault.measured_value,
        reason: fault.reason || fault.evidence || "",
        timestamp: fault.timestamp
      };
      if (DEBUG) console.log("[AI] New fault for PZEM " + pzemNumber + ": " + fault.fault_type + " severity=" + fault.severity);
    },
    (error) => {
      console.error("[DASHBOARD] Firebase faults child_changed error", error);
    }
  );
  /* Also listen for child_added and child_removed on faults */
  firebase.database().ref("ai/faults").on(
    "child_added",
    (snapshot) => {
      const fault = snapshot.val();
      const pzemNumber = fault.pzem_number;
      if (!pzemNumber) return;
      meterAIStates["pzem_" + pzemNumber] = {
        type: "fault",
        faultType: fault.fault_type || "unknown",
        severity: fault.severity || "NORMAL",
        measuredValue: fault.measured_value,
        reason: fault.reason || fault.evidence || "",
        timestamp: fault.timestamp
      };
    },
    (error) => {
      console.error("[DASHBOARD] Firebase faults child_added error", error);
    }
  );
  firebase.database().ref("ai/faults").on(
    "child_removed",
    (snapshot) => {
      const pzemNumber = snapshot.val()?.pzem_number;
      if (pzemNumber) {
        delete meterAIStates["pzem_" + pzemNumber];
      }
    },
    (error) => {
      console.error("[DASHBOARD] Firebase faults child_removed error", error);
    }
  );
}

firebase.auth().signInAnonymously().catch((error) => {
  console.error("[DASHBOARD] Firebase Auth error", error.code, error.message);
  showConnectionError("Sign-in failed — see console");
});

firebase.auth().onAuthStateChanged((user) => {
  if (user) {
    attachLiveListener();
    attachFaultAlertListener();
    attachAIImplListener();
  }
});

$("powerRange").addEventListener("change", (event) => {
  loadPowerHistory(event.target.value);
});

$("pzemSelect").addEventListener("change", (event) => {
  selectedPzem = event.target.value;
  applyPzemSelection();
});

unitRate.addEventListener("change", () => renderBillUI());

function applyTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  localStorage.setItem("theme", theme);
}

const savedTheme = localStorage.getItem("theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
applyTheme(savedTheme);

$("themeToggle").addEventListener("click", () => {
  applyTheme(document.body.classList.contains("dark") ? "light" : "dark");
  updateChartThemeColors();
});

function getUnitForType(type) {
  switch (type) {
    case "overvoltage": return "V";
    case "undervoltage": return "V";
    case "overcurrent": return "A";
    case "power_factor_drop": return "PF";
    case "frequency_deviation": return "Hz";
    case "high_power": return "W";
    case "comm_degraded": return "";
    default: return "";
  }
}
function formatKolkataDateTime(timestampMs) {
  const dateFmt = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", day: "2-digit", month: "2-digit", year: "numeric" });
  const timeFmt = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false });
  const date = new Date(timestampMs);
  return { date: dateFmt.format(date).replace(/\//g, "-"), time: timeFmt.format(date) };
}

/* Exports the archival 5-minute HISTORY data (history/pzem_N), not the live
   snapshot — one row per stored reading, across all 9 meters. This is
   separate from live monitoring on purpose: live data drives the dashboard
   cards/graphs, history data is what gets exported and later analyzed. */
$("exportButton").addEventListener("click", async () => {
  const originalLabel = $("exportButton").textContent;
  $("exportButton").disabled = true;
  $("exportButton").textContent = "Preparing export…";

  try {
    const snapshots = await Promise.all(
      Array.from({ length: 9 }, (_, index) =>
        firebase.database().ref(`history/pzem_${index + 1}`).once("value")
      )
    );

    const rows = [["Date", "Time", "PZEM ID", "Voltage (V)", "Current (A)", "Power (W)", "Energy (kWh)", "Frequency (Hz)", "Power Factor"]];

    snapshots.forEach((snapshot, meterIndex) => {
      const pzemId = `PZEM ${meterIndex + 1}`;
      Object.entries(snapshot.val() || {}).forEach(([timestampKey, reading]) => {
        if (!reading || typeof reading !== "object") return; // skip malformed entries, never invent values

        const { date, time } = formatKolkataDateTime(timestampMilliseconds(timestampKey));
        rows.push([
          date,
          time,
          pzemId,
          reading.voltage ?? 0,
          reading.current ?? 0,
          reading.power ?? 0,
          reading.energy ?? 0,
          reading.frequency ?? 0,
          reading.pf ?? 0
        ]);
      });
    });

    if (rows.length === 1) {
      // No historical records exist yet (expected until the ESP32 side
      // starts writing to history/pzem_N) — say so honestly, don't
      // download a misleading near-empty file.
      alert("No historical data is available yet. 5-minute historical records will appear here once meter history starts being recorded.");
      return;
    }

    const link = document.createElement("a");
    link.href = URL.createObjectURL(
      new Blob([rows.map((row) => row.join(",")).join("\n")], { type: "text/csv" })
    );
    link.download = "pzem-historical-readings.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  } catch (error) {
    console.error(error);
    alert("Could not export historical data right now. Please try again.");
  } finally {
    $("exportButton").disabled = false;
    $("exportButton").textContent = originalLabel;
  }
});


renderDashboard();
loadPowerHistory("7d");
loadBillHistoricalData();

/* Refresh the bill's historical consumption on the same cadence new
   "history/pzem_N" rows actually arrive (HISTORY_SLOT_MS = 5 min), so the
   bill stays current without needing a page reload. Independent of the 5 s
   freshness-recheck timer below, which only re-evaluates live/offline
   status and does not touch bill data. */
setInterval(loadBillHistoricalData, HISTORY_SLOT_MS);

/* =========================================================================
   PZEM DETAIL POPUP (ADD-ON FEATURE)
   Everything below is new and additive. It reuses the existing metersData /
   getMeter() live state and the existing Firebase "history/pzem_N" path
   (same one loadPowerHistory() already reads) — no new connections, no
   duplicate polling, no fake data. Nothing above this line was changed
   except the two small hooks already marked above.
   ========================================================================= */

/* NOTE: there is intentionally no equipment/appliance ON-OFF power-threshold
   logic here (no "fan on", "load on", etc). PZEM 1-9 are generic meter
   identifiers; what they're wired to is unknown to this dashboard. The only
   state tracked below is METER connectivity — whether a PZEM is currently
   sending fresh data (LIVE) or not (OFF) — via isMeterFresh() above. */

const alertThresholds = {
  highPower: 2500,                  // W
  highCurrent: 13,                  // A
  lowVoltage: 200,                  // V
  highVoltage: 250                  // V
};

let meterRuntimeState = {};         // per-meter: online flag, current session, session history, alerts, live series
let activeMeterNumber = null;
let overviewChart = null;
let modalHistoryChart = null;
let modalHistoryRequestId = 0;

/* --- Stage 9 forecast panel state --- */
let forecastChart = null;
let forecastHorizon = "24h";          // "24h" | "7d"
let selectedForecastPzem = "system";  // "system" | "pzem_N"
const forecastCache = {};            // leaf ("pzem_N" | "system") -> latest record
let energySavingCache = null;        // latest /ai/energy_saving record (or null)
const FORECAST_DEMO = new URLSearchParams(location.search).has("forecastDemo");

// Init is deferred to here (after the `let forecastChart` declaration above is
// initialized) so ensureForecastChart() does not hit a temporal-dead-zone
// ReferenceError when called during top-level script execution.
initForecastPanel();

function getRuntimeState(n) {
  if (!meterRuntimeState[n]) {
    meterRuntimeState[n] = { online: false, sessionStart: null, sessions: [], historicalSessions: [], alerts: [], liveSeries: [], flags: {} };
  }
  return meterRuntimeState[n];
}

function pushAlert(state, type, label, detail) {
  state.alerts.push({ type, label, detail, time: Date.now() });
  if (state.alerts.length > 40) state.alerts.shift();
}

/* Reads the same live meter data the dashboard cards already use and derives
   PZEM connectivity (LIVE/OFF) sessions + electrical threshold alerts
   locally. Does not fetch anything new. */
function trackPzemRuntimeState() {
  const now = Date.now();

  for (let i = 1; i <= 9; i++) {
    const meter = getMeter(i);
    const state = getRuntimeState(i);
    const isLive = isMeterFresh(meter);
    const power = normalizePowerWatts(meter);
    const voltage = Number(meter.voltage || 0);
    const current = Number(meter.current || 0);

    // Only a FRESH valid reading may create a live point. While this meter is
    // OFFLINE/stale we append NOTHING — no new timestamp, no zero, no reuse of
    // the last stale value — so the live timeline freezes instead of filling
    // with artificial points every update cycle. Points resume automatically
    // on the next fresh reading; already-collected points stay visible.
    if (isLive) {
      state.liveSeries.push({ t: now, power });
      if (state.liveSeries.length > 30) state.liveSeries.shift();
    }

    // Meter connectivity (LIVE/OFF) transition — this is the ONLY state
    // transition tracked here. There is no equipment/appliance ON-OFF
    // detection: PZEM 1-9 are generic meter identifiers, and a meter is
    // LIVE regardless of how much (or how little) power it's currently
    // reporting.
    if (isLive !== state.online) {
      console.log(`[PZEM ${i} STATUS] ${state.online ? "LIVE" : "OFF"} → ${isLive ? "LIVE" : "OFF"}`);

      if (isLive) {
        state.sessionStart = now;
        pushAlert(state, "info", "PZEM Live", "");
      } else {
        if (state.sessionStart) {
          state.sessions.push({ start: state.sessionStart, end: now });
          if (state.sessions.length > 60) state.sessions.shift();
        }
        state.sessionStart = null;
        pushAlert(state, "offline", "PZEM Offline", "Meter stopped reporting fresh data");
      }
      state.online = isLive;
    }

    if (isLive) {
      const wasHighPower = state.flags.highPower;
      state.flags.highPower = power > alertThresholds.highPower;
      if (state.flags.highPower && !wasHighPower) pushAlert(state, "highPower", "High Power", `Power exceeded ${alertThresholds.highPower} W`);

      const wasHighCurrent = state.flags.highCurrent;
      state.flags.highCurrent = current > alertThresholds.highCurrent;
      if (state.flags.highCurrent && !wasHighCurrent) pushAlert(state, "highCurrent", "High Current", `Current exceeded ${alertThresholds.highCurrent} A`);

      const wasLowVoltage = state.flags.lowVoltage;
      state.flags.lowVoltage = voltage > 0 && voltage < alertThresholds.lowVoltage;
      if (state.flags.lowVoltage && !wasLowVoltage) pushAlert(state, "lowVoltage", "Low Voltage", `Voltage dropped below ${alertThresholds.lowVoltage} V`);

      const wasHighVoltage = state.flags.highVoltage;
      state.flags.highVoltage = voltage > alertThresholds.highVoltage;
      if (state.flags.highVoltage && !wasHighVoltage) pushAlert(state, "highVoltage", "High Voltage", `Voltage exceeded ${alertThresholds.highVoltage} V`);
    }
  }

  if (activeMeterNumber) refreshOpenModalLiveParts();
}

function ensureOverviewChart() {
  if (overviewChart) return overviewChart;
  const c = getChartColors();
  overviewChart = new Chart($("overviewLiveChart"), {
    type: "line",
    plugins: [graphDepth],
    data: { labels: [], datasets: [{ label: "Power", data: [], borderColor: "#1e6bd6", backgroundColor: "rgba(30,107,214,.12)", borderWidth: 2, tension: 0.35, pointRadius: 0, fill: true, spanGaps: true }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: c.tooltipBg,
          titleColor: c.tooltipText,
          bodyColor: c.tooltipText,
          titleFont: { size: 12, weight: 600, family: "'Space Grotesk', sans-serif" },
          bodyFont: { size: 11, family: "'DM Sans', sans-serif" },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (context) => `Power: ${formatPowerLabel(Number(context.parsed.y || 0))}`
          }
        },
        zoom: {
          pan: { enabled: true, mode: 'xy' },
          zoom: { wheel: { enabled: true, modifierKey: 'ctrl' }, pinch: { enabled: true }, mode: 'xy' },
          limits: { x: { min: 'original', max: 'original', minRange: 10 }, y: { min: 'original', max: 'original' } }
        }
      },
      scales: {
        x: { 
          grid: { display: false, color: c.grid }, 
          ticks: { 
            maxTicksLimit: 6, 
            color: c.tick,
            font: { size: 9, family: "'DM Sans', sans-serif" },
            maxRotation: 0
          } 
        },
        y: { 
          beginAtZero: true, 
          title: { display: true, text: "Power (W)", color: c.title, font: { size: 10, weight: 600, family: "'DM Sans', sans-serif" } }, 
          ticks: { 
            color: c.tick,
            font: { size: 9, family: "'DM Sans', sans-serif" },
            callback(value, index, ticks) { return powerAxisTickLabel(value, index, ticks, this); },
            padding: 6
          },
          grid: { color: c.grid, drawBorder: false }
        }
      }
    }
  });
  return overviewChart;
}

function ensureModalHistoryChart() {
  if (modalHistoryChart) return modalHistoryChart;
  const c = getChartColors();
  modalHistoryChart = new Chart($("modalHistoryChart"), {
    type: "line",
    plugins: [graphDepth],
    data: { labels: [], datasets: [{ label: "Power", data: [], borderColor: "#7b4cf6", backgroundColor: "rgba(123,76,246,.13)", borderWidth: 2, tension: 0.3, pointRadius: 0, fill: true, spanGaps: true }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: c.tooltipBg,
          titleColor: c.tooltipText,
          bodyColor: c.tooltipText,
          titleFont: { size: 12, weight: 600, family: "'Space Grotesk', sans-serif" },
          bodyFont: { size: 11, family: "'DM Sans', sans-serif" },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (context) => `Power: ${formatPowerLabel(Number(context.parsed.y || 0))}`
          }
        },
        zoom: {
          pan: { enabled: true, mode: 'xy' },
          zoom: { wheel: { enabled: true, modifierKey: 'ctrl' }, pinch: { enabled: true }, mode: 'xy' },
          limits: { x: { min: 'original', max: 'original', minRange: 10 }, y: { min: 'original', max: 'original' } }
        }
      },
      scales: {
        x: { 
          grid: { display: false, color: c.grid }, 
          ticks: { 
            maxTicksLimit: 8, 
            color: c.tick,
            font: { size: 9, family: "'DM Sans', sans-serif" },
            maxRotation: 0,
            autoSkip: true,
            autoSkipPadding: 30
          } 
        },
        y: { 
          beginAtZero: true, 
          title: { display: true, text: "Power (W)", color: c.title, font: { size: 10, weight: 600, family: "'DM Sans', sans-serif" } }, 
          ticks: { 
            color: c.tick,
            font: { size: 9, family: "'DM Sans', sans-serif" },
            callback(value, index, ticks) { return powerAxisTickLabel(value, index, ticks, this); },
            padding: 6
          },
          grid: { color: c.grid, drawBorder: false }
        }
      }
    }
  });
  return modalHistoryChart;
}

/* =========================================================================
   STAGE 9: FORECAST PANEL (additive — does not touch any existing graph)
   Reads /ai/forecast/pzem_N/<ts> and /ai/forecast/system/<ts> written by
   the AI backend. Shows historical ACTUAL 5-minute data (from the same
   history/pzem_N path the rest of the dashboard uses) followed by the
   forecast continuation. Honestly shows "Insuffient data" when no valid
   forecast exists — never a fake line.
   ========================================================================= */

function ensureForecastChart() {
  if (forecastChart) return forecastChart;
  const c = getChartColors();
  forecastChart = new Chart($("forecastChart"), {
    type: "line",
    plugins: [graphDepth],
    data: {
      datasets: [
        // Order matters: upper/lower are a filled uncertainty band (upper fills
        // down to the next dataset, lower), kept out of the legend.
        { label: "Upper bound", data: [], borderColor: "transparent", backgroundColor: "rgba(123,76,246,0.12)", borderWidth: 0, pointRadius: 0, fill: "+1", tension: 0.3 },
        { label: "Lower bound", data: [], borderColor: "transparent", borderWidth: 0, pointRadius: 0, fill: false, tension: 0.3 },
        { label: "Actual (5-min history)", data: [], borderColor: "#7b4cf6", backgroundColor: "rgba(123,76,246,0.10)", borderWidth: 2, tension: 0.3, pointRadius: 0, fill: false, spanGaps: true },
        { label: "Forecast", data: [], borderColor: "#f59e0b", borderWidth: 2, borderDash: [6, 4], tension: 0.3, pointRadius: 0, fill: false, spanGaps: true },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: {
            boxWidth: 9, usePointStyle: true, pointStyle: "circle", color: c.legend,
            font: { size: 11, weight: 500, family: "'DM Sans', sans-serif" },
            filter: (item) => !/bound/i.test(item.text)
          }
        },
        tooltip: {
          backgroundColor: c.tooltipBg, titleColor: c.tooltipText, bodyColor: c.tooltipText,
          titleFont: { size: 12, weight: 600, family: "'Space Grotesk', sans-serif" },
          bodyFont: { size: 11, family: "'DM Sans', sans-serif" },
          padding: 12, cornerRadius: 8, displayColors: true,
          callbacks: { label: (ctx) => `${ctx.dataset.label}: ${formatPowerLabel(Number(ctx.parsed.y || 0))}` }
        },
        zoom: {
          pan: { enabled: true, mode: "xy" },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "xy" },
          limits: { x: { min: "original", max: "original" }, y: { min: "original", max: "original" } }
        }
      },
      scales: {
        x: {
          type: "linear",
          grid: { display: false, color: c.grid },
          ticks: {
            maxTicksLimit: 8, color: c.tick, font: { size: 10, family: "'DM Sans', sans-serif" },
            callback: (value) => {
              const d = new Date(value);
              return d.toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
            }
          }
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "Power (W)", color: c.title, font: { size: 11, weight: 600, family: "'DM Sans', sans-serif" } },
          ticks: {
            color: c.tick, font: { size: 10, family: "'DM Sans', sans-serif" },
            callback: function (value, index, ticks) { return powerAxisTickLabel(value, index, ticks, this); }
          },
          grid: { color: c.grid, drawBorder: false }
        }
      }
    }
  });
  return forecastChart;
}

function horizonKey(horizon) {
  return horizon === "24h" ? "forecast_24h" : "forecast_7d";
}

function storeForecast(leaf, rec) {
  if (!rec) return;
  const cur = forecastCache[leaf];
  if (!cur || (rec.anchor_timestamp || 0) >= (cur.anchor_timestamp || 0)) {
    forecastCache[leaf] = rec;
  }
}

function ingestForecastTree(tree) {
  Object.entries(tree || {}).forEach(([leaf, records]) => {
    Object.entries(records || {}).forEach(([anchor, rec]) => storeForecast(leaf, rec));
  });
}

/* Loads the latest forecast records once, then keeps them live. Mirrors the
   Stage 6 child_added pattern used for anomalies/faults. */
function loadForecastCache() {
  try {
    const ref = firebase.database().ref("ai/forecast");
    ref.once("value").then((snap) => { ingestForecastTree(snap.val() || {}); renderBillPrediction(); });
    ref.on("child_added", (snap) => { storeForecast(snap.key, snap.val()); renderBillPrediction(); });
  } catch (err) {
    console.error("[DASHBOARD] Forecast cache load failed", err);
  }
}

function latestForecastRecord(meterKey) {
  return forecastCache[meterKey] || null;
}

/* ---------------------------------------------------------------------------
   Stage 11 — Energy Saving suggestions (historical/AI data only, never live).
   Loads the latest record from /ai/energy_saving once, then keeps it live via
   child_added — the same pattern as the Stage 6/9 caches. Rendering only ever
   reads the cached AI record; no recommendation is computed from 10s readings.
   --------------------------------------------------------------------------- */
function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function _esPriorityRank(p) {
  return p === "HIGH" ? 3 : p === "MEDIUM" ? 2 : 1;
}

function storeEnergySaving(rec) {
  if (!rec || rec.timestamp == null) return;
  if (!energySavingCache || rec.timestamp > energySavingCache.timestamp) {
    energySavingCache = rec;
    renderEnergySaving();
  }
}

function ingestEnergySavingTree(tree) {
  if (!tree) return;
  Object.keys(tree).forEach((k) => storeEnergySaving(tree[k]));
}

function loadEnergySavingCache() {
  try {
    const ref = firebase.database().ref("ai/energy_saving");
    ref.once("value").then((snap) => { ingestEnergySavingTree(snap.val() || {}); });
    ref.on("child_added", (snap) => { storeEnergySaving(snap.val()); });
  } catch (err) {
    console.error("[DASHBOARD] Energy-saving cache load failed", err);
  }
}

function renderEnergySavingItem(r) {
  const meter = r.pzem_number == null ? "SYSTEM" : ("PZEM " + r.pzem_number);
  const sav = [];
  if (r.potential_saving_kwh != null) sav.push("≈ " + Number(r.potential_saving_kwh).toFixed(2) + " kWh");
  if (r.potential_cost_saving != null) sav.push("≈ " + inr(r.potential_cost_saving));
  const savText = sav.length ? ` <span class="es-sav">Est. saving ${sav.join(" · ")} (estimate)</span>` : "";
  const win = r.evidence_window ? ` · window ${escapeHtml(r.evidence_window)}` : "";
  return `<div class="es-item es-${escapeHtml(r.priority)}">
    <div class="es-top">
      <span class="es-badge es-badge-${escapeHtml(r.priority)}">${escapeHtml(r.priority)}</span>
      <span class="es-meter">${escapeHtml(meter)}</span>
      <span class="es-type">${escapeHtml(r.recommendation_type)}</span>
    </div>
    <p class="es-reason">${escapeHtml(r.reason)}</p>
    <p class="es-evidence">${escapeHtml(r.recommendation)}${savText}${win}</p>
  </div>`;
}

function renderEnergySaving() {
  const note = $("energySavingNote");
  const list = $("energySavingList");
  if (!list) return;
  if (!energySavingCache || !energySavingCache.recommendations ||
      energySavingCache.recommendations.length === 0) {
    if (note) note.innerHTML = `<span class="forecast-pill none">No suggestions</span>`;
     list.innerHTML = `<p class="es-empty">No energy-saving suggestions yet. Recommendations are generated from historical data and AI analysis.</p>`;
    return;
  }
  const recs = energySavingCache.recommendations.slice()
    .sort((a, b) => _esPriorityRank(b.priority) - _esPriorityRank(a.priority));
  if (note) {
    const pill = energySavingCache.status === "NO_RECOMMENDATION" ? "none" : "high";
    note.innerHTML = `<span class="forecast-pill ${pill}">${recs.length} suggestion(s)</span>`;
  }
  list.innerHTML = recs.map(renderEnergySavingItem).join("");
}

/* Fetches real 5-minute history for the chart's ACTUAL line (never live 10s
   data). For "system" it sums all 9 meters per 5-minute slot — the same
   simultaneous-sum convention as the AI backend's system forecast. */
async function fetchActualForForecast(meterKey, windowSeconds) {
  const end = Date.now();
  const start = end - windowSeconds * 1000;
  const slotMs = 5 * 60 * 1000;
  if (meterKey === "system") {
    const snaps = await Promise.all(
      Array.from({ length: 9 }, (_, i) =>
        firebase.database().ref(`history/pzem_${i + 1}`).orderByKey().startAt(String(Math.floor(start / 1000))).once("value"))
    );
    const sums = {};
    snaps.forEach((snap) => {
      const val = snap.val() || {};
      Object.entries(val).forEach(([k, reading]) => {
        const t = timestampMilliseconds(k);
        if (t < start || t > end) return;
        const p = normalizePowerWatts(reading);
        if (!Number.isFinite(p)) return;
        const slot = Math.floor(t / slotMs) * slotMs;
        sums[slot] = (sums[slot] || 0) + p;
      });
    });
    return Object.entries(sums).map(([t, p]) => ({ t: Number(t), power: p })).sort((a, b) => a.t - b.t);
  }
  const n = Number(meterKey.split("_")[1]);
  const snap = await firebase.database().ref(`history/pzem_${n}`).orderByKey().startAt(String(Math.floor(start / 1000))).once("value");
  const val = snap.val() || {};
  return Object.entries(val)
    .map(([k, reading]) => ({ t: timestampMilliseconds(k), power: normalizePowerWatts(reading) }))
    .filter((p) => p.t <= end && Number.isFinite(p.power))
    .sort((a, b) => a.t - b.t);
}

/* Synthetic seasonal-naive forecast used ONLY in ?forecastDemo=1 mode so the
   chart can be visually verified without a real backend run. Clearly labeled
   as synthetic everywhere it is shown — it is NOT a production forecast. */
function synthForecastFromActual(actual, horizon) {
  if (!actual.length) return null;
  const durMs = (horizon === "24h" ? 86400 : 7 * 86400) * 1000;
  const end = actual[actual.length - 1].t;
  const start = end - durMs;
  const seg = actual.filter((p) => p.t > start);
  if (!seg.length) return null;
  const shift = durMs;
  const ts = [], pw = [], lo = [], hi = [];
  seg.forEach((p) => {
    ts.push(Math.round((p.t + shift) / 1000));
    pw.push(p.power);
    lo.push(Math.max(0, p.power * 0.9));
    hi.push(p.power * 1.1);
  });
  return {
    status: "FORECAST", confidence: "low",
    timestamps: ts, forecast_power_w: pw, lower_bound: lo, upper_bound: hi,
    reason: "Synthetic demo only — not a production forecast."
  };
}

function renderForecastInsufficient(reason) {
  const chart = ensureForecastChart();
  chart.data.datasets.forEach((d) => { d.data = []; });
  resetChartZoom(chart);
  chart.update();
  const note = $("forecastNote");
  const cap = $("forecastCaption");
  note.textContent = "Forecast unavailable";
  const pill = `<span class="forecast-pill none">Insufficient data</span>`;
  cap.innerHTML = `${pill} ${reason || "No valid forecast has been generated yet."}` +
    (FORECAST_DEMO ? ` <em>Demo mode active — add a real /ai/forecast source to show production forecasts.</em>` : "");
}

async function renderForecast() {
  const chart = ensureForecastChart();
  const note = $("forecastNote");
  const cap = $("forecastCaption");
  const meterKey = selectedForecastPzem;
  const hk = horizonKey(forecastHorizon);

  note.textContent = "Loading forecast…";

  let actual = [];
  try {
    actual = await fetchActualForForecast(meterKey, forecastHorizon === "24h" ? 86400 : 7 * 86400);
  } catch (err) {
    console.error("[DASHBOARD] Forecast actual-history load failed", err);
  }

  let rec = latestForecastRecord(meterKey);
  const hasReal = rec && rec.status === "FORECAST" && rec[hk] && rec[hk].status === "FORECAST";

  if (!hasReal) {
    if (FORECAST_DEMO) {
      const synth = synthForecastFromActual(actual, forecastHorizon);
      if (synth) {
        rec = synth;
      } else {
        renderForecastInsufficient("Not enough history to synthesize a demo either.");
        return;
      }
    } else {
      const why = rec && rec[hk] ? rec[hk].reason : (rec ? "Forecast exists but this horizon is unavailable." : null);
      renderForecastInsufficient(why);
      return;
    }
  }

  const h = rec[hk];
  const ts = h.timestamps, pw = h.forecast_power_w, lo = h.lower_bound, hi = h.upper_bound;
  const actualPts = actual.map((p) => ({ x: p.t, y: p.power }));
  const fcPts = ts.map((t, i) => ({ x: t * 1000, y: pw[i] }));
  const loPts = ts.map((t, i) => ({ x: t * 1000, y: lo[i] }));
  const hiPts = ts.map((t, i) => ({ x: t * 1000, y: hi[i] }));

  chart.data.datasets[0].data = hiPts;
  chart.data.datasets[1].data = loPts;
  chart.data.datasets[2].data = actualPts;
  chart.data.datasets[3].data = fcPts;
  resetChartZoom(chart);
  chart.update();

  const conf = h.confidence || (rec.confidence || "low");
  const pillClass = conf === "high" ? "high" : conf === "medium" ? "medium" : "low";
  const pillLabel = FORECAST_DEMO && !hasReal ? "DEMO" : conf;
  note.innerHTML = `<span class="forecast-pill ${pillClass}">${pillLabel} confidence</span>`;
  const meta = rec.anchor_timestamp ? ` · anchor ${new Date(rec.anchor_timestamp * 1000).toLocaleString()}` : "";
  const why = h.reason ? ` · ${h.reason}` : "";
  cap.innerHTML = `Historical 5-min actuals + ${forecastHorizon} forecast for <b>${meterKey === "system" ? "System (all valid meters)" : meterKey.replace("_", " ").toUpperCase()}</b>${meta}${why}` +
    (FORECAST_DEMO && !hasReal ? ` <em>· DEMO SYNTHETIC DATA</em>` : "");
}

function initForecastPanel() {
  loadForecastCache();
  loadEnergySavingCache();
  renderEnergySaving();
  const sel = $("forecastPzem");
  if (sel) {
    sel.addEventListener("change", (e) => {
      selectedForecastPzem = e.target.value;
      renderForecast();
      renderBillPrediction();
    });
  }
  document.querySelectorAll(".forecast-toggle .pzem-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      forecastHorizon = btn.dataset.horizon;
      document.querySelectorAll(".forecast-toggle .pzem-tab").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", String(on));
      });
      renderForecast();
      renderBillPrediction();
    });
  });
  renderForecast();
  renderBillPrediction();
  setInterval(renderForecast, HISTORY_SLOT_MS);
}

function switchModalTab(tabName) {
  document.querySelectorAll(".pzem-tab").forEach((btn) => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".pzem-tab-panel").forEach((panel) => {
    panel.hidden = panel.dataset.panel !== tabName;
  });
}

function renderModalOverview() {
  if (!activeMeterNumber) return;
  const meter = getMeter(activeMeterNumber);
  const isLive = isMeterFresh(meter);

  // Same UI-only zero placeholder as the cards when OFF/stale (Part 1/6) —
  // never "--", never the last stale Firebase reading.
  $("modalPower").innerHTML = isLive ? `${number(normalizePowerWatts(meter), 1)} <small>W</small>` : `0 <small>W</small>`;
  $("modalVoltage").innerHTML = isLive ? `${number(meter.voltage, 1)} <small>V</small>` : `0 <small>V</small>`;
  $("modalCurrent").innerHTML = isLive ? `${number(meter.current, 2)} <small>A</small>` : `0.00 <small>A</small>`;
  $("modalEnergy").innerHTML = isLive ? `${number(meter.energy, 2)} <small>kWh</small>` : `0.00 <small>kWh</small>`;
  $("modalPF").textContent = isLive ? number(meter.pf, 2) : "0.00";

  const { communication, ac } = getThreeStatus(meter);

  const statusPill = $("modalStatusPill");
  statusPill.classList.remove("online", "offline");
  statusPill.classList.add(communication ? "online" : "offline");
  $("modalStatusText").textContent = communication ? "CONNECTED" : "OFFLINE";

  $("ovPzemStatus").textContent = communication ? "CONNECTED" : "OFFLINE";
  $("ovAcStatus").textContent = ac === true ? "AC ON" : ac === false ? "AC OFF" : "UNKNOWN";

  updateOverviewChartData();
}

function updateOverviewChartData() {
  if (!activeMeterNumber) return;
  const chart = ensureOverviewChart();
  const state = getRuntimeState(activeMeterNumber);

  chart.data.labels = state.liveSeries.map((point) => new Date(point.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
  chart.data.datasets[0].data = state.liveSeries.map((point) => point.power);
  chart.update();
}

function formatDuration(ms) {
  const totalMinutes = Math.max(0, Math.round(ms / 60000));
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

/* Reconstructs "meter was reporting continuously" (connectivity) sessions
   from stored history/pzem_N points — purely from GAPS between consecutive
   stored readings, never from a power level. history/pzem_N is written once
   per HISTORY_SLOT_MS (5 min) whenever the firmware has a valid reading; if
   the meter went offline, that slot is simply never written, leaving a gap
   in the timestamps. A gap wider than a couple of slots means the meter
   wasn't reporting during that stretch, so it splits into a new session.
   Also returns totalReportingMs (sum of covered duration across the series)
   so the history tab's "Meter uptime" stat is derived from the same rule as
   the sessions themselves. This has nothing to do with equipment power
   thresholds — it is connectivity only. */
function computeSessionsFromPoints(points) {
  const GAP_TOLERANCE_MS = HISTORY_SLOT_MS * 2; // > 2 missed slots = meter was offline
  const sessions = [];
  let totalReportingMs = 0;

  if (!points.length) return { sessions, totalReportingMs };

  let sessionStart = points[0].t;
  let prev = points[0].t;

  for (let i = 1; i < points.length; i++) {
    const gap = points[i].t - prev;
    if (gap > GAP_TOLERANCE_MS) {
      sessions.push({ start: sessionStart, end: prev });
      totalReportingMs += prev - sessionStart;
      sessionStart = points[i].t;
    }
    prev = points[i].t;
  }

  sessions.push({ start: sessionStart, end: prev });
  totalReportingMs += prev - sessionStart;

  return { sessions, totalReportingMs };
}

/* Reconstructed history sessions are 5-minute-granularity approximations
   (history/pzem_N only stores one reading per slot); live sessions come from
   second-by-second live updates since the page was opened. A live session is
   skipped as a duplicate only when a historical session's start AND end both
   land within one history slot (5 min) of it — otherwise both are kept. */
function mergeSessions(historicalSessions, liveSessions) {
  const merged = [...historicalSessions];

  liveSessions.forEach((liveSession) => {
    const isDuplicate = historicalSessions.some((historySession) =>
      Math.abs(historySession.start - liveSession.start) <= HISTORY_SLOT_MS &&
      Math.abs(historySession.end - liveSession.end) <= HISTORY_SLOT_MS
    );
    if (!isDuplicate) merged.push(liveSession);
  });

  return merged.sort((a, b) => a.start - b.start);
}

function renderModalSessions() {
  if (!activeMeterNumber) return;
  const state = getRuntimeState(activeMeterNumber);

  const currentBlock = $("modalCurrentSession");
  if (state.online && state.sessionStart) {
    currentBlock.innerHTML = `
      <span class="session-live-pill">● LIVE</span>
      <dl class="pzem-overview-grid">
        <div><dt>Reporting since</dt><dd>${new Date(state.sessionStart).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</dd></div>
        <div><dt>Duration</dt><dd>${formatDuration(Date.now() - state.sessionStart)}</dd></div>
      </dl>`;
  } else {
    currentBlock.innerHTML = `<span class="session-live-pill off">● OFF</span>`;
  }

  const list = $("modalSessionList");
  list.replaceChildren();

  const combinedSessions = mergeSessions(state.historicalSessions, state.sessions);

  if (!combinedSessions.length) {
    list.innerHTML = `<p class="dialog-note">No completed sessions found in the selected history range, and none recorded yet since this dashboard was opened.</p>`;
    return;
  }

  [...combinedSessions].reverse().forEach((session) => {
    const row = document.createElement("div");
    row.className = "pzem-session-row";
    const dateLabel = new Date(session.start).toLocaleDateString([], { day: "2-digit", month: "short", year: "numeric" });
    row.innerHTML = `
      <span class="session-date">${dateLabel}</span>
      <span class="session-time">${new Date(session.start).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} → ${new Date(session.end).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
      <span class="session-duration">${formatDuration(session.end - session.start)}</span>`;
    list.appendChild(row);
  });
}

function alertIcon(type) {
  return { highPower: "🔴", highCurrent: "🟠", lowVoltage: "🟠", highVoltage: "🔴", offline: "⚫", info: "🟢", on: "🟢", off: "⚫" }[type] || "🔵";
}

function renderModalAlerts() {
  if (!activeMeterNumber) return;
  const state = getRuntimeState(activeMeterNumber);
  const list = $("modalAlertList");
  list.replaceChildren();

  if (!state.alerts.length) {
    list.innerHTML = `<p class="dialog-note">No alerts recorded yet since this dashboard was opened.</p>`;
    return;
  }

  [...state.alerts].reverse().forEach((alert) => {
    const item = document.createElement("div");
    item.className = "pzem-alert-item";
    item.innerHTML = `
      <span class="alert-icon">${alertIcon(alert.type)}</span>
      <span class="alert-body">
        <b>${alert.label}</b>
        ${alert.detail ? `<small>${alert.detail}</small>` : ""}
      </span>
      <span class="alert-time">${new Date(alert.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>`;
    list.appendChild(item);
  });
}

function historyRangeToWindow(range) {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();

  switch (range) {
    case "1h": return { start: Date.now() - 3600 * 1000, end: Date.now() };
    case "6h": return { start: Date.now() - 6 * 3600 * 1000, end: Date.now() };
    case "12h": return { start: Date.now() - 12 * 3600 * 1000, end: Date.now() };
    case "today": return { start: startOfToday, end: Date.now() };
    case "yesterday": return { start: startOfToday - 24 * 3600 * 1000, end: startOfToday };
    case "30d": return { start: Date.now() - 30 * 24 * 3600 * 1000, end: Date.now() };
    case "7d":
    default: return { start: Date.now() - 7 * 24 * 3600 * 1000, end: Date.now() };
  }
}

/* Reuses the exact same Firebase "history/pzem_N" ref that loadPowerHistory()
   already reads for the main chart — same source, filtered to one meter. */
async function renderModalHistory(range) {
  if (!activeMeterNumber) return;
  const requestId = ++modalHistoryRequestId;
  const meterN = activeMeterNumber;
  const note = $("modalHistoryMessage");
  const chart = ensureModalHistoryChart();
  const { start, end } = historyRangeToWindow(range);

  note.textContent = "Loading historical data…";

  try {
    const snapshot = await firebase.database()
      .ref(`history/pzem_${meterN}`)
      .orderByKey()
      .startAt(String(Math.floor(start / 1000)))
      .once("value");

    if (requestId !== modalHistoryRequestId || meterN !== activeMeterNumber) return;

    const points = Object.entries(snapshot.val() || {})
      .map(([timestamp, data]) => ({ t: timestampMilliseconds(timestamp), power: normalizePowerWatts(data), energy: Number(data.energy ?? NaN) }))
      .filter((point) => point.t <= end)
      .sort((a, b) => a.t - b.t);

    if (!points.length) {
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    resetChartZoom(chart);
    chart.update();
    note.textContent = "No stored history for this meter in the selected range.";
      $("modalHistorySummary").replaceChildren();
      getRuntimeState(meterN).historicalSessions = [];
      if (meterN === activeMeterNumber) renderModalSessions();
      return;
    }

    chart.data.labels = points.map((point) => new Date(point.t).toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }));
    chart.data.datasets[0].data = points.map((point) => point.power);
    resetChartZoom(chart); // new range/meter = fresh axis
    chart.update();

    const powers = points.map((point) => point.power).filter(Number.isFinite);
    const peakPower = powers.length ? Math.max(...powers) : 0;
    const avgPower = powers.length ? powers.reduce((sum, value) => sum + value, 0) / powers.length : 0;

    // Part 11: identify suspiciously large stored values instead of
    // silently hiding/clamping them. A household PZEM circuit realistically
    // tops out in the low thousands of watts, so anything at/above 50,000 W
    // almost certainly means a pre-object-format legacy row (or a genuine
    // upstream sensor fault) rather than a frontend parsing bug — but it's
    // still shown as-is; this just makes it traceable in the console.
    const suspicious = points.filter((point) => point.power >= 50000);
    if (suspicious.length) {
      console.warn(`[HISTORY DEBUG] PZEM ${meterN} found ${suspicious.length} suspiciously large stored value(s) (>=50000 W), included as-is — example rawPower=${suspicious[0].power}`);
    }
    console.log(`[HISTORY DEBUG] PZEM ${meterN} points=${points.length} peak=${peakPower.toFixed(2)} average=${avgPower.toFixed(2)}`);

    // Reconstructs meter connectivity ("was this PZEM actually reporting")
    // sessions for this range straight from the stored 5-minute readings, so
    // the session-history list survives a page refresh instead of only
    // reflecting what happened since this tab was opened. Also drives
    // "Meter uptime" from the same gap-based rule the sessions themselves
    // use. This is connectivity only — not an equipment ON/OFF power reading.
    const { sessions: reconstructedSessions, totalReportingMs } =
      computeSessionsFromPoints(points);
    getRuntimeState(meterN).historicalSessions = reconstructedSessions;
    if (meterN === activeMeterNumber) renderModalSessions();

    const liveMeter = getMeter(meterN);
    note.textContent = `${points.length} readings · ${range === "1h" ? "1 hour" : range === "6h" ? "6 hours" : range === "12h" ? "12 hours" : range === "today" ? "today" : range === "yesterday" ? "yesterday" : range === "30d" ? "30 days" : "7 days"}`;

    $("modalHistorySummary").innerHTML = `
      <div><span>Today's energy</span><strong>${number(liveMeter.energy, 2)} kWh</strong></div>
      <div><span>Peak power</span><strong>${number(peakPower, 1)} W</strong></div>
      <div><span>Average power</span><strong>${number(avgPower, 1)} W</strong></div>
      <div><span>Meter uptime</span><strong>${formatDuration(totalReportingMs)}</strong></div>`;
  } catch (error) {
    console.error(error);
    if (requestId !== modalHistoryRequestId) return;
    note.textContent = "Historical data unavailable right now.";
  }
}

function refreshOpenModalLiveParts() {
  renderModalOverview();
  renderModalSessions();
  renderModalAlerts();
}

function openMeterDetail(n) {
  activeMeterNumber = n;

  $("dialogMeterName").textContent = `PZEM ${n}`;
  $("dialogMeterId").textContent = `PZEM_${n}`;

  switchModalTab("overview");
  renderModalOverview();
  renderModalSessions();
  renderModalAlerts();
  renderModalHistory($("modalHistoryRange").value);

  const dialog = $("meterDialog");
  if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
}

function closeMeterDetail() {
  const dialog = $("meterDialog");
  if (dialog.open) dialog.close();
  activeMeterNumber = null;
}

/* Delegated listener: cards are re-rendered on every live update, so a single
   listener on the container (rather than one per card) keeps working forever. */
dashboard.addEventListener("click", (event) => {
  const card = event.target.closest(".meter-card");
  if (card && card.dataset.meterNumber) openMeterDetail(Number(card.dataset.meterNumber));
});

$("closeDialog").addEventListener("click", closeMeterDetail);

/* Clicking the <dialog> backdrop targets the dialog element itself (not its
   content), so this closes on outside-click without extra markup. */
$("meterDialog").addEventListener("click", (event) => {
  if (event.target === $("meterDialog")) closeMeterDetail();
});

/* <dialog> already closes natively on ESC and fires "cancel" — just reset state. */
$("meterDialog").addEventListener("cancel", () => { activeMeterNumber = null; });
$("meterDialog").addEventListener("close", () => { activeMeterNumber = null; });

/* -------------------------------------------------------------------------
   INFO / ABOUT PANEL (Stage 16 Info button)
   Uses a native <dialog>, so Escape, focus trap and backdrop are handled for
   us; we only wire the button, the close button and outside-click.
   ------------------------------------------------------------------------- */
(function setupInfoPanel() {
  const infoButton = $("infoButton");
  const infoDialog = $("infoDialog");
  if (!infoButton || !infoDialog) return;

  function openInfo() {
    if (typeof infoDialog.showModal === "function" && !infoDialog.open) {
      infoDialog.showModal();
      const body = infoDialog.querySelector(".info-body");
      if (body) body.scrollTop = 0;
    }
  }
  function closeInfo() {
    if (infoDialog.open) infoDialog.close();
  }

  infoButton.addEventListener("click", openInfo);
  const closeBtn = infoDialog.querySelector(".info-close");
  if (closeBtn) closeBtn.addEventListener("click", closeInfo);
  infoDialog.addEventListener("click", (event) => {
    if (event.target === infoDialog) closeInfo();
  });
})();

document.querySelectorAll(".pzem-tab").forEach((btn) => {
  btn.addEventListener("click", () => switchModalTab(btn.dataset.tab));
});

$("modalHistoryRange").addEventListener("change", (event) => renderModalHistory(event.target.value));

/* =========================================================================
   FRESHNESS RECHECK TIMER
   -------------------------------------------------------------------------
   Firebase's "value" listener (attachLiveListener()) only fires when
   meters/pzem_N actually changes. If a PZEM stops sending live updates, the
   firmware stops writing to it — nothing changes in Firebase, so the
   listener stays completely silent and nothing above would ever re-render.
   Left alone, a card that went stale would keep showing LIVE until some
   OTHER meter's update happened to trigger a re-render.
   This local timer does not read Firebase and does not reload the page — it
   just re-evaluates isMeterFresh() against the wall clock on the data
   already cached in metersData, so LIVE -> OFF (and OFF -> LIVE once real
   updates resume) shows up promptly and automatically. Interval is well
   under FRESHNESS_TIMEOUT_MS so a transition is never missed by more than a
   few seconds. */
function refreshFreshnessOnly() {
  if (!Object.keys(metersData).length) return;

  // Update only the parts that depend on freshness: status pills, power values, summary KPIs
  const entries = Array.from({ length: 9 }, (_, index) => [`pzem_${index + 1}`, getMeter(index + 1)]);

  entries.forEach(([id, meter], index) => {
    const card = dashboard.children[index];
    if (!card) return;

    const isLive = isMeterFresh(meter);
    const power = isLive ? normalizePowerWatts(meter) : 0;
    const { communication, ac } = getThreeStatus(meter);

    card.querySelector(".meter-power strong").textContent = isLive ? number(power, 1) : "0";
    card.querySelector(".voltage").textContent = isLive ? `${number(meter.voltage, 1)} V` : "0 V";
    card.querySelector(".current").textContent = isLive ? `${number(meter.current, 2)} A` : "0.00 A";
    card.querySelector(".energy").textContent = isLive ? `${number(meter.energy, 2)} kWh` : "0.00 kWh";
    card.querySelector(".pf").textContent = isLive ? number(meter.pf, 2) : "0.00";
    const freqCell = card.querySelector(".freq");
    if (freqCell) freqCell.textContent = isLive ? `${number(meter.frequency, 1)} Hz` : "0.00 Hz";
    card.querySelector(".power-track span").style.width = `${isLive ? Math.min((power / maxPower) * 100, 100) : 0}%`;
    updatePowerViz(card, power, isLive);

    const commStatus = card.querySelector(".comm-status");
    commStatus.className = "meter-status comm-status " + (communication ? "online" : "offline");
    commStatus.querySelector("b").textContent = communication ? "CONNECTED" : "OFFLINE";

    const acStatus = card.querySelector(".ac-status");
    acStatus.className = "meter-status ac-status " + (ac === true ? "online" : ac === false ? "off" : "unknown");
    acStatus.querySelector("b").textContent = ac === true ? "AC ON" : ac === false ? "AC OFF" : "UNKNOWN";

    // Update offline class on card
    if (isLive) {
      card.classList.remove("offline");
    } else {
      card.classList.add("offline");
    }
    
    // STAGE 4: Maintain emergency-fault class based on alert cache
    const alert = meterAlerts[`pzem_${index + 1}`];
    if (alert && alert.severity === "EMERGENCY") {
      card.classList.add("emergency-fault");
      card.classList.remove("offline");
      // Re-apply fault info display
      const faultInfo = card.querySelector(".fault-info");
      if (faultInfo) {
        faultInfo.innerHTML = `
          <div class="fault-badge">${alert.type.replace(/_/g, ' ')}</div>
          <div class="fault-details">
            <span class="fault-severity">[${alert.severity}]</span>
            ${alert.measured_value !== undefined ? `<span class="fault-value">${alert.measured_value} ${getUnitForType(alert.type)}</span>` : ""}
            ${alert.timestamp ? `<span class="fault-time">${new Date(alert.timestamp * 1000).toLocaleTimeString()}</span>` : ""}
          </div>
        `;
      }
    } else if (alert && alert.severity === "WARNING") {
      // WARNING severity: add subtle indicator but no red blink
      const faultInfo = card.querySelector(".fault-info");
      if (faultInfo) {
        faultInfo.innerHTML = `
          <div class="fault-badge warning-badge">${alert.type.replace(/_/g, ' ')}</div>
          <div class="fault-details"><span class="fault-severity">[${alert.severity}]</span></div>
        `;
      }
    } else {
      // No active emergency or warning alert — remove fault classes
      card.classList.remove("emergency-fault");
    }

    // STAGE 6: Maintain AI status area based on current AI state cache
    const meterAiState = meterAIStates[`pzem_${index + 1}`];
    const aiStatus = card.querySelector(".ai-status");
    if (aiStatus) {
      if (meterAiState && meterAiState.type && meterAiState.type !== "anomaly" && meterAiState.type !== "fault") {
        // No active AI anomaly/fault — show NORMAL
        aiStatus.className = "ai-status ai-status-normal";
        aiStatus.innerHTML = `<span class="ai-status-pill">NORMAL</span><span class="ai-status-details">AI monitoring active</span>`;
      } else if (meterAiState && meterAiState.type === "anomaly") {
        // Active anomaly
        const severity = meterAiState.severity || "NORMAL";
        const scoreDisplay = meterAiState.score !== null ? ` (score: ${Number(meterAiState.score).toFixed(2)})` : "";
        aiStatus.className = `ai-status ai-status-anomaly`;
        aiStatus.innerHTML = `<span class="ai-status-pill">ANOMALY</span><span class="ai-status-details"> ${meterAiState.label}${scoreDisplay} [${severity}]</span><span class="ai-status-details"> ${new Date(meterAiState.timestamp * 1000).toLocaleTimeString()}</span>`;
      } else if (meterAiState && meterAiState.type === "fault") {
        // Active fault
        const severity = String(meterAiState.severity || "NORMAL");
        const faultType = String(meterAiState.faultType || "unknown");
        const valueDisplay = meterAiState.measuredValue !== undefined ? ` — ${meterAiState.measuredValue}` : "";
        aiStatus.className = `ai-status ai-status-${severity.toLowerCase() === "emergency" ? "emergency" : severity.toLowerCase() === "warning" ? "warning" : "anomaly"}`;
        aiStatus.innerHTML = `<span class="ai-status-pill">${severity}</span><span class="ai-status-details"> ${faultType}${valueDisplay} [${severity}]</span><span class="ai-status-details"> ${new Date((meterAiState.timestamp || 0) * 1000).toLocaleTimeString()}</span>`;
      } else {
        // No AI data yet
        aiStatus.className = "ai-status ai-status-no-data";
        aiStatus.innerHTML = `<span class="ai-status-pill">NO AI DATA</span><span class="ai-status-details">AI monitoring not yet active</span>`;
      }
    }
  });

  const online = entries.filter(([, meter]) => isMeterFresh(meter));
  const totalPower = online.reduce((sum, [, meter]) => sum + normalizePowerWatts(meter), 0);
  const totalEnergy = online.reduce((sum, [, meter]) => sum + Number(meter.energy || 0), 0);
  const averageVoltage = online.length
    ? online.reduce((sum, [, meter]) => sum + Number(meter.voltage || 0), 0) / online.length
    : 0;

  $("totalPower").innerHTML = `${number(totalPower, 1)} <small>W</small>`;
  $("totalEnergy").innerHTML = `${number(totalEnergy, 2)} <small>kWh</small>`;
  $("onlineMeters").innerHTML = `${online.length} <small>/ 9</small>`;
  $("averageVoltage").innerHTML = `${number(averageVoltage, 1)} <small>V</small>`;

  lastLiveSummary.freshMeters = online.length;
  lastLiveSummary.power = totalPower;
  lastLiveSummary.voltage = averageVoltage;

  updateSystemStatus();
}

setInterval(() => {
  if (!Object.keys(metersData).length) return;
  refreshFreshnessOnly();
  updateFrequency();
  updateLivePower();
  trackPzemRuntimeState();
}, 5000);