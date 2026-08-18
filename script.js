const firebaseConfig = {
  apiKey: "AIzaSyDCQJgHdIb5CkGRhAPOI-ynVfHdNFSo6bs",
  authDomain: "smart-energy-monitoring-5a2a4.firebaseapp.com",
  databaseURL: "https://smart-energy-monitoring-5a2a4-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "smart-energy-monitoring-5a2a4",
  storageBucket: "smart-energy-monitoring-5a2a4.firebasestorage.app",
  messagingSenderId: "275361980378",
  appId: "1:275361980378:web:c9f9ae5742be617e0aaae3"
};

firebase.initializeApp(firebaseConfig);

const maxPower = 3000;
const colors = ["#2578ff", "#7b4cf6", "#f28d2f", "#13b887", "#e45f92", "#16a7d9", "#a269d8", "#d88323", "#2c9e72"];
const $ = (id) => document.getElementById(id);

const dashboard = $("dashboardContent");
const meterTemplate = $("meterTemplate");
const unitRate = $("unitRate");
let metersData = {};
let powerHistoryMode = false;
let historyRequestId = 0;

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

// Cache of the last computed live-summary values, written by renderDashboard()
// and read by updateFrequency() (which always runs immediately afterward) to
// print one combined [LIVE SUMMARY] diagnostic line per update.
let lastLiveSummary = { freshMeters: 0, power: 0, voltage: 0, frequency: 0 };

function number(value, decimals = 0) {
  return Number(value || 0).toFixed(decimals);
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

/* Add common frequency card, frequency graph and power range selector */
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
      <option value="1d">1 day</option>
      <option value="1w">1 week</option>
      <option value="1m">1 month</option>
    </select>
  `;

  if (oldNote) controls.appendChild(oldNote);
  powerHeading.appendChild(controls);

  const frequencyPanel = document.createElement("section");
  frequencyPanel.className = "chart-panel";
  frequencyPanel.innerHTML = `
    <div class="chart-heading">
      <div>
        <p class="eyebrow">COMMON FREQUENCY MONITORING</p>
        <h2>Live common frequency</h2>
      </div>
      <span class="chart-note">Latest 100 updates</span>
    </div>
    <div class="chart-wrap">
      <canvas id="frequencyChart"></canvas>
    </div>
  `;

 /* Place REAL-TIME POWER and COMMON FREQUENCY MONITORING side-by-side (layout only) */
  const chartsRow = document.createElement("div");
  chartsRow.className = "charts-row";

  const powerPanel = document.querySelector(".chart-panel"); // existing "Real-time power" panel, untouched
  powerPanel.parentNode.insertBefore(chartsRow, powerPanel);
  chartsRow.appendChild(powerPanel);
  chartsRow.appendChild(frequencyPanel);
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

/* Computes the power (W) axis max + tick step from the highest value
   currently on the chart. Genuinely dynamic and unbounded — a 300 W floor
   for near-zero readings, and no ceiling: 5 kW isn't a hard cap, it just
   naturally falls out of niceNumber() for typical highs around there, and
   higher values (6 kW, 8 kW, 10 kW, ...) round up the same way. */
function computePowerAxisRange(highestPowerWatts) {
  const safeHighest = Number.isFinite(highestPowerWatts) && highestPowerWatts > 0 ? highestPowerWatts : 0;
  const max = Math.max(300, niceNumber(safeHighest));
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

const powerChart = new Chart($("powerChart"), {
  type: "line",
  data: { labels: [], datasets: createDatasets() },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { boxWidth: 9, usePointStyle: true, pointStyle: "circle", padding: 14 } },
      tooltip: {
        callbacks: {
          label: (context) => `${context.dataset.label}: ${formatPowerLabel(Number(context.parsed.y || 0))}`
        }
      }
    },
    scales: {
      x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
      y: {
        beginAtZero: true,
        min: 0,
        max: 300,
        title: { display: true, text: "Power" },
        ticks: { stepSize: 60, callback: (value) => formatPowerLabel(Number(value)) }}
    }
  }
});

const frequencyChart = new Chart($("frequencyChart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "Common frequency",
      data: [],
      borderColor: "#2578ff",
      backgroundColor: "rgba(37, 120, 255, .15)",
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
      legend: { labels: { boxWidth: 9, usePointStyle: true, pointStyle: "circle" } },
      tooltip: {
        callbacks: {
          label: (context) => `Common frequency: ${Number(context.parsed.y || 0).toFixed(2)} Hz`
        }
      }
    },
    scales: {
      x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
      y: {
  min: 48,
  max: 52,
  title: {
    display: true,
    text: "Frequency (Hz)"
  },
  ticks: {
    stepSize: 1,
    callback: (value) => `${value} Hz`
  },
  grid: {
    color: "rgba(101, 115, 136, 0.15)"
  }
}
    }
  }
});

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

  frequencyChart.data.labels.push(label);
  frequencyChart.data.datasets[0].data.push(common);

  if (frequencyChart.data.labels.length > 100) {
    frequencyChart.data.labels.shift();
    frequencyChart.data.datasets[0].data.shift();
  }

  frequencyChart.update();

  // Combined live-summary diagnostic: total power/voltage were computed a
  // moment earlier in renderDashboard() (always called immediately before
  // this function, both on real Firebase updates and on the freshness
  // recheck timer) and cached in lastLiveSummary.
  lastLiveSummary.frequency = common;
  console.log(`[LIVE SUMMARY] freshMeters=${lastLiveSummary.freshMeters} power=${lastLiveSummary.power.toFixed(1)} voltage=${lastLiveSummary.voltage.toFixed(1)} frequency=${lastLiveSummary.frequency.toFixed(2)}`);
}
function updatePowerYAxis() {
  const values = powerChart.data.datasets
    .flatMap((dataset) => dataset.data)
    .map(Number)
    .filter((value) => Number.isFinite(value) && value >= 0);

  const highestPower = values.length ? Math.max(...values) : 0;
  const { max, step } = computePowerAxisRange(highestPower);

  powerChart.options.scales.y.min = 0;
  powerChart.options.scales.y.max = max;
  powerChart.options.scales.y.ticks.stepSize = step;

  // Temporary safe diagnostics: per-meter dataset range + the resulting
  // shared axis max, so a mis-scaled axis can be traced back to the exact
  // value that caused it.
  powerChart.data.datasets.forEach((dataset, index) => {
    const nums = dataset.data.map(Number).filter(Number.isFinite);
    if (!nums.length) return;
    console.log(`[CHART DEBUG] PZEM ${index + 1} datasetMin=${Math.min(...nums)} datasetMax=${Math.max(...nums)} axisMax=${max}`);
  });
}
function updateLivePower() {
  if (powerHistoryMode) return;

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

function updateBill(entries) {
  const rate = Number(unitRate.value || 0);
  const totalUnits = entries.reduce((sum, [, meter]) => sum + Number(meter.energy || 0), 0);

  $("billTotalUnits").innerHTML = `${number(totalUnits, 2)} <small>kWh</small>`;
  $("billTotalCost").textContent = rate
    ? new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(totalUnits * rate)
    : "—";

  $("billRateText").textContent = rate ? `At ₹${rate.toFixed(2)} per unit` : "Select your price per unit";
  $("billMeterRows").replaceChildren();

  entries.forEach(([id, meter]) => {
    const energy = Number(meter.energy || 0);
    const row = document.createElement("tr");

    row.innerHTML = `
      <td><b>${id.toUpperCase()}</b></td>
      <td>${number(energy, 2)} kWh</td>
      <td>${rate ? `₹${number(energy * rate, 2)}` : "Select price"}</td>
    `;

    $("billMeterRows").appendChild(row);
  });
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

    const status = card.querySelector(".meter-status");
    status.classList.add(isLive ? "online" : "offline");
    status.querySelector("b").textContent = isLive ? "LIVE" : "OFF";

    card.querySelector(".meter-card").dataset.meterNumber = String(index + 1); /* enables click-to-open popup */

    dashboard.appendChild(card);

    // Temporary safe diagnostics: freshness age + resulting status only,
    // never any credentials/tokens.
    const age = meterAgeMs(meter);
    console.log(`[DASHBOARD FRESHNESS] PZEM ${index + 1} age=${age === null ? "n/a" : age + "ms"} status=${isLive ? "LIVE" : "OFF"}`);
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

  // Cached for the [LIVE SUMMARY] diagnostic printed by updateFrequency(),
  // which always runs immediately after this function.
  lastLiveSummary.freshMeters = online.length;
  lastLiveSummary.power = totalPower;
  lastLiveSummary.voltage = averageVoltage;

  updateBill(online);
}

function timestampMilliseconds(timestamp) {
  const value = Number(timestamp);
  return String(timestamp).length > 10 ? value : value * 1000;
}

async function loadPowerHistory(range) {
  const note = document.querySelector(".chart-panel .chart-note");
  const requestId = ++historyRequestId;
  const hours = { "1d": 24, "1w": 168, "1m": 720 }[range];
  const start = Math.floor(Date.now() / 1000) - hours * 3600;

  powerHistoryMode = true;
  note.textContent = "Loading history...";

  try {
    const snapshots = await Promise.all(
      Array.from({ length: 9 }, (_, index) =>
        firebase.database()
          .ref(`history/pzem_${index + 1}`)
          .orderByKey()
          .startAt(String(start))
          .once("value")
      )
    );

    if (requestId !== historyRequestId) return;

    const timeline = new Map();

    snapshots.forEach((snapshot, meterIndex) => {
      Object.entries(snapshot.val() || {}).forEach(([timestamp, data]) => {
        const time = timestampMilliseconds(timestamp);

        if (!timeline.has(time)) timeline.set(time, Array(9).fill(null));
        timeline.get(time)[meterIndex] = normalizePowerWatts(data);
      });
    });

    const points = [...timeline.entries()].sort(([a], [b]) => a - b);

    if (!points.length) {
      powerHistoryMode = false;
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
    updatePowerYAxis();
    powerChart.update();

    note.textContent = `${points.length} readings · ${range === "1d" ? "1 day" : range === "1w" ? "1 week" : "1 month"}`;
  } catch (error) {
    console.error(error);
    powerHistoryMode = false;
    note.textContent = "History unavailable — showing live data";
  }
}

function useLiveData(data) {
  metersData = data || {};

  renderDashboard();
  updateFrequency();
  updateLivePower();

  $("connectionStatus").className = "connection-pill online";
  $("connectionStatus").innerHTML = "<span></span> System online";
  $("lastUpdated").textContent = `Last synchronised ${new Date().toLocaleTimeString()}`;

  trackPzemRuntimeState(); /* isolated: powers the new per-meter popup only, does not affect anything above */
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

firebase.auth().onAuthStateChanged((user) => {
  if (user) attachLiveListener();
});

$("powerRange").addEventListener("change", (event) => {
  loadPowerHistory(event.target.value);
});

unitRate.addEventListener("change", () => renderDashboard());

$("themeToggle").addEventListener("click", () => {
  document.body.classList.toggle("dark");
});

/* Formats a unix-seconds-or-ms timestamp as separate Date/Time strings in
   Asia/Kolkata, matching the RTDB contract (history/pzem_N/<timestamp>
   stores no date/time fields — they're derived from the key). */
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
loadPowerHistory("1d");

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

    // Only push a real sample while the meter is actually fresh; push null
    // while OFF so the per-meter popup chart shows a genuine gap instead of
    // a flat line at the last stale reading.
    state.liveSeries.push({ t: now, power: isLive ? power : null });
    if (state.liveSeries.length > 30) state.liveSeries.shift();

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
  overviewChart = new Chart($("overviewLiveChart"), {
    type: "line",
    data: { labels: [], datasets: [{ label: "Power", data: [], borderColor: "#2578ff", backgroundColor: "rgba(37,120,255,.15)", borderWidth: 2, tension: 0.35, pointRadius: 0, fill: true, spanGaps: true }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 6 } },
        y: { beginAtZero: true, title: { display: true, text: "Power (W)" } }
      }
    }
  });
  return overviewChart;
}

function ensureModalHistoryChart() {
  if (modalHistoryChart) return modalHistoryChart;
  modalHistoryChart = new Chart($("modalHistoryChart"), {
    type: "line",
    data: { labels: [], datasets: [{ label: "Power", data: [], borderColor: "#7b4cf6", backgroundColor: "rgba(123,76,246,.13)", borderWidth: 2, tension: 0.3, pointRadius: 0, fill: true, spanGaps: true }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
        y: { beginAtZero: true, title: { display: true, text: "Power (W)" } }
      }
    }
  });
  return modalHistoryChart;
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

  const statusPill = $("modalStatusPill");
  statusPill.classList.remove("online", "offline");
  statusPill.classList.add(isLive ? "online" : "offline");
  $("modalStatusText").textContent = isLive ? "LIVE" : "OFF";

  $("ovPzemStatus").textContent = isLive ? "LIVE" : "OFF — no fresh reading";
  const espOnline = $("connectionStatus").classList.contains("online");
  $("ovEspStatus").textContent = espOnline ? "Connected" : "Disconnected";
  $("ovLastUpdated").textContent = $("lastUpdated").textContent.replace("Last synchronised ", "") || "—";

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
      chart.update();
      note.textContent = "No stored history for this meter in the selected range.";
      $("modalHistorySummary").replaceChildren();
      getRuntimeState(meterN).historicalSessions = [];
      if (meterN === activeMeterNumber) renderModalSessions();
      return;
    }

    chart.data.labels = points.map((point) => new Date(point.t).toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }));
    chart.data.datasets[0].data = points.map((point) => point.power);
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
setInterval(() => {
  if (!Object.keys(metersData).length) return;
  renderDashboard();
  updateFrequency();
  updateLivePower();
  trackPzemRuntimeState();
}, 5000);