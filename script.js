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
    spanGaps: true
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
  min: 40,
  max: 60,
  title: {
    display: true,
    text: "Frequency (Hz)"
  },
  ticks: {
    stepSize: 2,
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
  const values = [];

  for (let i = 1; i <= 9; i++) {
    const value = Number(getMeter(i).frequency);
    if (Number.isFinite(value) && value > 0) values.push(value);
  }

  const common = values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;

  $("commonFrequency").innerHTML = `${number(common, 2)} <small>Hz</small>`;
  $("frequencyCaption").textContent = values.length
    ? `${values.length} live meters · ${Math.min(...values).toFixed(2)}–${Math.max(...values).toFixed(2)} Hz`
    : "Waiting for live frequency data";

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
    dataset.data.push(Number(getMeter(index + 1).power || 0));
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
    const power = Number(meter.power || 0);
    const isOnline = Object.keys(meter).length > 0;

    card.querySelector(".meter-number").textContent = String(index + 1).padStart(2, "0");
    card.querySelector(".meter-name").textContent = `PZEM ${index + 1}`;
    card.querySelector(".meter-id").textContent = id.toUpperCase();
    card.querySelector(".meter-power strong").textContent = number(power, 1);
    card.querySelector(".voltage").textContent = `${number(meter.voltage, 1)} V`;
    card.querySelector(".current").textContent = `${number(meter.current, 2)} A`;
    card.querySelector(".energy").textContent = `${number(meter.energy, 2)} kWh`;
    card.querySelector(".pf").textContent = number(meter.pf, 2);
    card.querySelector(".power-track span").style.width = `${Math.min((power / maxPower) * 100, 100)}%`;

    const status = card.querySelector(".meter-status");
    status.classList.add(isOnline ? "online" : "offline");
    status.querySelector("b").textContent = isOnline ? "Live" : "Waiting";

    card.querySelector(".meter-card").dataset.meterNumber = String(index + 1); /* enables click-to-open popup */

    dashboard.appendChild(card);
  });

  const online = entries.filter(([, meter]) => Object.keys(meter).length > 0);
  const totalPower = online.reduce((sum, [, meter]) => sum + Number(meter.power || 0), 0);
  const totalEnergy = online.reduce((sum, [, meter]) => sum + Number(meter.energy || 0), 0);
  const averageVoltage = online.length
    ? online.reduce((sum, [, meter]) => sum + Number(meter.voltage || 0), 0) / online.length
    : 0;

  $("totalPower").innerHTML = `${number(totalPower, 1)} <small>W</small>`;
  $("totalEnergy").innerHTML = `${number(totalEnergy, 2)} <small>kWh</small>`;
  $("onlineMeters").innerHTML = `${online.length} <small>/ 9</small>`;
  $("averageVoltage").innerHTML = `${number(averageVoltage, 1)} <small>V</small>`;
  $("meterCount").textContent = "9 meters";

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
        timeline.get(time)[meterIndex] = Number(data.power ?? data);
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

// Hysteresis band for equipment ON/OFF detection (NOT the same as PZEM
// online/offline — a PZEM can be online while the equipment it measures is
// idle). While OFF, power must exceed ON_POWER_THRESHOLD to switch ON; while
// ON, power must drop below OFF_POWER_THRESHOLD to switch OFF. Power readings
// between the two thresholds preserve whatever state was already active, so
// noise around a single cutoff can't cause rapid ON/OFF flapping.
const ON_POWER_THRESHOLD = 5;        // W — OFF state switches to ON above this
const OFF_POWER_THRESHOLD = 3;       // W — ON state switches to OFF below this

/* THE single authoritative ON/OFF calculation. Every place in this file that
   needs to know whether equipment is "on" — the live tracker below, and the
   Firebase-history reconstruction used for the session-history list — calls
   this same function with the same two constants above. Nothing else in
   script.js independently decides ON/OFF from power. */
function nextEquipmentState(currentIsOn, power) {
  return currentIsOn ? !(power < OFF_POWER_THRESHOLD) : power > ON_POWER_THRESHOLD;
}

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
    meterRuntimeState[n] = { online: false, isOn: false, sessionStart: null, sessions: [], historicalSessions: [], alerts: [], liveSeries: [], flags: {} };
  }
  return meterRuntimeState[n];
}

function pushAlert(state, type, label, detail) {
  state.alerts.push({ type, label, detail, time: Date.now() });
  if (state.alerts.length > 40) state.alerts.shift();
}

/* Reads the same live meter data the dashboard cards already use and derives
   ON/OFF sessions + threshold alerts locally. Does not fetch anything new. */
function trackPzemRuntimeState() {
  const now = Date.now();

  for (let i = 1; i <= 9; i++) {
    const meter = getMeter(i);
    const state = getRuntimeState(i);
    const isOnline = Object.keys(meter).length > 0;
    const power = Number(meter.power || 0);
    const voltage = Number(meter.voltage || 0);
    const current = Number(meter.current || 0);

    state.liveSeries.push({ t: now, power });
    if (state.liveSeries.length > 30) state.liveSeries.shift();

    // Equipment ON/OFF is only re-evaluated while the PZEM is actually
    // reporting. If the meter drops offline, we don't know the equipment's
    // real state from a missing/zero reading, so the last known ON/OFF state
    // is preserved rather than being misread as "OFF". This runs on every
    // Firebase "meters" update (useLiveData() -> trackPzemRuntimeState()),
    // for every PZEM 1-9, using the one shared nextEquipmentState() rule.
    if (isOnline) {
      const isOn = nextEquipmentState(state.isOn, power);

      if (isOn !== state.isOn) {
        // Safe debug output: power/on-off state only, never credentials.
        console.log(`[PZEM ${i} STATE] power=${power.toFixed(1)}W previous=${state.isOn ? "ON" : "OFF"} new=${isOn ? "ON" : "OFF"}`);
      }

      if (isOn && !state.isOn) {
        state.sessionStart = now;
        pushAlert(state, "on", "Equipment Started", "");
      } else if (!isOn && state.isOn) {
        if (state.sessionStart) {
          state.sessions.push({ start: state.sessionStart, end: now });
          if (state.sessions.length > 60) state.sessions.shift();
        }
        state.sessionStart = null;
        pushAlert(state, "off", "Equipment Stopped", "");
      }
      state.isOn = isOn;
    }

    if (isOnline && !state.online) pushAlert(state, "info", "PZEM Online", "");
    if (!isOnline && state.online) pushAlert(state, "offline", "PZEM Offline", "Meter stopped reporting");
    state.online = isOnline;

    if (isOnline) {
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
  const isOnline = Object.keys(meter).length > 0;

  $("modalPower").innerHTML = `${number(meter.power, 1)} <small>W</small>`;
  $("modalVoltage").innerHTML = `${number(meter.voltage, 1)} <small>V</small>`;
  $("modalCurrent").innerHTML = `${number(meter.current, 2)} <small>A</small>`;
  $("modalEnergy").innerHTML = `${number(meter.energy, 2)} <small>kWh</small>`;
  $("modalPF").textContent = number(meter.pf, 2);

  const statusPill = $("modalStatusPill");
  statusPill.classList.remove("online", "offline");
  statusPill.classList.add(isOnline ? "online" : "offline");
  $("modalStatusText").textContent = isOnline ? "Online" : "Offline";

  $("ovPzemStatus").textContent = isOnline ? "Live" : "Waiting for data";
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

/* Applies the SAME nextEquipmentState() rule used by trackPzemRuntimeState(),
   but over an ordered series of {t, power} points instead of one live sample
   at a time. Used to reconstruct completed ON/OFF sessions from Firebase
   history/pzem_N, so session history survives a page refresh instead of
   living only in browser memory. Also returns totalOnMs (sum of ON duration
   across the series) so the history tab's "Total runtime" stat is derived
   from the same rule as the sessions themselves. */
function computeSessionsFromPoints(points) {
  const sessions = [];
  let isOn = false;
  let sessionStart = null;
  let totalOnMs = 0;

  for (const point of points) {
    const wasOn = isOn;
    isOn = nextEquipmentState(isOn, point.power);

    if (isOn && !wasOn) {
      sessionStart = point.t;
    } else if (!isOn && wasOn && sessionStart !== null) {
      sessions.push({ start: sessionStart, end: point.t });
      totalOnMs += point.t - sessionStart;
      sessionStart = null;
    }
  }

  // Still ON at the end of the window: count elapsed time up to the last
  // sample, but leave the session itself open (no "end" yet) — the live
  // in-memory session covers that ongoing period more precisely.
  if (isOn && sessionStart !== null && points.length) {
    totalOnMs += points[points.length - 1].t - sessionStart;
  }

  return { sessions, totalOnMs };
}

/* Reconstructed history sessions are 5-minute-granularity approximations
   (history/pzem_N only stores one reading per slot); live sessions come from
   second-by-second live updates since the page was opened. A live session is
   skipped as a duplicate only when a historical session's start AND end both
   land within one history slot (5 min) of it — otherwise both are kept. */
function mergeSessions(historicalSessions, liveSessions) {
  const HISTORY_SLOT_MS = 5 * 60 * 1000;
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
  if (state.isOn && state.sessionStart) {
    currentBlock.innerHTML = `
      <span class="session-live-pill">● ON</span>
      <dl class="pzem-overview-grid">
        <div><dt>Start time</dt><dd>${new Date(state.sessionStart).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</dd></div>
        <div><dt>Running duration</dt><dd>${formatDuration(Date.now() - state.sessionStart)}</dd></div>
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
      .map(([timestamp, data]) => ({ t: timestampMilliseconds(timestamp), power: Number(data.power ?? data), energy: Number(data.energy ?? NaN) }))
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

    const powers = points.map((point) => point.power).filter((value) => Number.isFinite(value));
    const peakPower = powers.length ? Math.max(...powers) : 0;
    const avgPower = powers.length ? powers.reduce((sum, value) => sum + value, 0) / powers.length : 0;

    // Reconstructs completed ON/OFF sessions for this range straight from the
    // stored 5-minute readings, so the session-history list survives a page
    // refresh instead of only reflecting what happened since this tab was
    // opened. Also drives "Total runtime" with the same hysteresis rule the
    // sessions themselves use, rather than a separate single-threshold sum.
    const { sessions: reconstructedSessions, totalOnMs: runtimeMs } =
      computeSessionsFromPoints(points);
    getRuntimeState(meterN).historicalSessions = reconstructedSessions;
    if (meterN === activeMeterNumber) renderModalSessions();

    const liveMeter = getMeter(meterN);
    note.textContent = `${points.length} readings · ${range === "1h" ? "1 hour" : range === "6h" ? "6 hours" : range === "12h" ? "12 hours" : range === "today" ? "today" : range === "yesterday" ? "yesterday" : range === "30d" ? "30 days" : "7 days"}`;

    $("modalHistorySummary").innerHTML = `
      <div><span>Today's energy</span><strong>${number(liveMeter.energy, 2)} kWh</strong></div>
      <div><span>Peak power</span><strong>${number(peakPower, 1)} W</strong></div>
      <div><span>Average power</span><strong>${number(avgPower, 1)} W</strong></div>
      <div><span>Total runtime</span><strong>${formatDuration(runtimeMs)}</strong></div>`;
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