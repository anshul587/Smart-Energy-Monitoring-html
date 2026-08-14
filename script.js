const firebaseConfig = {
  apiKey: "AIzaSyDtlOe9Qx1ZlnSTDgMezoqUFed_XHjI6yU",
  authDomain: "energy-monitoring-system-65d79.firebaseapp.com",
  databaseURL: "https://energy-monitoring-system-65d79-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "energy-monitoring-system-65d79",
  storageBucket: "energy-monitoring-system-65d79.firebasestorage.app",
  messagingSenderId: "78868370157",
  appId: "1:78868370157:web:d9b2b0bbc65114d7c732e4"
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
          label: (context) => `${context.dataset.label}: ${Number(context.parsed.y || 0).toFixed(1)} W`
        }
      }
    },
    scales: {
      x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } },
      y: {
        beginAtZero: true,
        min: 0,
        max: 300,
        title: { display: true, text: "Power (W)" },
        ticks: { stepSize: 30 }}
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
  const axisMaximum = Math.max(300, Math.ceil(highestPower / 300) * 300);

  powerChart.options.scales.y.min = 0;
  powerChart.options.scales.y.max = axisMaximum;
  powerChart.options.scales.y.ticks.stepSize = axisMaximum / 10;
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
}

/* Supports both /meters/pzem_1 and your ESP code /energy/pzem1 */
firebase.database().ref("meters").on("value", (snapshot) => {
  if (Object.keys(snapshot.val() || {}).length) useLiveData(snapshot.val());
});

firebase.database().ref("energy").on("value", (snapshot) => {
  if (Object.keys(snapshot.val() || {}).length) useLiveData(snapshot.val());
});

$("powerRange").addEventListener("change", (event) => {
  loadPowerHistory(event.target.value);
});

unitRate.addEventListener("change", () => renderDashboard());

$("themeToggle").addEventListener("click", () => {
  document.body.classList.toggle("dark");
});

$("exportButton").addEventListener("click", () => {
  const rows = [["Meter", "Voltage", "Current", "Power", "Energy", "Frequency", "Power Factor"]];

  for (let i = 1; i <= 9; i++) {
    const meter = getMeter(i);
    rows.push([
      `PZEM ${i}`,
      meter.voltage || 0,
      meter.current || 0,
      meter.power || 0,
      meter.energy || 0,
      meter.frequency || 0,
      meter.pf || 0
    ]);
  }

  const link = document.createElement("a");
  link.href = URL.createObjectURL(
    new Blob([rows.map((row) => row.join(",")).join("\n")], { type: "text/csv" })
  );
  link.download = "smart-monitoring-readings.csv";
  link.click();
  URL.revokeObjectURL(link.href);
});

renderDashboard();
loadPowerHistory("1d");
