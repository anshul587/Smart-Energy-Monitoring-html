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
const dashboard = document.getElementById("dashboardContent");
const meterTemplate = document.getElementById("meterTemplate");
const connectionStatus = document.getElementById("connectionStatus");
const lastUpdated = document.getElementById("lastUpdated");
const totalPower = document.getElementById("totalPower");
const totalEnergy = document.getElementById("totalEnergy");
const onlineMeters = document.getElementById("onlineMeters");
const averageVoltage = document.getElementById("averageVoltage");
const meterCount = document.getElementById("meterCount");
const unitRate = document.getElementById("unitRate");
const billTotalUnits = document.getElementById("billTotalUnits");
const billTotalCost = document.getElementById("billTotalCost");
const billRateText = document.getElementById("billRateText");
const billMeterRows = document.getElementById("billMeterRows");
const meterDialog = document.getElementById("meterDialog");
const dialogMeterName = document.getElementById("dialogMeterName");
const historyMessage = document.getElementById("historyMessage");
let meterHistoryChart = null;
const chartColors = [
  "#2578ff", "#7b4cf6", "#f28d2f", "#13b887", "#e45f92",
  "#16a7d9", "#a269d8", "#d88323", "#2c9e72"
];

const powerChart = new Chart(document.getElementById("powerChart"), {
  type: "line",
  data: {
    labels: [],
    datasets: Array.from({ length: 9 }, (_, index) => ({
      label: `PZEM ${index + 1}`,
      data: [],
      borderColor: chartColors[index],
      backgroundColor: chartColors[index],
      borderWidth: 2,
      tension: 0.35,
      pointRadius: 0,
      pointHoverRadius: 4
    }))
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { boxWidth: 9, boxHeight: 9, usePointStyle: true, pointStyle: "circle", padding: 14 } },
      tooltip: { callbacks: { label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(1)} W` } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { maxTicksLimit: 6 } },
      y: { beginAtZero: true, title: { display: true, text: "Power (W)" }, grid: { color: "rgba(101, 115, 136, 0.15)" } }
    }
  }
});

const demoMeters = Object.fromEntries(
  Array.from({ length: 9 }, (_, index) => {
    const number = index + 1;
    return [`pzem_${number}`, {
      name: `PZEM ${number}`,
      voltage: 0,
      current: 0,
      power: 0,
      energy: 0,
      pf: 0,
      demo: true
    }];
  })
);

let liveMeters = {};
let currentEntries = [];

function number(value, decimals = 0) {
  return Number(value || 0).toFixed(decimals);
}

function displayMeters() {
  return { ...demoMeters, ...liveMeters };
}

function formatRupees(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
}

function updateBillCalculator(entries = currentEntries) {
  const hasRate = unitRate.value !== "";
  const rate = hasRate ? Math.max(0, Number(unitRate.value)) : 0;
  const activeMeters = entries.filter(([, meter]) => !meter.demo);
  const totalUnits = activeMeters.reduce((total, [, meter]) => total + Number(meter.energy || 0), 0);

  billTotalUnits.innerHTML = `${number(totalUnits, 2)} <small>kWh</small>`;
  billTotalCost.textContent = hasRate ? formatRupees(totalUnits * rate) : "—";
  billRateText.textContent = hasRate ? `At ${formatRupees(rate)} per unit` : "Select your price per unit";
  billMeterRows.replaceChildren();

  if (!activeMeters.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="3">Waiting for live PZEM energy readings…</td>';
    billMeterRows.append(row);
    return;
  }

  activeMeters.forEach(([id, meter], index) => {
    const energy = Number(meter.energy || 0);
    const row = document.createElement("tr");
    const meterCost = hasRate ? formatRupees(energy * rate) : "Select price";
    row.innerHTML = `<td><b>${meter.name || `PZEM ${index + 1}`}</b><small>${id.toUpperCase()}</small></td><td>${number(energy, 2)} kWh</td><td>${meterCost}</td>`;
    billMeterRows.append(row);
  });
}

function updatePowerChart() {
  if (!Object.keys(liveMeters).length) return;

  const label = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });

  powerChart.data.labels.push(label);

  powerChart.data.datasets.forEach((dataset, index) => {
    const meter = liveMeters[`pzem_${index + 1}`];
    dataset.data.push(Number(meter?.power || 0));
  });

  if (powerChart.data.labels.length > 30) {
    powerChart.data.labels.shift();
    powerChart.data.datasets.forEach((dataset) => dataset.data.shift());
  }

  powerChart.update();
}

function openMeterHistory(id, meter) {
  dialogMeterName.textContent = `${meter.name || id} power history`;
  historyMessage.textContent = "Loading 30-minute readings…";
  meterDialog.showModal();

  firebase.database().ref(`history/${id}`).orderByKey().limitToLast(48).once("value")
    .then((snapshot) => {
      const rows = Object.entries(snapshot.val() || {}).sort(([a], [b]) => Number(a) - Number(b));

      if (!rows.length) {
        historyMessage.textContent = "No stored history yet. The first point appears after the next 30-minute log.";
        if (meterHistoryChart) {
          meterHistoryChart.destroy();
          meterHistoryChart = null;
        }
        return;
      }

      historyMessage.textContent = `Showing ${rows.length} stored readings (one point every 30 minutes).`;
      const labels = rows.map(([timestamp]) => new Date(Number(timestamp) * 1000).toLocaleString([], {
        day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
      }));
      const values = rows.map(([, item]) => Number(item.power || 0));

      if (meterHistoryChart) meterHistoryChart.destroy();
      meterHistoryChart = new Chart(document.getElementById("meterHistoryChart"), {
        type: "line",
        data: {
          labels,
          datasets: [{
            label: "Active power (W)",
            data: values,
            borderColor: "#2578ff",
            backgroundColor: "rgba(37, 120, 255, .16)",
            fill: true,
            borderWidth: 3,
            tension: .35,
            pointRadius: 3
          }]
        },
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
    })
    .catch((error) => {
      console.error(error);
      historyMessage.textContent = "Unable to load history. Check Firebase Database Rules.";
    });
}

function renderDashboard() {
  const meters = displayMeters();
  const entries = Object.entries(meters).sort(([a], [b]) =>
    Number(a.replace("pzem_", "")) - Number(b.replace("pzem_", ""))
  );
  currentEntries = entries;

  dashboard.replaceChildren();

  entries.forEach(([id, meter], index) => {
    const card = meterTemplate.content.cloneNode(true);
    const meterCard = card.querySelector(".meter-card");
    const isOnline = !meter.demo;
    const power = Number(meter.power || 0);

    card.querySelector(".meter-number").textContent = String(index + 1).padStart(2, "0");
    card.querySelector(".meter-name").textContent = meter.name || `PZEM ${index + 1}`;
    card.querySelector(".meter-id").textContent = id.replace("_", " ").toUpperCase();
    card.querySelector(".meter-power strong").textContent = number(power, 1);
    card.querySelector(".voltage").textContent = `${number(meter.voltage, 1)} V`;
    card.querySelector(".current").textContent = `${number(meter.current, 2)} A`;
    card.querySelector(".energy").textContent = `${number(meter.energy, 2)} kWh`;
    card.querySelector(".pf").textContent = number(meter.pf, 2);
    card.querySelector(".power-track span").style.width = `${Math.min((power / maxPower) * 100, 100)}%`;

    const status = card.querySelector(".meter-status");
    status.classList.add(isOnline ? "online" : "offline");
    status.querySelector("b").textContent = isOnline ? "Live" : "Waiting";
    meterCard.tabIndex = 0;
    meterCard.setAttribute("role", "button");
    meterCard.setAttribute("aria-label", `Show ${meter.name || id} history`);
    meterCard.addEventListener("click", () => openMeterHistory(id, meter));
    meterCard.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") openMeterHistory(id, meter);
    });
    dashboard.append(card);
  });

  const online = entries.filter(([, meter]) => !meter.demo);
  const active = online.length ? online : [];
  const sum = (key) => active.reduce((total, [, meter]) => total + Number(meter[key] || 0), 0);
  const average = active.length ? sum("voltage") / active.length : 0;

  totalPower.innerHTML = `${number(sum("power"), 1)} <small>W</small>`;
  totalEnergy.innerHTML = `${number(sum("energy"), 2)} <small>kWh</small>`;
  onlineMeters.innerHTML = `${online.length} <small>/ 9</small>`;
  averageVoltage.innerHTML = `${number(average, 1)} <small>V</small>`;
  meterCount.textContent = `${entries.length} meters`;
  updateBillCalculator(entries);
}

function setStatus(text, state) {
  connectionStatus.className = `connection-pill ${state}`;
  connectionStatus.innerHTML = `<span></span> ${text}`;
}

firebase.database().ref("meters").on(
  "value",
  (snapshot) => {
    liveMeters = snapshot.val() || {};
    renderDashboard();
    updatePowerChart();
    setStatus("System online", "online");
    lastUpdated.textContent = liveMeters && Object.keys(liveMeters).length
      ? `Last synchronised ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`
      : "Waiting for live PZEM data…";
  },
  (error) => {
    console.error(error);
    setStatus("Firebase error", "error");
    lastUpdated.textContent = "Check Firebase Database Rules and configuration.";
  }
);

document.getElementById("themeToggle").addEventListener("click", () => {
  document.body.classList.toggle("dark");
});

unitRate.addEventListener("change", () => updateBillCalculator());

document.getElementById("closeDialog").addEventListener("click", () => meterDialog.close());
meterDialog.addEventListener("click", (event) => {
  if (event.target === meterDialog) meterDialog.close();
});

document.getElementById("exportButton").addEventListener("click", () => {
  const rows = [["Meter", "Voltage (V)", "Current (A)", "Power (W)", "Energy (kWh)", "Power Factor"]];
  Object.entries(displayMeters()).forEach(([id, meter]) => {
    rows.push([id, meter.voltage || 0, meter.current || 0, meter.power || 0, meter.energy || 0, meter.pf || 0]);
  });

  const csv = rows.map((row) => row.join(",")).join("\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  link.download = "smart-monitoring-readings.csv";
  link.click();
  URL.revokeObjectURL(link.href);
});

renderDashboard();
