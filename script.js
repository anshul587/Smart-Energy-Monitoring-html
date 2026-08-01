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

function number(value, decimals = 0) {
  return Number(value || 0).toFixed(decimals);
}

function displayMeters() {
  return { ...demoMeters, ...liveMeters };
}

function renderDashboard() {
  const meters = displayMeters();
  const entries = Object.entries(meters).sort(([a], [b]) =>
    Number(a.replace("pzem_", "")) - Number(b.replace("pzem_", ""))
  );

  dashboard.replaceChildren();

  entries.forEach(([id, meter], index) => {
    const card = meterTemplate.content.cloneNode(true);
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
