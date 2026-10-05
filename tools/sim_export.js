// Runtime simulation of the dashboard export handler with fake Firebase data.
// Verifies actual CSV output: columns, mapping resolution, alert fallback.
const HISTORY_SLOT_MS = 5 * 60 * 1000;

const MAPPING = { pzem_1: { load_name: 'Fan 1', location: 'Classroom' } };
const getMappedLoadName = (k) => (MAPPING[k] && MAPPING[k].load_name) || k.toUpperCase().replace('_', '-');
const getMappedLoadLocation = (k) => (MAPPING[k] && MAPPING[k].location) || 'Unassigned';

function timestampMilliseconds(k) { return Number(k) * 1000; }
function classifyReading() { return { status: 'OK' }; }
function formatKolkataDateTime(ms) {
  const d = new Date(ms);
  return { date: d.toISOString().slice(0, 10), time: d.toISOString().slice(11, 19) };
}

// t0 in a clean 5-min slot; one reading with no alert, one with an alert.
const slot0 = 1700000100;
const slot1 = 1700000400;

const HISTORY = {
  pzem_1: {
    [slot0]: { voltage: 230, current: 1.2, power: 276, energy: 1.5, pf: 0.98, frequency: 50 },
    [slot1]: { voltage: 231, current: 1.3, power: 300, energy: 1.6, pf: 0.97, frequency: 50 },
  },
  pzem_2: {
    [slot0]: { voltage: 229, current: 0.4, power: 92, energy: 0.4, pf: 0.99, frequency: 49.9 },
  },
};
const ALERTS = {
  pzem_1: {
    [slot1 * 1000]: { type: 'over_voltage', severity: 'WARNING', timestamp: slot1 * 1000 },
  },
};

const snapshots = [1, 2].map((n) => ({ val: () => HISTORY[`pzem_${n}`] || null }));
const alertSnapshot = { val: () => ALERTS };

// ---- handler body (mirrors script.js export) ----
const rows = [["Date", "Time", "PZEM ID", "Load Name", "Location", "Voltage (V)", "Current (A)", "Power (W)", "Energy (kWh)", "PF", "Frequency (Hz)", "Alert"]];

const alertBySlot = {};
const alertTree = alertSnapshot && alertSnapshot.val ? alertSnapshot.val() : {};
Object.keys(alertTree || {}).forEach((pzemKey) => {
  const entries = alertTree[pzemKey];
  if (!entries || typeof entries !== 'object') return;
  Object.entries(entries).forEach(([ts, alert]) => {
    if (!alert || typeof alert !== 'object') return;
    const stamp = Number(alert.timestamp) || timestampMilliseconds(ts);
    if (!Number.isFinite(stamp) || stamp <= 0) return;
    const slot = Math.floor(stamp / HISTORY_SLOT_MS);
    const label = [alert.type, alert.severity].filter(Boolean).join(" / ");
    if (!label) return;
    const id = `${pzemKey}@${slot}`;
    alertBySlot[id] = alertBySlot[id] ? `${alertBySlot[id]}; ${label}` : label;
  });
});

snapshots.forEach((snapshot, meterIndex) => {
  const pzemKey = `pzem_${meterIndex + 1}`;
  const pzemId = `PZEM-${meterIndex + 1}`;
  const loadName = getMappedLoadName(pzemKey);
  const location = getMappedLoadLocation(pzemKey);
  Object.entries(snapshot.val() || {}).forEach(([timestampKey, reading]) => {
    if (!reading || typeof reading !== 'object') return;
    if (classifyReading(reading).status === 'INVALID') return;
    const stamp = timestampMilliseconds(timestampKey);
    const { date, time } = formatKolkataDateTime(stamp);
    const alert = alertBySlot[`${pzemKey}@${Math.floor(stamp / HISTORY_SLOT_MS)}`] || '\u2014';
    rows.push([date, time, pzemId, loadName, location,
      reading.voltage ?? 0, reading.current ?? 0, reading.power ?? 0, reading.energy ?? 0,
      reading.pf ?? 0, reading.frequency ?? 0, alert]);
  });
});

const csv = rows.map((row) => row.map((cell) => {
  const s = String(cell ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}).join(',')).join('\n');

console.log(csv);
console.log('---');

let fail = 0;
function check(cond, msg) { console.log((cond ? 'ok  - ' : 'FAIL- ') + msg); if (!cond) fail++; }

check(rows[0].length === 12, 'header has 12 columns');
check(rows.every((r) => r.length === 12), 'every row has 12 cells');
check(rows[0].join('|') === 'Date|Time|PZEM ID|Load Name|Location|Voltage (V)|Current (A)|Power (W)|Energy (kWh)|PF|Frequency (Hz)|Alert', 'exact column order');
check(!rows[0].includes('Type'), 'no Type column');
const pzem1 = rows.filter((r) => r[2] === 'PZEM-1');
check(pzem1.length === 2 && pzem1.every((r) => r[3] === 'Fan 1' && r[4] === 'Classroom'), 'PZEM-1 -> Fan 1 / Classroom');
check(pzem1[0][11] === '\u2014', 'no alert in slot -> em dash');
check(pzem1[1][11] === 'over_voltage / WARNING', 'real alert surfaced for its slot');
const pzem2 = rows.filter((r) => r[2] === 'PZEM-2');
check(pzem2.length === 1 && pzem2[0][3] === 'PZEM-2' && pzem2[0][4] === 'Unassigned', 'missing mapping -> default fallback');
check(!/NORMAL|No fault|No anomaly/.test(csv), 'no fabricated alert text');
check(pzem1[0][9] === 0.98 && pzem1[0][10] === 50, 'PF then Frequency order preserved');
process.exitCode = fail ? 1 : 0;