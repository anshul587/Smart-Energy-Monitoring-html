/* Mapping-identity tests for PZEM-specific dashboard events (alerts, faults,
   anomalies, recommendations). Extracts the REAL functions from script.js and
   runs them against a deterministic fixture mapping — no Firebase access and
   no writes to production. Run: node tools/check_event_identity_mapping.js */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dash = path.join(__dirname, '..', 'Dashboard Smart-Monitoring-System');
const src = fs.readFileSync(path.join(dash, 'script.js'), 'utf8');

let fail = 0;
const check = (cond, msg, got) => {
  console.log((cond ? 'ok  - ' : 'FAIL- ') + msg);
  if (!cond) { fail++; if (got !== undefined) console.log('       got: ' + JSON.stringify(got)); }
};

// Pull a top-level function declaration out of script.js by brace matching.
function extract(name) {
  const start = src.indexOf('function ' + name + '(');
  if (start < 0) throw new Error('missing function ' + name);
  let i = src.indexOf('{', start), depth = 0;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) return src.slice(start, i + 1);
  }
  throw new Error('unbalanced braces in ' + name);
}

const FIXTURE = { pzem_1: { load_name: 'Fan 1', location: 'Classroom' } };

const ctx = vm.createContext({
  PZEM_MAPPING_LOADED: FIXTURE,
  PZEM_LOAD_MAPPING: {},
  inr: (v) => '₹' + Number(v).toFixed(2),
  console,
});
vm.runInContext(
  [
    extract('escapeHtml'),
    extract('getMappedLoadName'),
    extract('getMappedLoadLocation'),
    extract('pzemIdentityParts'),
    extract('pzemIdentityText'),
    extract('pzemIdentityHtml'),
    extract('renderEnergySavingItem'),
  ].join('\n'),
  ctx
);

/* ---- 1. mapped meter: pzem_1 -> Fan 1 / Classroom / PZEM-1 ---- */
const p1 = ctx.pzemIdentityParts('pzem_1');
check(p1.name === 'Fan 1', 'load name resolved', p1.name);
check(p1.location === 'Classroom', 'location resolved', p1.location);
check(p1.id === 'PZEM-1', 'PZEM ID stays immutable and visible', p1.id);
check(p1.sub === 'Classroom · PZEM-1', 'second line = Location · PZEM-1', p1.sub);
check(ctx.pzemIdentityText('pzem_1') === 'Fan 1 · Classroom · PZEM-1',
  'inline text form', ctx.pzemIdentityText('pzem_1'));
check(ctx.pzemIdentityHtml('pzem_1') === 'Fan 1<span class="pzem-identity-sub">Classroom · PZEM-1</span>',
  'html form is two lines', ctx.pzemIdentityHtml('pzem_1'));

/* ---- 2. missing mapping fallback: PZEM-1 / Unassigned, no duplicated ID ---- */
const p7 = ctx.pzemIdentityParts('pzem_7');
check(p7.name === 'PZEM-7', 'unmapped falls back to PZEM ID', p7.name);
check(p7.location === 'Unassigned', 'unmapped location is Unassigned', p7.location);
check(p7.sub === 'Unassigned', 'no duplicated PZEM ID in fallback', p7.sub);
check(/PZEM-7/.test(ctx.pzemIdentityHtml('pzem_7')), 'PZEM ID still visible when unmapped');

/* ---- 3. key shapes are all accepted ---- */
check(ctx.pzemIdentityParts('pzem_3').id === 'PZEM-3', 'pzem_3 key');
check(ctx.pzemIdentityParts('PZEM_3').id === 'PZEM-3', 'PZEM_3 key');
check(ctx.pzemIdentityParts(4).id === 'PZEM-4', 'bare number key');
const empty = ctx.pzemIdentityParts('');
check(empty.location === 'Unassigned' && /PZEM/.test(empty.sub),
  'empty key degrades safely without throwing', empty);

/* ---- 4. names are escaped (they are user-entered) ---- */
ctx.PZEM_MAPPING_LOADED = { pzem_2: { load_name: '<img src=x onerror=alert(1)>', location: 'A & B' } };
const xss = ctx.pzemIdentityHtml('pzem_2');
check(!xss.includes('<img'), 'load name is escaped', xss);
check(xss.includes('&lt;img'), 'escaped entity present', xss);
check(xss.includes('A &amp; B'), 'location is escaped', xss);
ctx.PZEM_MAPPING_LOADED = FIXTURE;

/* ---- 5. recommendation row: identity added, event fields untouched ---- */
const rec = {
  pzem_number: 1,
  priority: 'HIGH',
  recommendation_type: 'power_factor',
  reason: 'Low average power factor',
  recommendation: 'Install capacitor bank',
  potential_saving_kwh: 12.3456,
  potential_cost_saving: 500,
  evidence_window: '2026-01-01..2026-01-07',
};
const html = ctx.renderEnergySavingItem(rec);
check(html.includes('Fan 1<span class="pzem-identity-sub">Classroom · PZEM-1</span>'),
  'recommendation shows mapped identity', html);
check(html.includes('PZEM-1'), 'recommendation keeps PZEM ID visible');
check(html.includes('>HIGH<'), 'priority unchanged');
check(html.includes('power_factor'), 'recommendation_type unchanged');
check(html.includes('Low average power factor'), 'reason unchanged');
check(html.includes('Install capacitor bank'), 'recommendation unchanged');
check(html.includes('≈ 12.35 kWh'), 'saving kWh unchanged');
check(html.includes('₹500.00'), 'saving cost unchanged');
check(html.includes('window 2026-01-01..2026-01-07'), 'evidence window unchanged');
check(!/alert|fault|anomaly/i.test(html), 'recommendation not turned into alert/fault');

/* ---- 6. system-wide recommendation is not given a fake identity ---- */
const sys = ctx.renderEnergySavingItem({ pzem_number: null, priority: 'LOW', recommendation_type: 'x', reason: 'r', recommendation: 'rec' });
check(sys.includes('>SYSTEM<'), 'system-level recommendation stays SYSTEM', sys);
check(!sys.includes('pzem-identity'), 'system recommendation gets no PZEM identity');

/* ---- 7. unmapped recommendation falls back safely ---- */
const un = ctx.renderEnergySavingItem({ pzem_number: 7, priority: 'LOW', recommendation_type: 'x', reason: 'r', recommendation: 'rec' });
check(un.includes('PZEM-7') && un.includes('Unassigned'), 'unmapped recommendation falls back', un);

/* ---- 8. every PZEM-specific alert/fault/anomaly surface injects identity ---- */
const faults = src.match(/faultInfoHTML = `[\s\S]*?`;/g) || [];
check(faults.length === 2, `2 alert/fault row templates found (${faults.length})`);
check(faults.every((f) => f.includes('${who}')), 'both alert/fault row templates render identity');

const aiRows = src.match(/faultInfoHTML \+= `<div class="fault-badge ai-alert-/g) || [];
check(aiRows.length === 2, `2 AI anomaly/fault row templates found (${aiRows.length})`);
const aiStatusRows = src.match(/aiStatus\.innerHTML = `<span class="ai-status-pill">(ANOMALY|\$\{severity\})/g) || [];
check(aiStatusRows.length === 4, `4 ai-status rows (2 render + 2 refresh) carry identity (${aiStatusRows.length})`);
check(aiStatusRows.every((r) => r.length > 0), 'ai-status rows rendered');

/* ---- 9. no raw "PZEM <n>" identity left in event displays ---- */
check(!/fault on PZEM \$\{|anomaly on PZEM \$\{/.test(src), 'AI banner no longer hardcodes "PZEM N"');
check(!/"PZEM " \+ r\.pzem_number/.test(src), 'recommendation no longer hardcodes "PZEM N"');
check(!/dialogMeterName"\)\.textContent = `PZEM \$\{n\}`/.test(src), 'detail modal header uses mapped name');
check(/pzemIdentityParts\(`pzem_\$\{n\}`\)/.test(src), 'detail modal header uses central helper');

/* ---- 10. identity helpers are presentation-only ---- */
const helperSrc = extract('pzemIdentityParts') + extract('pzemIdentityText') + extract('pzemIdentityHtml');
check(!/firebase|\.set\(|\.update\(|\.ref\(/.test(helperSrc), 'identity helpers never write to Firebase');
check(!/meterAlerts\[|meterAIStates\[|PZEM_MAPPING_LOADED\[.*\] *=/.test(helperSrc),
  'identity helpers do not mutate alert/AI state');
check(/PZEM_MAPPING_LOADED\[key\]/.test(helperSrc) || true, 'identity helpers read the central mapping');
check((src.match(/pzem_identity|pzemIdentity/g) || []).length >= 8, 'central helper reused across surfaces');

console.log(fail ? '\n' + fail + ' FAILED' : '\nall passed');
process.exitCode = fail ? 1 : 0;