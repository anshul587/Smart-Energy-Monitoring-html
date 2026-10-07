/* Tests for the click-to-open Recommendation Details modal (script.js):
   1. cards are clickable (data-idx + role/tabindex + affordance)
   2. click + keyboard delegation is wired to #energySavingList
   3. modal markup exists, starts hidden, unique ids
   4. PZEM-specific detail renders stored fields (identity, reason,
      evidence metrics, recommendation, impact, window, source)
   5. all 7 section headings are present
   6. missing fields fall back to "Not available"
   7. SYSTEM stays SYSTEM / whole-site (no fake PZEM identity)
   8. user-entered values are HTML-escaped
   9. open/close symmetry (div.modal, never showModal)
  10. Escape + backdrop close are wired
  11. the detail path never writes to Firebase
  12. mapping modal untouched + display order matches energySavingRecs
   No Firebase access — real functions run inside a vm. No writes. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dash = path.join(__dirname, '..', 'Dashboard Smart-Monitoring-System');
const html = fs.readFileSync(path.join(dash, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(dash, 'style.css'), 'utf8');
const src = fs.readFileSync(path.join(dash, 'script.js'), 'utf8');

let fail = 0;
const check = (cond, msg, got) => {
  console.log((cond ? 'ok  - ' : 'FAIL- ') + msg);
  if (!cond) { fail++; if (got !== undefined) console.log('       got: ' + JSON.stringify(got)); }
};

function extract(name) {
  const needle = 'function ' + name + '(';
  const match = src.indexOf(needle);
  if (match < 0) throw new Error('missing function ' + name);
  let i = src.indexOf('{', match), depth = 0;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) return src.slice(match, i + 1);
  }
  throw new Error('unbalanced braces in ' + name);
}
function extractConst(name) {
  const start = src.indexOf('const ' + name + ' = ');
  if (start < 0) throw new Error('missing const ' + name);
  const end = src.indexOf('};', start);
  return src.slice(start, end + 2);
}

/* ---- 1. cards are clickable ---- */
check(/function renderEnergySavingItem\(r, idx\)/.test(src), 'renderEnergySavingItem takes (r, idx)');
const card = extract('renderEnergySavingItem');
check(/" data-idx="\$\{Number\.isInteger\(idx\) \? idx : ""\}"/.test(card),
  'card carries data-idx for lookup', card.match(/" data-idx="[^"]*"/));
check(/value="undefined"/.test(card) === false, 'no literal "undefined" data-idx leak');
check(/role="button" tabindex="0"/.test(card), 'card is keyboard-focusable');
check(/es-more">Details</.test(card), 'card shows a Details affordance');

/* ---- 2. delegation (re-render safe) wired to the container ---- */
const dcl = src.slice(src.lastIndexOf("DOMContentLoaded"));
check(dcl.includes("getElementById('energySavingList')"), 'energySavingList container is bound');
check(dcl.includes("activateEnergySavingItem(e.target)"), 'click delegated from container');
check(dcl.includes("e.key === 'Enter' || e.key === ' '"), 'Enter/Space keyboard activation wired');
const activate = extract('activateEnergySavingItem');
check(/closest\('\.es-item'\)/.test(activate), 'activation resolves via closest(.es-item)');
check(/energySavingRecs\[Number\(idx\)\]/.test(activate), 'activation looks up energySavingRecs by index');
check(/openRecommendationDetail\(rec\)/.test(activate), 'activation opens the detail modal');

/* ---- 3. modal markup ---- */
const tag = html.match(/<div id="recommendationModal"[^>]*>/);
check(!!tag, '#recommendationModal exists');
check(tag && /\bhidden\b/.test(tag[0]) && tag[0].includes('aria-hidden="true"'),
  'modal starts hidden + aria-hidden', tag && tag[0]);
check(tag && tag[0].includes('role="dialog"') && tag[0].includes('aria-modal="true"') &&
  tag[0].includes('aria-labelledby="recDetailTitle"'), 'modal is an accessible dialog');
['recDetailTitle', 'recDetailSubtitle', 'recDetailBody', 'recDetailClose'].forEach((id) => {
  check(html.includes('id="' + id + '"'), 'unique id present: ' + id);
});
check((html.match(/class="modal"/g) || []).length === 2, 'exactly two div modals (mapping + recommendation)');

/* ---- 4-8. detail rendering against real functions ---- */
const FIXTURE = { pzem_1: { load_name: 'Fan 1', location: 'Classroom' } };
const ctx = vm.createContext({
  PZEM_MAPPING_LOADED: FIXTURE,
  PZEM_LOAD_MAPPING: {},
  inr: (v) => '₹' + Number(v).toFixed(2),
  console,
});
vm.runInContext(
  [
    extractConst('ES_TYPE_LABELS'),
    extractConst('ES_EVIDENCE_LABELS'),
    extract('escapeHtml'),
    extract('getMappedLoadName'),
    extract('getMappedLoadLocation'),
    extract('pzemIdentityParts'),
    extract('_esMetricValue'),
    extract('_esEvidenceRows'),
    extract('_esSourceLabel'),
    extract('formatRecTimestamp'),
    extract('renderRecommendationDetail'),
  ].join('\n'),
  ctx
);

const SHIFT = {
  pzem_number: 1,
  timestamp: 1780000000,
  recommendation_type: 'SHIFT_NON_CRITICAL_LOAD',
  priority: 'HIGH',
  recommendation: 'Shift non-critical loads outside the recurring peak window.',
  reason: 'Recurring high-demand window 09:00-10:00 (median 63 W vs typical 8 W).',
  evidence: { window: '09:00-10:00', peak_median_w: 63.41, typical_median_w: 8.12 },
  potential_saving_kwh: null,
  potential_cost_saving: null,
  estimated_percent_reduction: null,
  evidence_window: '2026-08-16 05:05 -> 2026-10-06 10:05 UTC',
  source_stages: ['stage1/history', 'stage2/preprocessing'],
};
const d = ctx.renderRecommendationDetail(SHIFT);
check(d.includes('Fan 1') && d.includes('Classroom') && d.includes('PZEM-1'),
  'PZEM-specific scope resolves mapped identity');
check(d.includes('<span class="rec-k">Type:</span> Shift non-critical load'),
  'WHAT HAPPENED names the flagged pattern, clearly labeled as Type',
  d.match(/What happened[\s\S]{0,80}/));
check(d.includes('Recurring high-demand window 09:00-10:00 (median 63 W vs typical 8 W).'),
  'WHY THIS RECOMMENDATION shows the stored casual explanation (reason)',
  d.match(/Why this recommendation[\s\S]{0,120}/));
const whySlice = d.slice(d.indexOf('Why this recommendation'), d.indexOf('Evidence'));
check(whySlice.includes('Recurring high-demand window') && !whySlice.includes('Shift non-critical load'),
  'WHY section = stored reason only, no bare type code presented as the why');
check(d.includes('Recurring time-of-day window') && d.includes('09:00-10:00'),
  'evidence window row = stored window label');
check(d.includes('Median power in window') && d.includes('63.41 W'), 'evidence median has label + W unit');
check(d.includes('Typical power (baseline)') && d.includes('8.12 W'), 'evidence typical has label + W unit');
check(d.includes('Shift non-critical loads outside the recurring peak window.'),
  'WHAT SHOULD I DO shows stored recommendation text');
check(d.includes('2026-08-16 05:05 -&gt; 2026-10-06 10:05 UTC'),
  'evidence window string shown verbatim (escaped only for HTML)', d.match(/\d{4}-\d{2}-\d{2}[\s\S]{0,40}UTC/));
check(d.includes('Stage 1 · history') && d.includes('Stage 2 · preprocessing'), 'source_stages humanized');
const headings = ['What happened', 'Why this recommendation', 'Evidence', 'What should I do?', 'Potential impact', 'Evidence window', 'Source'];
headings.forEach((h) => check(d.includes('<h3>' + h + '</h3>'), 'section heading present: ' + h));
check(d.indexOf('<h3>What happened</h3>') < d.indexOf('<h3>Source</h3>'), 'sections render in order');
check((d.match(/Not available/g) || []).length === 1,
  'null savings + null percent keep impact NOT available (evidence window + sources present)', d.match(/Not available/g));

const SPARSE = ctx.renderRecommendationDetail({
  pzem_number: null, reason: 'r', recommendation: 'do it', evidence: {},
});
check((SPARSE.match(/Not available/g) || []).length === 5,
  'sparse rec falls back to Not available for why/evidence/impact/window/source (5)', SPARSE.match(/Not available/g));

/* ---- no-fabrication matrix: zero stays 0, empty evidence -> NA, unmapped ---- */
const ZERO = ctx.renderRecommendationDetail({
  pzem_number: 1, recommendation_type: 'REDUCE_IDLE_CONSUMPTION', reason: 'r',
  recommendation: 'do it', evidence: { idle_baseline_w: 0 },
  potential_saving_kwh: 0, estimated_percent_reduction: 0,
});
check(ZERO.includes('≈ 0.00 kWh'), 'zero savings remains 0 (not Not available)');
check(ZERO.includes('≈ 0.00%'), 'zero percent reduction remains 0');
check(!/idle_baseline_w|Not available/.test(ZERO.slice(ZERO.indexOf('Evidence'), ZERO.indexOf('What should I do?'))),
  'evidence value 0 renders as a real number, not NA');
const NOEVID = ctx.renderRecommendationDetail({
  pzem_number: 1, reason: 'r', recommendation: 'do it', evidence: {}, source_stages: ['stage1/history'],
});
check((NOEVID.match(/Not available/g) || []).length >= 2,
  'empty evidence object -> Evidence + impact Not available', NOEVID.match(/Not available/g));
const UNMAPPED = ctx.renderRecommendationDetail({
  pzem_number: 9, recommendation_type: 'REDUCE_PEAK_LOAD', reason: 'r', recommendation: 'do',
});
check(UNMAPPED.includes('PZEM-9') && UNMAPPED.includes('Unassigned'),
  'unmapped PZEM recommendation keeps PZEM-N + Unassigned (no inventions)');

const SYSTEM = ctx.renderRecommendationDetail({
  pzem_number: null, recommendation_type: 'REDUCE_PEAK_LOAD', reason: 'System-wide peak', recommendation: 'Shed load during peaks', evidence: { peak_power_w: 1200 },
  source_stages: ['stage7/peak_detection'],
});
check(SYSTEM.includes('SYSTEM') && SYSTEM.includes('Whole-site · all meters'),
  'SYSTEM scope stays SYSTEM / whole-site');
check(!/PZEM-\d|pzem_identity/.test(SYSTEM), 'SYSTEM detail gets no fake PZEM identity');

const DANGER = ctx.renderRecommendationDetail({ pzem_number: 2, reason: '<script>alert(1)</script>', evidence: {} });
check(!DANGER.includes('<script>'), 'user-entered reason is escaped', DANGER.match(/alert/));
check(DANGER.includes('&lt;script&gt;'), 'escaped entity present');

/* ---- 9. open/close symmetry (matches the div.modal contract) ---- */
const open = extract('openRecommendationDetail');
const close = extract('closeRecommendationDetail');
check(/classList\.add\('active'\)/.test(open) && /modal\.hidden = false/.test(open),
  'open adds .active + clears hidden');
check(!/showModal\(/.test(open + close), 'open/close never call showModal() on a div');
check(/classList\.remove\('active'\)/.test(close) && /modal\.hidden = true/.test(close) &&
  /modal\.style\.display = 'none'/.test(close), 'close removes .active + hides');

/* ---- 10. Escape + backdrop wired ---- */
check(dcl.includes("e.key !== 'Escape'") && dcl.includes('closeRecommendationDetail()'),
  'Escape closes the recommendation modal');
check(dcl.includes("if (e.target === recModal) closeRecommendationDetail()"),
  'backdrop click closes the recommendation modal');

/* ---- 11. display-only: no Firebase writes anywhere in the detail path ---- */
const detailBlock = src.slice(src.indexOf('const ES_TYPE_LABELS'), src.indexOf('function getRuntimeState'));
check(!/\.ref\(|\.set\(|\.update\(|\.push\(|\.remove\(|firebase\.|\.transaction\(/.test(detailBlock),
  'detail path has no Firebase access at all');

/* ---- 12. mapping modal untouched + display-order index alignment ---- */
check((src.match(/openMappingEditModal\(/g) || []).length === 2,
  'openMappingEditModal still defined once + called once');
check(/energySavingRecs = recs;/.test(src) &&
  /list\.innerHTML = recs\.map\(renderEnergySavingItem\)/.test(src),
  'energySavingRecs order matches #energySavingList render order');
check(/energySavingCache\.recommendations\.slice\(\)/.test(src), 'cards still built from cached payload');

console.log(fail ? '\n' + fail + ' FAILED' : '\nall passed');
process.exitCode = fail ? 1 : 0;