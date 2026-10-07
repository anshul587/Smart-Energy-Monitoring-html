/* Focused tests for the load-mapping save flow (script.js):
   A. successful Firebase save
   B. permission-denied save (no false local persistence, useful reason)
   C. auth-not-ready save (write never attempted)
   D. correct path: config/pzem_mapping/pzem_N
   E. payload schema (only load_name / location, pzem key immutable)
   F. reload persistence (loadPzemMapping wired into onAuthStateChanged)
   G. modal open/close is guarded by tools/check_mapping_modal_state.js
   No Firebase access — the SDK is mocked inside a vm. No writes. */
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
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function extract(name) {
  const needle = 'function ' + name + '(';
  const match = src.indexOf(needle);
  if (match < 0) throw new Error('missing function ' + name);
  const start = match > 6 && src.slice(match - 6, match) === 'async ' ? match - 6 : match;
  let i = src.indexOf('{', start), depth = 0;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) return src.slice(start, i + 1);
  }
  throw new Error('unbalanced braces in ' + name);
}

const state = {
  authReady: true,
  updateError: null,
  refPath: null,
  payload: null,
  loadedValue: null,
};
const fb = {
  auth: () => ({ get currentUser() { return state.authReady ? { uid: 'anon-1' } : null; } }),
  database: () => ({
    ref: (p) => ({
      update: async (m) => {
        state.refPath = p;
        state.payload = m;
        if (state.updateError) throw state.updateError;
      },
      once: async () => ({ val: () => state.loadedValue }),
    }),
  }),
};

const ctx = vm.createContext({ PZEM_MAPPING_LOADED: {}, firebase: fb, console });
vm.runInContext(
  [extract('savePzemMapping'), extract('mappingSaveFailureReason'), extract('loadPzemMapping')].join('\n'),
  ctx
);

const MAP = { pzem_3: { load_name: 'light 1', location: 'lab 1' } };
const reset = () => {
  state.authReady = true; state.updateError = null; state.refPath = null; state.payload = null; state.loadedValue = null;
  ctx.PZEM_MAPPING_LOADED = {};
};

/* ---- A. successful save ---- */
(async () => {
  reset();
  const result = await ctx.savePzemMapping(MAP);
  check(result && result.ok === true, 'A. save resolves ok:true', result);
  check(eq(state.payload, MAP), 'A. exact payload was sent to Firebase', state.payload);
  check(eq(ctx.PZEM_MAPPING_LOADED, MAP), 'A. local state merged after success', ctx.PZEM_MAPPING_LOADED);

  /* ---- D. exact path ---- */
  check(state.refPath === 'config/pzem_mapping', 'D. write path is exactly config/pzem_mapping', state.refPath);
  check(state.payload && Object.keys(state.payload).length === 1 && Object.keys(state.payload)[0] === 'pzem_3',
    'D. key is pzem_3 so full path = config/pzem_mapping/pzem_3');

  /* ---- E. payload schema ---- */
  const child = state.payload && state.payload.pzem_3;
  check(child && Object.keys(child).sort().join(',') === 'load_name,location',
    'E. entry has exactly load_name + location', child);
  check(child.load_name === 'light 1' && child.location === 'lab 1', 'E. values preserved', child);

  /* ---- B. permission-denied save ---- */
  reset();
  state.updateError = Object.assign(new Error('Permission denied'), { code: 'PERMISSION_DENIED' });
  ctx.PZEM_MAPPING_LOADED = { pzem_1: { load_name: 'Fan 1', location: 'Room' } };
  const before = JSON.parse(JSON.stringify(ctx.PZEM_MAPPING_LOADED));
  const denied = await ctx.savePzemMapping(MAP);
  check(denied && denied.ok === false && denied.code === 'PERMISSION_DENIED',
    'B. denied save returns ok:false + PERMISSION_DENIED', denied);
  check(eq(ctx.PZEM_MAPPING_LOADED, before), 'B. local state NOT updated on failure (no false persistence)');
  const reason = ctx.mappingSaveFailureReason(denied);
  check(/config\/pzem_mapping/.test(reason) && /permission|rule/i.test(reason),
    'B. failure reason is useful (names path + rules)', reason);

  /* ---- C. auth-not-ready save ---- */
  reset();
  state.authReady = false;
  const notReady = await ctx.savePzemMapping(MAP);
  check(notReady && notReady.ok === false && notReady.code === 'auth-not-ready',
    'C. auth-not-ready returns ok:false + auth-not-ready', notReady);
  check(state.refPath === null, 'C. no Firebase write is attempted before auth', state.refPath);
  check(eq(ctx.PZEM_MAPPING_LOADED, {}), 'C. local state not mutated', ctx.PZEM_MAPPING_LOADED);
  check(/sign-in/i.test(ctx.mappingSaveFailureReason(notReady)), 'C. failure reason says sign-in not finished');

  /* ---- F. reload persistence ---- */
  reset();
  state.loadedValue = MAP;
  const loaded = await ctx.loadPzemMapping();
  check(loaded && loaded.pzem_3.load_name === 'light 1' && loaded.pzem_3.location === 'lab 1',
    'F. loadPzemMapping repopulates local state from Firebase', loaded);
  const authBlock = src.slice(src.indexOf('firebase.auth().onAuthStateChanged'),
    src.indexOf('$("powerRange")'));
  check(authBlock.includes('loadPzemMapping()'), 'F. loadPzemMapping() is invoked after auth (reload persists)');
  check((src.match(/loadPzemMapping\(/g) || []).length === 2,
    'F. loadPzemMapping defined once + invoked once (no duplicate loaders)');

  /* ---- F2/F3/F4. refresh-persistence regression guards (T2) ----
     The live bug was: save succeeded, Firebase held config/pzem_mapping/pzem_5,
     but the deployed build never called loadPzemMapping() after auth, so a
     refresh reset PZEM_MAPPING_LOADED to {} and static defaults won. */
  check(/loadPzemMapping\(\)\s*\.then\s*\(/.test(authBlock) &&
      /renderDashboard/.test(authBlock.slice(authBlock.indexOf('loadPzemMapping'))),
    'F2. auth block re-renders ONLY after loadPzemMapping resolves (.then chaining)',
    (authBlock.match(/loadPzemMapping\([^\n]*/) || [])[0]);
  const writes = src.match(/PZEM_MAPPING_LOADED\s*=(?!=)/g) || [];
  check(writes.length === 3,
    'F3. PZEM_MAPPING_LOADED written only by decl + load + save (no renderer clobbers it)', writes.length);
  const preferRemote = /\(PZEM_MAPPING_LOADED&&PZEM_MAPPING_LOADED\[k\]\)\|\|\(PZEM_LOAD_MAPPING&&PZEM_LOAD_MAPPING\[k\]\)/g;
  check((src.match(preferRemote) || []).length === 2,
    'F4. both readers prefer PZEM_MAPPING_LOADED over the static default table');

  /* ---- G. modal open/close behavior is covered by check_mapping_modal_state.js ---- */
  check(fs.existsSync(path.join(__dirname, 'check_mapping_modal_state.js')),
    'G. modal open/close guarded by check_mapping_modal_state.js');

  console.log(fail ? '\n' + fail + ' FAILED' : '\nall passed');
  process.exitCode = fail ? 1 : 0;
})();