// Guards the mapping modal's initial visibility: it must be hidden on load and
// only opened by a .edit-meter click.
const fs = require('fs');
const path = require('path');
const dash = path.join(__dirname, '..', 'Dashboard Smart-Monitoring-System');
const html = fs.readFileSync(path.join(dash, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(dash, 'style.css'), 'utf8');
const js = fs.readFileSync(path.join(dash, 'script.js'), 'utf8');

let fail = 0;
const check = (cond, msg) => { console.log((cond ? 'ok  - ' : 'FAIL- ') + msg); if (!cond) fail++; };

const tag = html.match(/<div id="editLoadModal"[^>]*>/);
check(!!tag, '#editLoadModal exists as a div');
check(tag && /\bhidden\b/.test(tag[0]), 'markup starts with the hidden attribute');
check(tag && tag[0].includes('aria-hidden="true"'), 'markup starts aria-hidden');

// .modal must be display:none by default; only .active may show it
const modalRule = css.match(/\n\.modal \{([^}]*)\}/);
check(modalRule && /display:\s*none/.test(modalRule[1]), '.modal default is display:none');
check(!/\.modal:not\(\[hidden\]\)/.test(css), 'no .modal:not([hidden]) auto-show rule');
check(/\.modal\[hidden\]\s*\{\s*display:\s*none\s*!important;?\s*\}/.test(css), 'hidden attribute wins over CSS');
const activeRule = css.match(/\.modal\.active\s*\{([^}]*)\}/);
check(activeRule && /display:\s*flex/.test(activeRule[1]), '.modal.active is the show rule');

// open/close symmetry
const open = js.slice(js.indexOf('function openMappingEditModal'), js.indexOf('async function saveMappingFromModal'));
check(/modal\.classList\.add\('active'\)/.test(open), 'open adds .active');
check(/modal\.hidden = false/.test(open), 'open clears hidden');
check(!/showModal\(/.test(open), 'open never calls showModal() on a div');
const closeFn = open.slice(open.indexOf('function closeMappingEditModal'));
check(/modal\.classList\.remove\('active'\)/.test(closeFn), 'close removes .active');
check(/modal\.hidden = true/.test(closeFn), 'close sets hidden');
check(/modal\.style\.display = 'none'/.test(closeFn), 'close resets inline display');

// exactly one caller, inside the delegated edit handler
const calls = [...js.matchAll(/openMappingEditModal\(/g)].length;
check(calls === 2, `openMappingEditModal defined once + called once (found ${calls})`);
const callIdx = js.indexOf('openMappingEditModal(pzemKey, Number(pzemNumber));');
const handler = js.slice(js.lastIndexOf('dashboard.addEventListener("click"', callIdx), callIdx);
check(/closest\("\.edit-meter"\)/.test(handler), 'only call site is the .edit-meter delegated handler');
check(!/DOMContentLoaded|loadPzemMapping|renderDashboard\(\)|attachLiveListener/.test(handler),
  'no init-time caller of openMappingEditModal');

process.exitCode = fail ? 1 : 0;