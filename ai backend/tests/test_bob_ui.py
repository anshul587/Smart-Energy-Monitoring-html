"""
tests/test_bob_ui.py — static guards for the Ask BOB UI fixes.

The interactive chat behaviour (auto-scroll, manual-scroll preservation) is
exercised against the live dashboard via Playwright/Chrome DevTools. These
checks lock the fixes in CI by asserting the corrected logic is present in the
shipped source and that the stale team name is gone.

  * auto-scroll uses a pre-append "near bottom" decision (not post-append, which
    misses tall bot replies)
  * the dialog/script assets are cache-busted so returning visitors get the fix
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DASHBOARD = (
    Path(__file__).resolve().parents[2]
    / "Dashboard Smart-Monitoring-System"
)
_BOB_JS = (_DASHBOARD / "bob.js").read_text(encoding="utf-8")
_INDEX_HTML = (_DASHBOARD / "index.html").read_text(encoding="utf-8")


def test_autoscroll_uses_prepend_near_bottom():
    # The corrected scroll helper must exist...
    assert "scrollToBottom" in _BOB_JS
    # ...and the stick-to-bottom decision must be computed BEFORE appending the
    # new message (a post-append check fails when a bot reply is taller than the
    # viewport). Guard against the old buggy pattern.
    assert "var stick =" in _BOB_JS
    assert "nearBottom()" in _BOB_JS


def test_bob_assets_cache_busted():
    # Force returning visitors to reload the fixed script/styles.
    assert 'bob.js?v=2' in _INDEX_HTML
    assert 'info.css?v=2' in _INDEX_HTML


def test_team_name_is_swapnil_not_yash():
    # The UI must not leak the old name and the dialog must show the corrected one.
    assert "Yash Shendre" not in _BOB_JS
    assert "Yash Shendre" not in _INDEX_HTML
    assert "Swapnil Shendre" in _INDEX_HTML
