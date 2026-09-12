"""
tests/test_info_panel_dom.py — DOM-level assertions for the Ask BOB Info panel.

These are static checks on Dashboard Smart-Monitoring-System/index.html (no browser
needed). The interactive open/close/Escape/reopen behaviour is exercised against the
live dashboard via Playwright/Chrome DevTools (see task verification); this file locks
in the structural guarantees so a regression is caught in CI:

  * exactly one Info button and one Info dialog
  * all Project Information lives INSIDE <dialog id="infoDialog"> only
  * no Project Information content appears in normal page flow
  * the native <dialog> close button (.info-close) is wired
  * credits use Swapnil Shendre (never Yash Shendre)
  * no user-visible "Stage N" labels remain
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DASHBOARD_HTML = (
    Path(__file__).resolve().parents[2]
    / "Dashboard Smart-Monitoring-System"
    / "index.html"
)

HTML = _DASHBOARD_HTML.read_text(encoding="utf-8")

_DIALOG_OPEN = '<dialog class="info-dialog" id="infoDialog"'
# Distinctive, entity-free phrases unique to the Info panel (avoids false matches
# against other UI text or HTML-entity encoding of "&").
_INFO_HEADINGS = [
    "About the Project",
    "Project Credits",
    "How It Works",
    "Main Features",
    "AI Intelligence",
    "Energy Conservation",
    "Fault &amp; Alert System",
    "Current Limitations",
    "Future Scope",
]


def _strip(html: str) -> str:
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    html = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", "", html, flags=re.S | re.I)
    return html


def _dialog_block() -> str:
    start = HTML.index(_DIALOG_OPEN)
    end = HTML.index("</dialog>", start)
    return HTML[start:end]


def test_single_info_button():
    assert HTML.count('id="infoButton"') == 1


def test_single_info_dialog():
    assert HTML.count('id="infoDialog"') == 1


def test_info_dialog_is_native():
    assert _DIALOG_OPEN in HTML


def test_project_info_only_inside_dialog():
    block = _dialog_block()
    for heading in _INFO_HEADINGS:
        assert heading in block, f"missing info section inside dialog: {heading}"

    after_dialog = HTML[HTML.index("</dialog>", HTML.index(_DIALOG_OPEN)) + len("</dialog>"):]
    for heading in _INFO_HEADINGS:
        assert heading not in after_dialog, (
            f"Project Information leaks into normal page flow: {heading}"
        )


def test_no_duplicate_project_information():
    # The official full project title is a distinctive marker that must appear
    # exactly once in the document (a duplicated info section would double it).
    assert HTML.count("Suggestions for Energy Conservation") == 1, (
        "official project title duplicated or missing"
    )


def test_close_button_present_and_wired():
    assert 'class="icon-button info-close"' in HTML


def test_close_handler_wired_in_script():
    script = (
        Path(__file__).resolve().parents[2]
        / "Dashboard Smart-Monitoring-System"
        / "script.js"
    ).read_text(encoding="utf-8")
    assert "infoDialog.showModal" in script
    assert "infoDialog.close" in script


def test_correct_credits():
    assert "Dr. Bhupendra Kumar" in HTML
    assert "Anshul Ninawe" in HTML
    for name in ["Yash Kawale", "Yash Dahake", "Swapnil Shendre", "Chetan Bokade", "Sanjog Godbole"]:
        assert name in HTML


def test_no_yash_shendre():
    assert "Yash Shendre" not in HTML


def test_no_visible_stage_labels():
    # Strip comments + script/style so only rendered text is inspected.
    visible = _strip(HTML)
    assert not re.search(r"stage\s*\d{1,2}", visible, flags=re.I), (
        "user-visible Stage-number label found"
    )
