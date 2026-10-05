"""PZEM load mapping resolution used by export + monthly PDF.

One central source (config/pzem_mapping) with the static table as fallback;
display metadata only, never touching calculations.
"""

from __future__ import annotations

from ai import api_store, mapping, report_generator as rg
from ai.report_generator import ReportInput


def _use(mapping_data):
    api_store.set_db_get(lambda path: mapping_data if path == mapping.FIREBASE_MAPPING_PATH else {})
    mapping.refresh_pzem_load_mapping()


def _none(_path):
    raise RuntimeError("firebase unavailable")


def test_names_and_locations_resolve():
    _use({"pzem_1": {"loadName": "Fan 1", "location": "Classroom"}})
    assert mapping.meter_label(1) == "Fan 1"
    assert mapping.meter_label_with_id(1) == "Fan 1 (PZEM-1)"
    assert mapping.get_load_location("pzem_1") == "Classroom"
    assert mapping.get_pzem_display_id("pzem_1") == "PZEM-1"


def test_snake_case_keys_also_accepted():
    _use({"pzem_2": {"load_name": "Fan 2", "location": "Lab"}})
    assert mapping.meter_label(2) == "Fan 2"
    assert mapping.get_load_location("pzem_2") == "Lab"


def test_missing_mapping_falls_back_to_default_not_crash():
    _use({})
    assert mapping.meter_label(1) == "PZEM-1"
    assert mapping.get_load_location("pzem_1") == "Unassigned"


def test_firebase_unavailable_falls_back():
    api_store.set_db_get(_none)
    mapping.refresh_pzem_load_mapping()
    assert mapping.meter_label(4) == "PZEM-4"
    assert mapping.meter_label_with_id(4) == "PZEM-4"


def test_malformed_mapping_entries_ignored():
    _use({"pzem_1": "nonsense", "pzem_2": None, "pzem_3": {"loadName": "", "location": ""}})
    assert mapping.meter_label(1) == "PZEM-1"
    assert mapping.meter_label(3) == "PZEM-3"


def test_all_nine_meters_present():
    merged = mapping.get_pzem_load_mapping()
    assert [f"pzem_{i}" for i in range(1, 10)] == sorted(merged, key=lambda k: int(k.split("_")[1]))


def test_pdf_meter_wise_rows_use_mapping(tmp_path):
    _use({"pzem_1": {"loadName": "Fan 1", "location": "Classroom"}})
    report = rg.build_report(ReportInput(pzem_count=2), 0, 1, "monthly")
    assert [r["pzem"] for r in report["pzem_rows"]] == [1, 2]  # IDs drive the data
    # Labels are resolved at render time, so assert on what render_pdf emits.
    # Path must be under tmp_path: render_pdf really writes the file, so a
    # relative path would drop a PDF into the CWD (pytest runs in ai-backend).
    captured = _capture(rg.render_pdf, report, str(tmp_path / "render.pdf"))
    assert "Fan 1" in captured
    assert "PZEM-1" in captured          # ID stays traceable
    assert "Classroom" in captured      # location shown


def _capture(fn, *args, **kwargs):
    """Collect text/table cells render_pdf writes, without touching real output."""
    texts, tables = [], []
    orig_text, orig_table = rg._PDF.text, rg._PDF.table
    rg._PDF.text = lambda self, s, **k: (texts.append(str(s)), orig_text(self, s, **k))[1]
    rg._PDF.table = lambda self, h, r, w=None, **k: (
        tables.append(list(map(str, h))), tables.append([list(map(str, row)) for row in r]),
        orig_table(self, h, r, w, **k))[2]
    try:
        fn(*args, **kwargs)
    finally:
        rg._PDF.text, rg._PDF.table = orig_text, orig_table
    flat = [str(x) for t in tables for row in t for x in (row if isinstance(row, list) else [row])]
    return " | ".join(texts + flat)


def test_pdf_chart_labels_use_mapping():
    _use({"pzem_1": {"loadName": "Fan 1", "location": "Classroom"}})
    charts = rg._build_chart_data(rg.demo_input(2), 0, 2 ** 31 - 1, "monthly")
    labels = charts["pzem_energy"]["labels"]
    assert "Fan 1" in labels
    assert "PZEM 1" not in labels


def test_pdf_missing_mapping_still_renders(tmp_path):
    _use({})
    # output_dir is required: the default writes into the tracked dashboard
    # reports dir and would overwrite latest.pdf during a test run.
    res = rg.generate_monthly_report(data=rg.demo_input(2), output_dir=str(tmp_path))
    with open(res["pdf"], "rb") as fh:
        assert b"PZEM-1" in fh.read()