"""
run_stage2_report.py
---------------------
Runs the Stage 2 preprocessing pipeline against your REAL Firebase project
(using whatever's in .env) and prints the full 9-PZEM data-discovery
report. Nothing about which meters have data is assumed — it asks the
data loader for all 9 and reports whatever actually comes back.

Usage (from the project root, with your venv active and .env filled in):
    python run_stage2_report.py
"""

from ai.preprocessing import run_preprocessing_pipeline, format_report, print_debug_tracebacks

if __name__ == "__main__":
    results = run_preprocessing_pipeline()
    print(format_report(results))
    print_debug_tracebacks(results)
