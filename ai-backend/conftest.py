import os
import sys

# Ensure the `ai` package is importable when pytest is run from the repository
# root (not just from inside this directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
