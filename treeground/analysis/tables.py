import runpy
import sys
from pathlib import Path

# post step for queue requests: regenerate the paper tables (scripts/make_tables.py) from results/metrics
if __name__ == "__main__":
    script = Path(__file__).resolve().parents[2] / "scripts" / "make_tables.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
