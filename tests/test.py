import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "src" / "Utils"

sys.path.append( str(PROJECT_DIR))
print(sys.path)
