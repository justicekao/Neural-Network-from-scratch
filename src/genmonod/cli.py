"""
cli.py

The actual console-command entry point (registered in pyproject.toml as
`genmonod-app`). Launches the Streamlit app via `streamlit run`, using
this package's own installed location to find app.py — this works
correctly whether you installed with `pip install -e .` or a normal
`pip install`.

NEEDS YOUR INPUT: nothing.
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:])


if __name__ == "__main__":
    main()
