"""
MathVision Main Launcher Script.
Run this script to launch the Streamlit web application.
Usage:
    python run.py
"""
import sys
import subprocess
from pathlib import Path

def check_environment() -> str | None:
    """Return an error message if this interpreter cannot run MathVision."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return (
            "TensorFlow is not installed for this Python "
            f"({sys.executable}).\n"
            "You are probably using the system Python instead of the project "
            "environment.\n\n"
            "Fix — run from the project directory with the venv interpreter:\n"
            '    .\\venv\\Scripts\\python.exe run.py\n'
            "or, after activating the venv:\n"
            "    .\\venv\\Scripts\\activate\n"
            "    python run.py"
        )
    try:
        import streamlit  # noqa: F401
        import streamlit_drawable_canvas  # noqa: F401
    except ImportError:
        return ("Streamlit or streamlit-drawable-canvas is not installed for "
                f"this Python ({sys.executable}).\n"
                "Fix — use the project environment:\n"
                '    .\\venv\\Scripts\\python.exe run.py')
    return None


def main():
    project_root = Path(__file__).parent.resolve()
    main_app = project_root / "app" / "main.py"

    print("=" * 60)
    print(" MathVision — Handwritten Math Expression Recognition & Solver")
    print("=" * 60)
    print(f"Project Directory: {project_root}")

    problem = check_environment()
    if problem is not None:
        print(f"\n[-] {problem}")
        sys.exit(2)
    print(f"Launching Streamlit Application: {main_app}\n")

    cmd = [sys.executable, "-m", "streamlit", "run", str(main_app)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[+] MathVision application stopped by user.")
    except Exception as e:
        print(f"\n[-] Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
