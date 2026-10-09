"""Install the project's add-ons on any computer:

    python install.py

Like `pip install -r requirements.txt`, but it never tries to compile
anything (which needs Microsoft's C++ tools on Windows). If an add-on has no
ready-made version for this computer, as happens on Windows on ARM laptops,
it installs everything else, skips pieces the project doesn't need, and then
checks that the project actually works.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
OPTIONAL = {"pyarrow": "data will be stored as CSV instead of Parquet (works the same, slightly slower)"}
CHECK = "import ccxt, pandas, numpy, dotenv; ccxt.alpaca()"


def pip_install(*args):
    """Run pip with ready-made builds only. Returns (succeeded, output)."""
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].strip().lower()


def read_requirements():
    with open(REQUIREMENTS) as f:
        lines = [line.split("#")[0].strip() for line in f]
    return [line for line in lines if line]


def ccxt_dependencies():
    """ccxt's declared dependencies that apply to this computer."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    # Ask a fresh interpreter, so a just-installed ccxt is visible.
    out = subprocess.run([sys.executable, "-c",
                          "from importlib.metadata import requires; print('\\n'.join(requires('ccxt') or []))"],
                         capture_output=True, text=True, check=True).stdout
    deps = []
    for line in out.splitlines():
        req = Requirement(line)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            deps.append(f"{req.name}{req.specifier}")
    return deps


def fail(message, output=""):
    if output:
        print("\n".join(output.strip().splitlines()[-15:]))
    print(f"\n{message}")
    sys.exit(1)


def main():
    print(f"Python {sys.version.split()[0]} on {sys.platform} ({os.environ.get('PROCESSOR_ARCHITECTURE', '')})")
    print("Installing add-ons (this can take a few minutes)...")
    ok, output = pip_install("-r", REQUIREMENTS)

    if not ok:
        print("Some add-ons have no ready-made version for this computer. Installing piece by piece...")
        reqs = read_requirements()
        ccxt_req = next(r for r in reqs if package_name(r) == "ccxt")
        others = [r for r in reqs if package_name(r) != "ccxt"]

        ok, output = pip_install(*others)
        if not ok:
            fail("Couldn't install the core add-ons (see above). Paste this output to get help.", output)
        ok, output = pip_install("--no-deps", ccxt_req)
        if not ok:
            fail("Couldn't install ccxt (see above). Paste this output to get help.", output)

        deps = ccxt_dependencies()
        skipped = []
        if not pip_install(*deps)[0]:
            for dep in deps:
                if not pip_install(dep)[0]:
                    skipped.append(dep)
        if skipped:
            print("Skipped (no version for this computer; only used as optional speed-ups): "
                  + ", ".join(skipped))

    for pkg, consequence in OPTIONAL.items():
        if not pip_install(pkg)[0]:
            print(f"Optional: {pkg} isn't available for this computer, so {consequence}.")

    check = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
    if check.returncode != 0:
        fail("Installed, but the project can't start (see above). Paste this output to get help.",
             check.stderr)
    print("\nAll set. Next:  python main.py data update")


if __name__ == "__main__":
    main()
