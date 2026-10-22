"""Command centre: every project command in a numbered menu.

Double-click menu.bat (Windows) or menu.command (Mac), or run:  python menu.py

After each command its output is saved to results.txt, and you can open it
in a text editor to copy and paste.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.txt")


def run(args, input_fn=input):
    """Run a command, show its output live, save it to results.txt."""
    cmd = [sys.executable, "-u", *args]
    print(f"\n> python {' '.join(args)}\n")
    with open(RESULTS, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="")
            out.write(line)
        proc.wait()
    print(f"\n(Saved to results.txt)")
    if input_fn("Open it to copy? [y/N] ").strip().lower() == "y":
        open_file(RESULTS)


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # opens in Notepad (or your default text editor)
    elif sys.platform == "darwin":
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])


def signal_names():
    import importlib
    import inspect
    sys.path.insert(0, HERE)
    module = importlib.import_module("my_signals")
    importlib.reload(module)  # pick up edits made while the menu is open
    return [name for name, fn in inspect.getmembers(module, inspect.isfunction)
            if fn.__module__ == module.__name__ and not name.startswith("_")]


def choose(title, options, input_fn=input):
    """Show numbered options; return the chosen one or None."""
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    pick = input_fn("Number (Enter = back): ").strip()
    if pick.isdigit() and 1 <= int(pick) <= len(options):
        return options[int(pick) - 1]
    return None


def desktop_shortcut():
    """Put a 'Bob Trading' launcher on the Desktop that opens this menu."""
    home = os.path.expanduser("~")
    candidates = [os.path.join(home, "OneDrive", "Desktop"), os.path.join(home, "Desktop")]
    desktop = next((d for d in candidates if os.path.isdir(d)), None)
    if desktop is None:
        print("Couldn't find your Desktop folder.")
        return
    if sys.platform.startswith("win"):
        path = os.path.join(desktop, "Bob Trading.bat")
        with open(path, "w") as f:
            f.write(f'@echo off\ncd /d "{HERE}"\npython menu.py\npause\n')
    else:
        path = os.path.join(desktop, "Bob Trading.command")
        with open(path, "w") as f:
            f.write(f'#!/bin/bash\ncd "{HERE}"\npython3 menu.py\n')
        os.chmod(path, 0o755)
    print(f"Done: double-click 'Bob Trading' on your Desktop to open this menu from now on.\n({path})")


MENU = [
    ("Get the latest version of the project", lambda i: run(["update.py"], i)),
    ("Download new price data", lambda i: run(["main.py", "data", "update"], i)),
    ("Check data quality", lambda i: run(["main.py", "data", "check"], i)),
    ("How every coin performed (benchmarks)", lambda i: run(["main.py", "data", "benchmarks"], i)),
    ("Research: test one of my ideas", None),
    ("Run a lesson", None),
    ("Paper trading: check status (never trades)", lambda i: run(["main.py", "status"], i)),
    ("Paper trading: connect my Alpaca paper account", lambda i: run(["main.py", "setup"], i)),
    ("Put a shortcut to this menu on my Desktop", lambda i: desktop_shortcut()),
]


def research(input_fn):
    names = signal_names()
    if not names:
        print("No ideas in my_signals.py yet.")
        return
    name = choose("Which idea? (they live in my_signals.py)", names, input_fn)
    if name:
        run(["main.py", "research", name], input_fn)


def lesson(input_fn):
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "learn", "*.py")))
    name = choose("Which lesson? (open the file in learn\\ to read the code)", files, input_fn)
    if name:
        run([os.path.join("learn", name)], input_fn)


def main(input_fn=input):
    while True:
        print("\n" + "=" * 52)
        print("  BOB TRADING - COMMAND CENTRE")
        print("=" * 52)
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  {i}. {label}")
        print("  0. Exit")
        pick = input_fn("\nChoose a number: ").strip()
        if pick in ("0", "q"):
            return
        if not pick.isdigit() or not 1 <= int(pick) <= len(MENU):
            if pick:
                print("Type one of the numbers above.")
            continue
        label, action = MENU[int(pick) - 1]
        if label.startswith("Research"):
            research(input_fn)
        elif label.startswith("Run a lesson"):
            lesson(input_fn)
        else:
            action(input_fn)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
