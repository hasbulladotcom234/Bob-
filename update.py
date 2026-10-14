"""Update this folder to the latest version of the project:

    python update.py

Downloads the newest code from GitHub and copies it over this folder.
Your own files are never touched: downloaded price data (data/), your
settings and keys (.env), the bot's saved position (state.json), its trade
journal (trades.csv) and anything else that isn't part of the project code.
If the add-ons the project needs have changed, it installs them too.
"""
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

REPO = "hasbulladotcom234/Bob-"
BRANCH = "claude/laughing-edison-5pcpzv"
URL = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"
HERE = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(HERE, ".version")


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def main():
    old_version = (read(VERSION_FILE) or b"unknown").decode().strip()
    print(f"Downloading the latest version from github.com/{REPO} ...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    new_version = archive.comment.decode().strip() or "unknown"  # GitHub stores the commit id here
    if new_version != "unknown" and new_version == old_version:
        print(f"Already up to date (version {new_version[:7]}).")
        return

    old_requirements = read(os.path.join(HERE, "requirements.txt"))
    root = archive.namelist()[0].split("/")[0] + "/"
    updated = 0
    for name in archive.namelist():
        if not name.startswith(root) or name.endswith("/"):
            continue
        relative = name[len(root):]
        target = os.path.join(HERE, *relative.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with archive.open(name) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        updated += 1
    with open(VERSION_FILE, "w") as f:
        f.write(new_version)
    print(f"Updated {updated} files: version {old_version[:7]} -> {new_version[:7]}.")

    if read(os.path.join(HERE, "requirements.txt")) != old_requirements:
        print("\nThe project's add-ons changed; installing them...\n")
        subprocess.call([sys.executable, os.path.join(HERE, "install.py")])
    print("\nDone. Your data and settings were left as they were.")


if __name__ == "__main__":
    main()
