import io
import os
import subprocess

import update


def github_zip(repo_dir):
    """A ZIP shaped like GitHub's branch download (top folder + commit id comment)."""
    out = subprocess.run(["git", "archive", "--format=zip", "--prefix=Bob--branch/", "HEAD"],
                         cwd=repo_dir, capture_output=True, check=True).stdout
    return out


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_update_replaces_code_but_keeps_user_files(tmp_path, monkeypatch, capsys):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blob = github_zip(repo)
    monkeypatch.setattr(update.urllib.request, "urlopen", lambda url, timeout=0: FakeResponse(blob))
    monkeypatch.setattr(update, "HERE", str(tmp_path))
    monkeypatch.setattr(update, "VERSION_FILE", str(tmp_path / ".version"))
    calls = []
    monkeypatch.setattr(update.subprocess, "call", lambda cmd: calls.append(cmd))

    (tmp_path / "data" / "alpaca" / "1h").mkdir(parents=True)
    (tmp_path / "data" / "alpaca" / "1h" / "BTC-USD.csv").write_text("my downloaded prices")
    (tmp_path / ".env").write_text("EXCHANGE_API_KEY=PKSECRET")
    (tmp_path / "state.json").write_text('{"position": null}')
    (tmp_path / "main.py").write_text("old code")
    (tmp_path / "requirements.txt").write_text("old requirements")

    update.main()
    assert "Updated" in capsys.readouterr().out
    committed = subprocess.run(["git", "show", "HEAD:main.py"], cwd=repo, capture_output=True, text=True).stdout
    assert (tmp_path / "main.py").read_text() == committed
    assert (tmp_path / "tradingbot" / "cleaning.py").exists()
    assert (tmp_path / "data" / "alpaca" / "1h" / "BTC-USD.csv").read_text() == "my downloaded prices"
    assert (tmp_path / ".env").read_text() == "EXCHANGE_API_KEY=PKSECRET"
    assert (tmp_path / "state.json").read_text() == '{"position": null}'
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert (tmp_path / ".version").read_text() == head
    assert len(calls) == 1 and calls[0][-1].endswith("install.py")   # requirements changed -> reinstall

    update.main()                                                    # second run: nothing to do
    assert "Already up to date" in capsys.readouterr().out
    assert len(calls) == 1
