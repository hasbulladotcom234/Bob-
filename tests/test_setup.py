import os

from tradingbot import setup_wizard
from tradingbot.setup_wizard import run_setup, update_env_file


def test_update_env_file_keeps_other_lines(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nTIMEFRAME=4h\nMODE=paper\n")
    update_env_file(str(p), {"MODE": "sandbox", "EXCHANGE_API_KEY": "PK1"})
    assert p.read_text() == "# comment\nTIMEFRAME=4h\nMODE=sandbox\nEXCHANGE_API_KEY=PK1\n"


def test_setup_saves_only_after_successful_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok = run_setup(input_fn=lambda _: "PKTEST", secret_fn=lambda _: "s",
                   check_fn=lambda k, s: {"cash": 100000.0, "coin": 0.0, "price": 60000.0})
    assert ok and "MODE=sandbox" in open(".env").read()


def test_setup_saves_nothing_when_keys_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def fail(k, s):
        raise RuntimeError("unauthorized")
    assert not run_setup(input_fn=lambda _: "PKTEST", secret_fn=lambda _: "s", check_fn=fail)
    assert not os.path.exists(".env")
