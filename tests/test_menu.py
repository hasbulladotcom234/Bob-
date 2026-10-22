import menu


def test_menu_routes_choices(monkeypatch):
    calls = []
    monkeypatch.setattr(menu, "run", lambda args, input_fn=None: calls.append(args))
    keys = iter(["7", "1",          # research -> first idea
                 "8", "1",          # lesson -> first lesson
                 "2",               # data update
                 "", "abc",         # stray Enter / typo: menu stays open
                 "0"])
    menu.main(input_fn=lambda prompt="": next(keys))
    first_idea = menu.signal_names()[0]
    assert calls[0] == ["main.py", "research", first_idea]
    assert calls[1][0].endswith("01_btc_stats.py")
    assert calls[2] == ["main.py", "data", "update"]
    assert len(calls) == 3


def test_run_shows_and_saves_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(menu, "RESULTS", str(tmp_path / "results.txt"))
    menu.run(["-c", "print('hello from the project')"], input_fn=lambda prompt="": "n")
    assert "hello from the project" in capsys.readouterr().out
    assert "hello from the project" in (tmp_path / "results.txt").read_text()


def test_desktop_shortcut(tmp_path, monkeypatch):
    monkeypatch.setattr(menu.os.path, "expanduser", lambda p: str(tmp_path))
    (tmp_path / "Desktop").mkdir()
    menu.desktop_shortcut()
    made = list((tmp_path / "Desktop").iterdir())
    assert len(made) == 1 and menu.HERE in made[0].read_text()


def test_shortcut_flag_creates_launcher_then_opens_menu(tmp_path):
    import os
    import subprocess
    import sys
    (tmp_path / "Desktop").mkdir()
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    out = subprocess.run([sys.executable, menu.__file__, "--shortcut"], input="0\n", capture_output=True,
                         text=True, env=env).stdout
    assert "Bob Trading" in out and "COMMAND CENTRE" in out
    assert len(list((tmp_path / "Desktop").iterdir())) == 1
