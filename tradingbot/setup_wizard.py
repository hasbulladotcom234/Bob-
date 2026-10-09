"""Interactive setup and health check, so getting to paper trading takes a
couple of minutes and no file editing."""
import getpass
import os

from .data import make_exchange

ENV_FILE = ".env"


def update_env_file(path: str, values: dict) -> None:
    """Set KEY=value lines in a .env file, keeping any other lines as they are."""
    lines = open(path).read().splitlines() if os.path.exists(path) else []
    remaining = dict(values)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in remaining.items()]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def check_alpaca_paper(api_key: str, api_secret: str, symbol: str = "BTC/USD") -> dict:
    """Log in to the Alpaca paper account and return what we can see."""
    ex = make_exchange("alpaca", api_key, api_secret, sandbox=True)
    free = ex.fetch_balance().get("free", {})
    base, quote = symbol.split("/")
    return {"cash": float(free.get(quote) or 0.0), "coin": float(free.get(base) or 0.0),
            "price": float(ex.fetch_ticker(symbol)["last"])}


def run_setup(input_fn=input, secret_fn=getpass.getpass, check_fn=check_alpaca_paper) -> bool:
    print("Alpaca paper trading setup\n")
    print("1. Log in at https://app.alpaca.markets")
    print("2. Make sure the account switcher (top left) says PAPER, not Live")
    print("3. On the Home page, find 'API Keys' and click Generate / Regenerate")
    print("   Copy both the Key and the Secret (the secret is shown only once)\n")
    key = input_fn("Paste your paper API Key: ").strip()
    secret = secret_fn("Paste your paper Secret (hidden as you type): ").strip()
    if not key or not secret:
        print("Both are needed. Run setup again when you have them.")
        return False
    print("\nChecking the keys with Alpaca...")
    try:
        info = check_fn(key, secret)
    except Exception as err:
        print(f"Alpaca rejected the keys or couldn't be reached: {err}")
        print("Double-check you copied the PAPER key and secret. Nothing was saved.")
        if key.startswith("AK"):
            print("(That key looks like a live-account key; paper keys usually start with PK.)")
        return False

    update_env_file(ENV_FILE, {"EXCHANGE_ID": "alpaca", "SYMBOL": "BTC/USD", "MODE": "sandbox",
                               "EXCHANGE_API_KEY": key, "EXCHANGE_API_SECRET": secret})
    print(f"Connected. Paper cash: ${info['cash']:,.2f} | BTC held: {info['coin']:.6f} | "
          f"BTC price: ${info['price']:,.2f}")
    print(f"Saved to {ENV_FILE} (kept private: it's in .gitignore).\n")
    print("Start the bot with:  python main.py run")
    return True
