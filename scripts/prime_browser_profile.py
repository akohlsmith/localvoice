"""One-off helper: copies selected cookies into LocalVoice's isolated
automation browser profile (tools/browser.py), so tool calls can access
specific logged-in sites without ever touching your everyday Brave
session. Does NOT copy Chromium's Cookies SQLite file wholesale — that's
brittle across Brave versions and copies everything, not just what you
mean to share. Uses Playwright's cookie API instead, so exactly what's
listed is what gets copied.

Two ways to supply cookies:

  --from-json cookies.json
      Load a manually-exported cookie file (e.g. from the "Cookie-Editor"
      browser extension: open it on the site you're logged into, export
      as JSON, save that file). You control exactly what's included, and
      this script never touches your real Brave profile at all.

  --from-profile "~/Library/Application Support/BraveSoftware/Brave-Browser" --domains example.com,news.ycombinator.com
      Pull cookies directly from your real Brave profile, filtered to the
      domain substrings you list. Requires Brave to be fully quit first —
      Chromium locks its profile directory while running, so this will
      fail with a clear error if Brave is still open.

Either way, every cookie about to be copied is printed before it's
written, so you can see exactly what's being shared with the automation
profile.

Usage:
    venv/bin/python scripts/prime_browser_profile.py --from-json cookies.json
    venv/bin/python scripts/prime_browser_profile.py \\
        --from-profile "~/Library/Application Support/BraveSoftware/Brave-Browser" \\
        --domains example.com,news.ycombinator.com
"""

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BRAVE_BINARY = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"


def load_from_json(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    # Normalize a Cookie-Editor-style export to Playwright's expected keys.
    cookies = []
    for c in raw:
        cookies.append(
            {
                "name": c["name"],
                "value": c["value"],
                "domain": c["domain"],
                "path": c.get("path", "/"),
                "expires": c.get("expirationDate", -1),
                "httpOnly": bool(c.get("httpOnly", False)),
                "secure": bool(c.get("secure", False)),
                "sameSite": (c.get("sameSite") or "Lax").capitalize(),
            }
        )
    return cookies


def load_from_profile(profile_dir: str, domains: list) -> list:
    expanded = str(Path(profile_dir).expanduser())
    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=expanded,
                executable_path=BRAVE_BINARY,
                headless=True,
            )
        except Exception as e:
            sys.exit(
                f"Could not open your real Brave profile at {expanded}: {e}\n"
                "Make sure Brave is fully quit first (Cmd+Q, not just closing the window) — "
                "Chromium locks its profile directory while running."
            )
        try:
            all_cookies = context.cookies()
        finally:
            context.close()
    return [c for c in all_cookies if any(d in c["domain"] for d in domains)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--automation-profile",
        default="~/.localvoice/brave-profile",
        help="Path to the automation profile (must match config.yaml's tools.browser.profile_dir).",
    )
    parser.add_argument("--from-json", help="Path to a Cookie-Editor style JSON export.")
    parser.add_argument("--from-profile", help="Path to your real Brave profile directory.")
    parser.add_argument(
        "--domains", help="Comma-separated domain substrings to include (used with --from-profile)."
    )
    args = parser.parse_args()

    if args.from_json:
        cookies = load_from_json(args.from_json)
    elif args.from_profile:
        if not args.domains:
            sys.exit("--domains is required with --from-profile")
        cookies = load_from_profile(args.from_profile, args.domains.split(","))
    else:
        sys.exit("Pass either --from-json or --from-profile (see --help).")

    if not cookies:
        print("No matching cookies found — nothing to prime.")
        return

    print(f"About to copy {len(cookies)} cookie(s) into {args.automation_profile}:")
    for c in cookies:
        print(f"  {c['domain']}{c['path']} :: {c['name']}")
    if input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    automation_dir = str(Path(args.automation_profile).expanduser())
    Path(automation_dir).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=automation_dir,
            executable_path=BRAVE_BINARY,
            headless=True,
        )
        try:
            context.add_cookies(cookies)
        finally:
            context.close()
    print("Done.")


if __name__ == "__main__":
    main()
