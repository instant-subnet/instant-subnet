#!/usr/bin/env python
"""End-to-end customer check, driven by the official OpenAI SDK.

The product claim is "change base_url and your OpenAI code works". curl cannot
test that claim -- it prints whatever bytes arrive, while an SDK *parses* them
and will reject a response curl was happy with. So this uses the real client.

It walks the customer's path and shows the request and the response at each
step, so a failure tells you which link broke rather than just "it didn't work".

    # against the local staging gateway
    scripts/e2e_sdk.py --api-key isk_...

    # mint a fresh key through the dashboard first, then use it: the full loop
    scripts/e2e_sdk.py --mint --control http://127.0.0.1:8080 \
        --email you@example.com --password ...

    # against production
    scripts/e2e_sdk.py --base-url https://api.instantsubnet.com/v1 --api-key isk_...

Exits non-zero if any check fails, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.request

PASS, FAIL, INFO = "  PASS", "  FAIL", "     ·"
_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def mint_key(control: str, email: str, password: str) -> tuple[str, str]:
    """Sign in to the dashboard and create a key, as an operator would."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def call(method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            control.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(req, timeout=30) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    status, _ = call("POST", "/api/session", {"email": email, "password": password})
    if status != 200:
        sys.exit(f"sign-in failed ({status}) — check --email/--password")
    status, created = call("POST", "/api/keys", {"name": "e2e_sdk"})
    if status != 201:
        sys.exit(f"key creation failed ({status}): {created.get('message')}")
    return created["key"], created["id"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    p.add_argument("--api-key", default=os.environ.get("INSTANT_API_KEY", ""))
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--prompt", default="In one short sentence: what is Bittensor?")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="reasoning is drawn from this budget; too small returns empty content",
    )
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--mint", action="store_true", help="create a key via the dashboard")
    p.add_argument("--control", default="http://127.0.0.1:8080")
    p.add_argument("--email", default=os.environ.get("INSTANT_EMAIL", ""))
    p.add_argument("--password", default=os.environ.get("INSTANT_PASSWORD", ""))
    args = p.parse_args(argv)

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("needs the openai package:  uv pip install openai")

    key_id = None
    if args.mint:
        print("STEP 1-2  dashboard: sign in and mint a key")
        args.api_key, key_id = mint_key(args.control, args.email, args.password)
        check(bool(args.api_key), "key minted", f"{args.api_key[:12]}…{args.api_key[-4:]}")
    if not args.api_key:
        sys.exit("no API key: pass --api-key, set INSTANT_API_KEY, or use --mint")

    client = OpenAI(
        api_key=args.api_key, base_url=args.base_url, max_retries=0, timeout=180
    )
    messages = [{"role": "user", "content": args.prompt}]
    request_body = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }

    print(f"\nSTEP 3    request  ->  {args.base_url}")
    print(json.dumps(request_body, indent=2))

    print("\nSTEP 4-7  non-streaming")
    started = time.perf_counter()
    try:
        raw = client.chat.completions.with_raw_response.create(**request_body)
    except Exception as exc:  # noqa: BLE001
        check(False, "request accepted", f"{type(exc).__name__}: {exc}")
        return 1
    completion = raw.parse()
    elapsed_ms = (time.perf_counter() - started) * 1000
    headers = {k.lower(): v for k, v in raw.headers.items()}
    message = completion.choices[0].message

    print("\n  response:")
    print(json.dumps(completion.model_dump(), indent=2, default=str)[:1400])
    print("\n  instant headers:")
    for name in sorted(h for h in headers if h.startswith("x-instant")):
        value = headers[name]
        print(f"{INFO}  {name}: {value if len(value) < 60 else f'<{len(value)} chars>'}")

    check(completion.model == args.model, "served the pinned model", completion.model)
    check(bool(message.content), "content returned", completion.choices[0].finish_reason)
    check(
        completion.choices[0].finish_reason == "stop",
        "completed, not truncated",
        "raise --max-tokens if this fails: reasoning shares the budget",
    )
    # Provenance and proof-of-work. A response without a receipt was not counted
    # by the platform and cannot be scored, so its absence is a real failure.
    check("x-instant-miner-uid" in headers, "names the miner that served it",
          f"uid {headers.get('x-instant-miner-uid')}")
    check("x-instant-receipt-sig" in headers, "carries a signed receipt")
    print(f"{INFO}  total {elapsed_ms:.0f} ms")

    print("\nSTEP 3b   streaming")
    started = time.perf_counter()
    first_word_ms = None
    text, chunks, reasoning_chunks = "", 0, 0
    try:
        stream = client.chat.completions.create(**request_body, stream=True)
        for chunk in stream:
            chunks += 1
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if getattr(delta, "reasoning", None):
                reasoning_chunks += 1
            if delta.content:
                if first_word_ms is None:
                    first_word_ms = (time.perf_counter() - started) * 1000
                text += delta.content
    except Exception as exc:  # noqa: BLE001
        check(False, "stream completed", f"{type(exc).__name__}: {exc}")
        return 1

    total_s = time.perf_counter() - started
    check(bool(text), "streamed content assembled", repr(text[:60]))
    # The number a reader experiences. gpt-oss streams its reasoning first, so
    # time-to-first-token and time-to-first-word are different measurements and
    # only the second one is what "fast" means to a customer.
    if first_word_ms is not None:
        rate = len(text) / 4 / max(total_s - first_word_ms / 1000, 1e-6)
        print(f"{INFO}  first word {first_word_ms:.0f} ms · ~{rate:.0f} tok/s · "
              f"{chunks} chunks ({reasoning_chunks} reasoning)")

    print("\nSTEP 3c   a bad key must be refused")
    try:
        OpenAI(api_key="isk_definitely_invalid", base_url=args.base_url,
               max_retries=0).chat.completions.create(**request_body)
        check(False, "invalid key rejected", "it was accepted")
    except Exception as exc:  # noqa: BLE001
        check(type(exc).__name__ == "AuthenticationError", "invalid key rejected",
              type(exc).__name__)

    if key_id:
        print(f"\n{INFO}  minted key id {key_id} — revoke it in the dashboard when done")

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): " + ", ".join(_failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
